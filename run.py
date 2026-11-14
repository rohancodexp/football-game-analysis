import argparse, json, csv, re
from pathlib import Path
from collections import Counter
import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.models import resnet18, ResNet18_Weights
from ultralytics import YOLO
from output_schema import (
    ACCOUNTING_VERSION,
    PLAYER_ID_UNRESOLVED,
    SCHEMA_VERSION,
    TEAM_UNKNOWN,
    build_detection_id,
    normalize_team_label,
    schema_descriptor,
    validate_accounting_record,
)


def unit(x):
    return x / max(float(np.linalg.norm(x)), 1e-8)

class Encoder:
    def __init__(self, device):
        self.device = device
        w = ResNet18_Weights.DEFAULT
        self.transform = w.transforms()
        self.model = resnet18(weights=w)
        self.model.fc = torch.nn.Identity()
        self.model.to(device).eval()

    @torch.inference_mode()
    def __call__(self, crop):
        im = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        tensor = self.transform(im).unsqueeze(0).to(self.device)
        return unit(self.model(tensor)[0].cpu().numpy())


MIN_CROP_HEIGHT = 40
MIN_CROP_WIDTH = 16

EMBEDDING_OK = 'ok'
EMBEDDING_INSUFFICIENT = 'insufficient_observations'
EMBEDDING_NO_VALID = 'no_valid_observations'
EMBEDDING_FAILED = 'embedding_failed'


def new_fragment(fragment_id, track_id, frame_idx):
    return {'fragment_id':fragment_id, 'track_id':track_id, 'start_frame':frame_idx, 'end_frame':frame_idx, 'numbers':[],
            '_detections':0, '_sample_attempts':0, '_valid':0, '_invalid':Counter(), '_encoding_failures':0, '_vectors':[]}


def is_sampling_turn(detection_index, sample_every):
    """True for a fragment's own 0th, Nth, 2Nth... detection, independent of absolute frame numbers."""
    return detection_index % sample_every == 0


def clip_crop(frame, xyxy, min_height=MIN_CROP_HEIGHT, min_width=MIN_CROP_WIDTH):
    """Return (crop, None) for a usable frame-clipped crop, else (None, reason)."""
    if frame is None or getattr(frame, 'size', 0) == 0:
        return None, 'empty_frame'
    coords = np.asarray(xyxy, dtype=float)
    if coords.shape != (4,) or not np.all(np.isfinite(coords)):
        return None, 'invalid_bbox'
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = map(int, coords)
    crop = frame[max(0,y1):min(h,y2), max(0,x1):min(w,x2)]
    if crop.size == 0:
        return None, 'empty_crop'
    if crop.shape[0] < min_height or crop.shape[1] < min_width:
        return None, 'crop_too_small'
    return crop, None


def record_observation(fragment, frame, xyxy, encoder, max_observations):
    """Validate one sampled crop; encode it while under the cap. Returns the crop only if it was encoded."""
    fragment['_sample_attempts'] += 1
    crop, reason = clip_crop(frame, xyxy)
    if reason:
        fragment['_invalid'][reason] += 1
        return None
    fragment['_valid'] += 1
    if len(fragment['_vectors']) >= max_observations:
        return None
    vector = np.asarray(encoder(crop), dtype=np.float32)
    if vector.ndim != 1 or not np.all(np.isfinite(vector)) or float(np.linalg.norm(vector)) < 1e-8:
        fragment['_encoding_failures'] += 1
        return None
    fragment['_vectors'].append(vector)
    return crop


def finalize_embedding(fragment, min_observations):
    """Replace working counters with an auditable appearance_evidence record and set embedding (or None)."""
    vectors = fragment.pop('_vectors')
    invalid = fragment.pop('_invalid')
    evidence = {
        'detections': fragment.pop('_detections'),
        'sample_attempts': fragment.pop('_sample_attempts'),
        'valid_observations': fragment.pop('_valid'),
        'invalid_observations': dict(sorted(invalid.items())),
        'encoding_failures': fragment.pop('_encoding_failures'),
        'samples_used': len(vectors),
        'min_observations': min_observations,
    }
    used = len(vectors)
    embedding = None
    if used >= min_observations:
        embedding = unit(np.mean(np.stack(vectors), axis=0))
        if not np.all(np.isfinite(embedding)):
            embedding = None
            status, reason = EMBEDDING_FAILED, 'mean embedding is not finite'
        else:
            status, reason = EMBEDDING_OK, f'{used} encoded observations >= min_observations {min_observations}'
    elif evidence['valid_observations'] == 0:
        detail = ', '.join(f'{k}={v}' for k, v in evidence['invalid_observations'].items()) or 'none'
        status, reason = EMBEDDING_NO_VALID, f"0 valid crops from {evidence['sample_attempts']} sampled detections ({detail})"
    elif used == 0:
        status, reason = EMBEDDING_FAILED, f"all {evidence['encoding_failures']} encodings were non-finite or zero"
    else:
        status, reason = EMBEDDING_INSUFFICIENT, f'{used} encoded observations < min_observations {min_observations}'
    evidence.update(embedding_generated=embedding is not None, status=status, reason=reason)
    fragment['appearance_evidence'] = evidence
    fragment['embedding'] = embedding
    return fragment


class RawDetectionCapture:
    """Read-only observer of the detections handed to ByteTrack.

    Registered on the model before the first ``track()`` call, so ultralytics runs it
    ahead of the tracker's own ``on_predict_postprocess_end`` callback (callbacks fire in
    registration order; the tracker registers itself on that first call). It only reads.
    """

    def __init__(self):
        self.confidences = None

    def __call__(self, predictor):
        results = getattr(predictor, 'results', None)
        if not results:
            self.confidences = []
            return
        boxes = results[0].boxes
        self.confidences = [] if boxes is None else [float(c) for c in boxes.conf.cpu().numpy()]


def confidence_bands(scores, high_thresh, low_thresh):
    """Mutually exclusive bands on the same boundaries ByteTrack itself uses.

    high: conf >= high_thresh. low: low_thresh < conf < high_thresh. below: conf <= low_thresh.
    With the pipeline's NMS gate at 0.3 the low and below bands are empty by construction:
    those detections were removed before the tracker ever saw them, so a zero here is not a
    ByteTrack rejection.
    """
    high = sum(1 for s in scores if s >= high_thresh)
    low = sum(1 for s in scores if low_thresh < s < high_thresh)
    return {'raw_high_conf': high, 'raw_low_conf': low, 'raw_below_low': len(scores) - high - low}


def build_accounting_record(*, frame_idx, fps, scores, emitted_track_ids, tracker_frame_id,
                            active_tracks, lost_tracks, high_thresh, low_thresh,
                            previous_tracker_frame_id):
    """One per-frame accounting record, validated before it can be written."""
    scores = list(scores)
    record = {
        'frame': frame_idx,
        'timestamp_s': frame_idx / fps,
        'raw_detections': len(scores),
        **confidence_bands(scores, high_thresh, low_thresh),
        'emitted_detections': len(emitted_track_ids),
        'dropped_detections': len(scores) - len(emitted_track_ids),
        'tracker_update_skipped': tracker_frame_id == previous_tracker_frame_id,
        'tracker_frame_id': tracker_frame_id,
        'active_tracks': active_tracks,
        'lost_tracks': lost_tracks,
        'emitted_track_ids': [int(i) for i in emitted_track_ids],
    }
    errors = validate_accounting_record(record, context=f'frame {frame_idx}',
                                        previous_tracker_frame_id=previous_tracker_frame_id)
    if errors:
        raise RuntimeError('inconsistent accounting record: ' + '; '.join(errors))
    return record


def new_accounting_totals():
    return {'raw_detections': 0, 'raw_high_conf': 0, 'raw_low_conf': 0, 'raw_below_low': 0,
            'emitted_detections': 0, 'dropped_detections': 0, 'frames_with_drop': 0,
            'max_dropped_detections': 0, 'frames_raw_zero': 0, 'frames_tracker_update_skipped': 0,
            'drop_histogram': Counter()}


def accumulate_accounting(totals, record):
    """Fold one record into the aggregate counters; keeps only O(1) state per frame."""
    for key in ('raw_detections', 'raw_high_conf', 'raw_low_conf', 'raw_below_low',
                'emitted_detections', 'dropped_detections'):
        totals[key] += record[key]
    dropped = record['dropped_detections']
    if dropped > 0:
        totals['frames_with_drop'] += 1
    totals['max_dropped_detections'] = max(totals['max_dropped_detections'], dropped)
    totals['drop_histogram'][dropped] += 1
    if record['raw_detections'] == 0:
        totals['frames_raw_zero'] += 1
    if record['tracker_update_skipped']:
        totals['frames_tracker_update_skipped'] += 1
    return totals


def accounting_consistency(totals, *, frames, boxes_rows, identified_rows, final_tracker_frame_id):
    """The equations that must hold if the accounting and the emitted files agree."""
    raw = totals['raw_detections']
    emitted = totals['emitted_detections']
    checks = {
        'bands_sum_equals_raw': totals['raw_high_conf'] + totals['raw_low_conf'] + totals['raw_below_low'] == raw,
        'emitted_le_raw': emitted <= raw,
        'dropped_equals_raw_minus_emitted': totals['dropped_detections'] == raw - emitted,
        'emitted_equals_boxes_rows': emitted == boxes_rows,
        'emitted_equals_identified_rows': emitted == identified_rows,
        'skipped_equals_zero_raw_frames': totals['frames_tracker_update_skipped'] == totals['frames_raw_zero'],
        'tracker_frame_id_equals_frames_minus_zero_raw_frames': final_tracker_frame_id == frames - totals['frames_raw_zero'],
    }
    return {'all_hold': all(checks.values()), 'checks': checks,
            'violations': [name for name, ok in checks.items() if not ok]}


def build_accounting_summary(*, totals, frames, fps, video, settings, final_tracker_frame_id,
                             boxes_rows, identified_rows):
    raw = totals['raw_detections']
    emission_rate = totals['emitted_detections'] / raw if raw else None
    return {
        'accounting_version': ACCOUNTING_VERSION,
        'video': video, 'frames_processed': frames, 'fps': fps,
        'settings': settings,
        'counter_definitions': {
            'raw_detections': 'boxes in Results.boxes immediately before the ByteTrack callback: after NMS, conf gate, class filter and max_det',
            'raw_high_conf': 'raw with conf >= track_high_thresh',
            'raw_low_conf': 'raw with track_low_thresh < conf < track_high_thresh',
            'raw_below_low': 'raw with conf <= track_low_thresh',
            'emitted_detections': 'tracker boxes with an id returned for this frame; equals rows written to boxes.jsonl',
            'dropped_detections': 'raw - emitted; a derived total with no per-detection cause',
            'tracker_update_skipped': 'BYTETracker.update was not called this frame (track.py skips empty detections)',
            'tracker_frame_id': "the tracker's own counter, not the video frame index",
            'active_tracks': 'len(tracker.tracked_stracks) after this frame',
            'lost_tracks': 'len(tracker.lost_stracks) after this frame',
            'emission_rate_denominator': 'raw_detections',
            'drop_rate_denominator': 'raw_detections',
            'frames_with_drop_denominator': 'frames_processed',
        },
        'totals': {k: (dict(sorted(v.items())) if isinstance(v, Counter) else v) for k, v in totals.items()},
        'rates': {'emission_rate': emission_rate,
                  'drop_rate': 1 - emission_rate if emission_rate is not None else None},
        'drop_histogram': {str(k): v for k, v in sorted(totals['drop_histogram'].items())},
        'confidence_band_totals': {k: totals[k] for k in ('raw_high_conf', 'raw_low_conf', 'raw_below_low')},
        'final_tracker_frame_id': final_tracker_frame_id,
        'tracker_frame_id_note': 'video frames minus frames with zero raw detections; deliberately not a frame index',
        'consistency': accounting_consistency(totals, frames=frames, boxes_rows=boxes_rows,
                                              identified_rows=identified_rows,
                                              final_tracker_frame_id=final_tracker_frame_id),
        'not_measured': {
            'per_detection_drop_reason': (
                'ByteTrack computes u_track/u_detection/u_unconfirmed inside update() and discards '
                'them; attributing a cause to each dropped detection would require patching the library.'),
            'removed_tracks_total': (
                'tracker.removed_stracks is truncated to its last 999 entries by byte_tracker.py, '
                'so it is not a valid counter.'),
            'low_confidence_second_association': (
                'with the NMS gate at conf=0.3 no detection falls in the (track_low_thresh, track_high_thresh) '
                'band, so the second association never receives candidates; the zero count is a property of '
                'the gate, not a ByteTrack rejection.'),
        },
    }


def resolve(fragments, roster, prototypes, threshold, margin):
    players = roster.get('players', [])
    assigned = {}
    accepted = {p['player_id']: [] for p in players}
    for f in sorted(fragments, key=lambda x: x['start_s']):
        scores = []
        votes = Counter(f['numbers'])
        jersey = None
        if votes:
            num, count = votes.most_common(1)[0]
            if count >= 3 and count / sum(votes.values()) >= .8:
                jersey = int(num)
        f['jersey'] = jersey
        for p in players:
            pid = p['player_id']
            if f['start_s'] < p.get('active_from', 0) or f['end_s'] > p.get('active_to', float('inf')):
                continue
            if any(f['start_frame'] <= e and f['end_frame'] >= s for s, e in accepted[pid]):
                continue
            if jersey is not None and jersey != int(p['jersey']):
                continue
            if pid not in prototypes or f['embedding'] is None:
                continue
            score = float(np.dot(f['embedding'], prototypes[pid]))
            scores.append((score, pid))
        scores.sort(reverse=True)
        best = scores[0] if scores else (0., None)
        gap = best[0] - scores[1][0] if len(scores) > 1 else best[0]
        if best[1] and best[0] >= threshold and gap >= margin:
            pid = best[1]
            accepted[pid].append((f['start_frame'], f['end_frame']))
            assigned[f['fragment_id']] = pid
            f.update(player_id=pid, identification_confidence=best[0], confidence=best[0], reason='reference_similarity_and_constraints', identification_reason='reference_similarity_and_constraints', team_id=normalize_team_label(next(p['team_id'] for p in players if p['player_id']==pid)))
        else:
            assigned[f['fragment_id']] = PLAYER_ID_UNRESOLVED
            why = 'insufficient_appearance_evidence' if f['embedding'] is None else 'insufficient_or_conflicting_evidence'
            f.update(player_id=PLAYER_ID_UNRESOLVED, identification_confidence=max(0., best[0]), confidence=max(0., best[0]), reason=why, identification_reason=why, team_id=TEAM_UNKNOWN)
    return assigned


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', required=True)
    ap.add_argument('--roster')
    ap.add_argument('--out', default='results')
    ap.add_argument('--weights', default='yolov8n.pt')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--ocr', action='store_true')
    ap.add_argument('--sample-every', type=int, default=15, help="sample every Nth detection of each fragment (fragment's own detection index, not absolute frame)")
    ap.add_argument('--min-observations', type=int, default=3, help='minimum encoded appearance observations required to emit a fragment embedding')
    ap.add_argument('--max-observations', type=int, default=32, help='maximum appearance observations encoded per fragment (memory bound)')
    ap.add_argument('--fragment-gap', type=int, default=30)
    ap.add_argument('--max-fragment-seconds', type=float, default=15)
    ap.add_argument('--threshold', type=float, default=.88)
    ap.add_argument('--margin', type=float, default=.08)
    ap.add_argument('--max-frames', type=int, default=0)
    ap.add_argument('--render', action='store_true')
    a = ap.parse_args()
    if a.sample_every < 1 or a.fragment_gap < 1 or a.max_fragment_seconds <= 0:
        ap.error('Sampling/gap/duration must be positive')
    if not 1 <= a.min_observations <= a.max_observations:
        ap.error('Require 1 <= --min-observations <= --max-observations')
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    roster_path = Path(a.roster) if a.roster else None
    roster = json.loads(roster_path.read_text()) if roster_path else {'players': [], 'seeds': {}}
    ids = [p['player_id'] for p in roster['players']]
    if len(ids) != len(set(ids)):
        raise ValueError('player_id must be unique')
    encoder = Encoder(a.device)
    prototypes = {}
    for pid, paths in roster.get('seeds', {}).items():
        vectors = []
        for path in paths:
            path = Path(path)
            if not path.is_absolute(): path = roster_path.parent / path
            im = cv2.imread(str(path))
            if im is None: raise ValueError(f'Cannot read reference {path}')
            vectors.append(encoder(im))
        if vectors: prototypes[pid] = unit(np.mean(vectors, axis=0))
    reader = None
    if a.ocr:
        import easyocr
        reader = easyocr.Reader(['en'], gpu=a.device.startswith('cuda'))
    model = YOLO(a.weights)
    capture = RawDetectionCapture()
    model.add_callback('on_predict_postprocess_end', capture)
    person_ids = [int(k) for k, v in model.names.items() if v in ('person', 'player', 'goalkeeper')]
    if not person_ids: raise ValueError('Detector must have person/player/goalkeeper class names')
    cap = cv2.VideoCapture(a.video)
    if not cap.isOpened(): raise ValueError('Cannot open input video')
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0: raise ValueError('Video FPS unavailable')
    active = {}; fragments = []; frame_idx = 0; detection_seq = Counter()
    samples_path = out/'boxes.jsonl'
    accounting_path = out/'detection_accounting.jsonl'
    accounting_totals = new_accounting_totals(); accounting_settings = None
    prev_tracker_frame_id = 0
    with samples_path.open('w') as log, accounting_path.open('w') as acct_log:
        while True:
            ok, frame = cap.read()
            if not ok or (a.max_frames and frame_idx >= a.max_frames): break
            result = model.track(frame, persist=True, tracker='bytetrack.yaml', classes=person_ids, conf=.3, imgsz=960, device=a.device, verbose=False)[0]
            emitted_ids = []
            if result.boxes is not None and result.boxes.id is not None:
                for xyxy, tid, cls, conf in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.id.int().cpu().tolist(), result.boxes.cls.int().cpu().tolist(), result.boxes.conf.cpu().tolist()):
                    emitted_ids.append(tid)
                    f = active.get(tid)
                    if f is None or frame_idx-f['end_frame'] > a.fragment_gap or (frame_idx-f['start_frame'])/fps >= a.max_fragment_seconds:
                        f = new_fragment(f'F{len(fragments)+1:06d}', tid, frame_idx)
                        fragments.append(f); active[tid] = f
                    f['end_frame'] = frame_idx
                    detection_index = f['_detections']; f['_detections'] += 1
                    # detection_id is unique per (frame, tagged key kind, key, occurrence) within this run.
                    seq = detection_seq[(frame_idx, tid)]
                    detection_seq[(frame_idx, tid)] += 1
                    row = {'frame':frame_idx, 'timestamp_s':frame_idx/fps, 'detection_id':build_detection_id(frame_idx, tid, seq), 'fragment_id':f['fragment_id'], 'track_id':tid, 'bbox_xyxy':[float(x) for x in xyxy], 'detector_confidence':float(conf), 'class_id':cls}
                    log.write(json.dumps(row)+'\n')
                    if is_sampling_turn(detection_index, a.sample_every):
                        crop = record_observation(f, frame, xyxy, encoder, a.max_observations)
                        if crop is not None and reader:
                            torso = crop[int(.15*len(crop)):int(.7*len(crop))]
                            for _, text, confidence in reader.readtext(torso, allowlist='0123456789'):
                                if confidence >= .85 and re.fullmatch(r'\d{1,2}',text) and 1<=int(text)<=99:
                                    f['numbers'].append(int(text))
            tracker = model.predictor.trackers[0]
            if capture.confidences is None:
                raise RuntimeError('detection accounting hook did not run for this frame')
            if accounting_settings is None:
                pargs = model.predictor.args
                accounting_settings = {
                    'nms_conf': float(pargs.conf), 'nms_iou': float(pargs.iou),
                    'max_det': int(pargs.max_det),
                    'imgsz': [int(x) for x in pargs.imgsz] if isinstance(pargs.imgsz, (list, tuple)) else int(pargs.imgsz),
                    'classes': person_ids,
                    'tracker': 'bytetrack.yaml',
                    'track_high_thresh': float(tracker.args.track_high_thresh),
                    'track_low_thresh': float(tracker.args.track_low_thresh),
                    'new_track_thresh': float(tracker.args.new_track_thresh),
                    'match_thresh': float(tracker.args.match_thresh),
                    'track_buffer': int(tracker.args.track_buffer), 'fuse_score': bool(tracker.args.fuse_score),
                }
            record = build_accounting_record(
                frame_idx=frame_idx, fps=fps, scores=capture.confidences, emitted_track_ids=emitted_ids,
                tracker_frame_id=tracker.frame_id, active_tracks=len(tracker.tracked_stracks),
                lost_tracks=len(tracker.lost_stracks), high_thresh=tracker.args.track_high_thresh,
                low_thresh=tracker.args.track_low_thresh, previous_tracker_frame_id=prev_tracker_frame_id)
            acct_log.write(json.dumps(record)+'\n')
            accumulate_accounting(accounting_totals, record)
            prev_tracker_frame_id = tracker.frame_id
            frame_idx += 1
            if frame_idx % 300 == 0: print(f'Processed {frame_idx} frames', flush=True)
    cap.release()
    for f in fragments:
        finalize_embedding(f, a.min_observations)
        f['start_s'] = f['start_frame']/fps; f['end_s'] = f['end_frame']/fps
    resolve(fragments, roster, prototypes, a.threshold, a.margin)
    by_fragment = {f['fragment_id']:f for f in fragments}
    groups = {p['player_id']: {'team_id':p['team_id'],'jersey':p['jersey'],'fragments':[]} for p in roster['players']}
    groups[PLAYER_ID_UNRESOLVED] = {'fragments':[]}
    for f in fragments:
        groups[f['player_id']]['fragments'].append(f['fragment_id'])
        if f['embedding'] is not None: f['embedding'] = f['embedding'].tolist()
    (out/'fragments.json').write_text(json.dumps(fragments, indent=2))
    (out/'players.json').write_text(json.dumps(groups, indent=2))
    with (out/'identified_boxes.jsonl').open('w') as dst, samples_path.open() as src:
        for line in src:
            row = json.loads(line); f = by_fragment[row['fragment_id']]
            row.update({k:f[k] for k in ('player_id','team_id','jersey','identification_confidence','identification_reason','confidence','reason')})
            dst.write(json.dumps(row)+'\n')
    boxes_rows = sum(1 for line in samples_path.open())
    identified_rows = sum(1 for line in (out/'identified_boxes.jsonl').open())
    summary = build_accounting_summary(
        totals=accounting_totals, frames=frame_idx, fps=fps, video=a.video,
        settings=accounting_settings, final_tracker_frame_id=prev_tracker_frame_id,
        boxes_rows=boxes_rows, identified_rows=identified_rows)
    (out/'detection_accounting_summary.json').write_text(json.dumps(summary, indent=2))
    if not summary['consistency']['all_hold']:
        print('WARNING: detection accounting consistency violations: '
              + ', '.join(summary['consistency']['violations']), flush=True)
    if a.render:
        cap = cv2.VideoCapture(a.video)
        width,height = int(cap.get(3)), int(cap.get(4))
        writer = cv2.VideoWriter(str(out/'identified.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),fps,(width,height))
        if not writer.isOpened(): raise RuntimeError('MP4 writer unavailable')
        with (out/'identified_boxes.jsonl').open() as src:
            row = next(src, None); row = json.loads(row) if row else None
            for idx in range(frame_idx):
                ok, frame = cap.read()
                if not ok: break
                while row is not None and row['frame']==idx:
                    x1,y1,x2,y2 = map(int,row['bbox_xyxy'])
                    color = (0,180,0) if row['player_id']!='UNK' else (0,165,255)
                    cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
                    cv2.putText(frame,f"{row['fragment_id']} {row['player_id']}",(x1,max(15,y1-5)),cv2.FONT_HERSHEY_SIMPLEX,.45,color,1)
                    line = next(src,None); row = json.loads(line) if line else None
                writer.write(frame)
        cap.release(); writer.release()
    (out/'run_manifest.json').write_text(json.dumps({'video':a.video,'frames':frame_idx,'fps':fps,'detector':a.weights,'embedding':'torchvision ResNet18 ImageNet pretrained, NOT player-ReID trained','trained_in_this_project':False,'ocr':'EasyOCR pretrained' if reader else None,'threshold':a.threshold,'margin':a.margin,'appearance_sampling':{'rule':'per-fragment detection index','sample_every':a.sample_every,'min_observations':a.min_observations,'max_observations':a.max_observations,'min_crop_height':MIN_CROP_HEIGHT,'min_crop_width':MIN_CROP_WIDTH},'detection_accounting':{'enabled':True,'accounting_version':ACCOUNTING_VERSION,'per_frame_file':'detection_accounting.jsonl','summary_file':'detection_accounting_summary.json','settings':accounting_settings,'raw_detections':accounting_totals['raw_detections'],'emitted_detections':accounting_totals['emitted_detections'],'dropped_detections':accounting_totals['dropped_detections']},'schema_version':SCHEMA_VERSION,'output_contract':schema_descriptor(),'identity_fields_note':'person_uid is reserved and not emitted; team_id in this stage is roster-inherited, not an independent classification','scope':'LTPI-inspired prototype; not official LTPI reproduction'},indent=2))
    print(f'Finished: {len(fragments)} fragments. Outputs in {out}')

if __name__=='__main__': main()
