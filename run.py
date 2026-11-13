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
    PLAYER_ID_UNRESOLVED,
    SCHEMA_VERSION,
    TEAM_UNKNOWN,
    build_detection_id,
    normalize_team_label,
    schema_descriptor,
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
    person_ids = [int(k) for k, v in model.names.items() if v in ('person', 'player', 'goalkeeper')]
    if not person_ids: raise ValueError('Detector must have person/player/goalkeeper class names')
    cap = cv2.VideoCapture(a.video)
    if not cap.isOpened(): raise ValueError('Cannot open input video')
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0: raise ValueError('Video FPS unavailable')
    active = {}; fragments = []; frame_idx = 0; detection_seq = Counter()
    samples_path = out/'boxes.jsonl'
    with samples_path.open('w') as log:
        while True:
            ok, frame = cap.read()
            if not ok or (a.max_frames and frame_idx >= a.max_frames): break
            result = model.track(frame, persist=True, tracker='bytetrack.yaml', classes=person_ids, conf=.3, imgsz=960, device=a.device, verbose=False)[0]
            if result.boxes is not None and result.boxes.id is not None:
                for xyxy, tid, cls, conf in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.id.int().cpu().tolist(), result.boxes.cls.int().cpu().tolist(), result.boxes.conf.cpu().tolist()):
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
    (out/'run_manifest.json').write_text(json.dumps({'video':a.video,'frames':frame_idx,'fps':fps,'detector':a.weights,'embedding':'torchvision ResNet18 ImageNet pretrained, NOT player-ReID trained','trained_in_this_project':False,'ocr':'EasyOCR pretrained' if reader else None,'threshold':a.threshold,'margin':a.margin,'appearance_sampling':{'rule':'per-fragment detection index','sample_every':a.sample_every,'min_observations':a.min_observations,'max_observations':a.max_observations,'min_crop_height':MIN_CROP_HEIGHT,'min_crop_width':MIN_CROP_WIDTH},'schema_version':SCHEMA_VERSION,'output_contract':schema_descriptor(),'identity_fields_note':'person_uid is reserved and not emitted; team_id in this stage is roster-inherited, not an independent classification','scope':'LTPI-inspired prototype; not official LTPI reproduction'},indent=2))
    print(f'Finished: {len(fragments)} fragments. Outputs in {out}')

if __name__=='__main__': main()
