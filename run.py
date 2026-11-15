import argparse, hashlib, json, csv, re, time
from pathlib import Path
from collections import Counter
from dataclasses import asdict, replace
from team_reliability import assess_separation, load_team_calibration
import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.models import resnet18, ResNet18_Weights
from ultralytics import YOLO
from crop_filter import CropFilterPolicy, FilterStats
from team_classifier import (
    CANDIDATE_ACTIONS,
    CANDIDATE_MODES,
    ROLE_NONE,
    TRACK_CANDIDATE_MEAN_SCORE,
    TRACK_CANDIDATE_SUPPORT_RATIO,
    TeamAppearanceClassifier,
    TeamPrediction,
    evaluate_role_candidate,
    extract_torso_crop,
)
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


# --- Independent team classification (Stage 4A) ----------------------------
#
# The kit classifier is the existing, tested one from team_classifier.py. Nothing
# below reads player_id, roster state or identification results, so a fragment can
# be TEAM_A while its player stays UNK. Only the fragment-level aggregation knobs
# are configurable here; the classifier's own minimums are preserved unchanged.

TEAM_MIN_SAMPLES = 12              # TeamAppearanceClassifier default; run_team_clustering.py:523 enforces the same floor
TEAM_MIN_CLUSTER_SAMPLES = 4       # TeamAppearanceClassifier default
TEAM_RANDOM_STATE = 17             # preserved: canonical ordering + fixed seed make labels reproducible
TEAM_OUTLIER_MULTIPLIER = 1.35
TEAM_CANDIDATE_MODE_DEFAULT = 'multi'
TEAM_CANDIDATE_ACTION_DEFAULT = 'flag'
TEAM_SCORE_KIND = 'distance_to_radius_heuristic_not_probability'
TEAM_SEMANTICS_NOTE = (
    'TEAM_A / TEAM_B are anonymous deterministic cluster slots of this fit, not club '
    'names and not automatically HOME / AWAY. They are only comparable within one run '
    'and carry no meaning across unrelated videos.'
)

TEAM_STATUS_FRAGMENT_MAJORITY = 'fragment_majority_vote'
TEAM_STATUS_NO_ACCEPTED_PREDICTIONS = 'no_accepted_crop_predictions'
TEAM_STATUS_NO_CROPS = 'no_torso_crops'
TEAM_STATUS_INSUFFICIENT_VOTES = 'insufficient_fragment_votes'
TEAM_STATUS_CONFLICTING_VOTES = 'conflicting_fragment_votes'
TEAM_STATUS_ROLE_CANDIDATE = 'persistent_role_candidate'
TEAM_STATUS_INSUFFICIENT_CROPS = 'insufficient_crops_for_clustering'
TEAM_STATUS_CLUSTERING_REJECTED = 'clustering_rejected'


def collect_team_crop(fragment, frame, xyxy, meta, *, policy, box_stats, crop_stats, max_crops):
    """Quality-filter one sampled detection and retain its torso crop for clustering.

    ``extract_torso_crop`` returns a view into ``frame``, so the crop is copied: keeping
    the view would pin the whole decoded frame in memory for the rest of the run.
    ``meta`` carries the detection's provenance (frame, timestamp_s, detection_id,
    detector_confidence) so a saved crop index can be written without a second decode.
    """
    height, width = frame.shape[:2]
    confidence = float(meta['detector_confidence'])
    result = policy.evaluate_box(xyxy, confidence, width, height)
    box_stats.record(result)
    if not result.is_valid:
        return False
    crop = extract_torso_crop(frame, xyxy)
    crop_result = policy.evaluate_torso_crop(crop)
    crop_stats.record(crop_result)
    if not crop_result.is_valid:
        return False
    stored = fragment.setdefault('_team_crops', [])
    if len(stored) >= max_crops:
        return False
    x1, y1, x2, y2 = map(float, xyxy)
    stored.append({'crop': np.array(crop, copy=True, order='C'),
                   'frame': int(meta['frame']), 'timestamp_s': float(meta['timestamp_s']),
                   'detection_id': meta['detection_id'],
                   'box_width': round(x2 - x1, 2), 'box_height': round(y2 - y1, 2),
                   'detector_confidence': round(confidence, 6)})
    return True


def balance_team_fit_set(fragments, max_per_fragment):
    """Evenly spaced per-fragment selection, mirroring run_team_clustering.balance_by_track.

    Without it one long fragment could dominate the two-cluster fit. Fragment order is
    sorted so the fitting multiset does not depend on collection order.
    """
    crops, ids = [], []
    for f in sorted(fragments, key=lambda x: x['fragment_id']):
        stored = f.get('_team_crops') or []
        if not stored:
            continue
        count = min(max_per_fragment, len(stored))
        if count == len(stored):
            indices = list(range(len(stored)))
        elif count == 1:
            indices = [len(stored) // 2]
        else:
            indices = sorted({int(i) for i in np.linspace(0, len(stored) - 1, count).round()})
        for i in indices:
            crops.append(stored[i]['crop'])
            ids.append(f"{f['fragment_id']}#{i}")
    return crops, ids


def build_team_classifier(*, candidate_mode, candidate_action):
    """Construct the existing classifier with its documented defaults preserved."""
    return TeamAppearanceClassifier(
        min_samples=TEAM_MIN_SAMPLES,
        min_cluster_samples=TEAM_MIN_CLUSTER_SAMPLES,
        random_state=TEAM_RANDOM_STATE,
        outlier_multiplier=TEAM_OUTLIER_MULTIPLIER,
        candidate_mode=candidate_mode,
        candidate_action=candidate_action,
    )


def aggregate_fragment_team(predictions, *, min_votes, min_majority,
                            support_ratio=TRACK_CANDIDATE_SUPPORT_RATIO,
                            mean_score_floor=TRACK_CANDIDATE_MEAN_SCORE):
    """Roll crop predictions into one fragment label.

    Same rule as the validated track-level vote in run_team_clustering.py: a label needs
    at least ``min_votes`` non-UNKNOWN votes and ``min_majority`` agreement, and only
    persistent role-candidate evidence (support and mean score both above their floors)
    can remove an accepted label. One odd crop never can.
    """
    votes = Counter(p.team_id for p in predictions if p.team_id != TEAM_UNKNOWN)
    known = sum(votes.values())
    scored = [p for p in predictions if p.candidate is not None]
    candidate_scores = [float(p.candidate.score) for p in scored if p.candidate.role != ROLE_NONE]
    support = len(candidate_scores) / len(scored) if scored else 0.0
    mean_score = float(np.mean(candidate_scores)) if candidate_scores else 0.0
    roles = Counter(p.candidate.role for p in scored if p.candidate.role != ROLE_NONE)
    record = {
        'team_id': TEAM_UNKNOWN,
        'evidence_status': TEAM_STATUS_NO_CROPS if not predictions else TEAM_STATUS_NO_ACCEPTED_PREDICTIONS,
        'crops_predicted': len(predictions),
        'votes_known': known,
        'crop_votes': {k: v for k, v in sorted(votes.items())},
        'majority_ratio': None,
        'winning_votes': 0,
        'score': 0.0,
        'score_kind': TEAM_SCORE_KIND,
        'cluster_fingerprint': None,
        'crop_abstention_reasons': dict(sorted(
            Counter(p.reason for p in predictions if p.team_id == TEAM_UNKNOWN).items())),
        'candidate_evidence': {
            'candidate_crops': len(candidate_scores),
            'scored_crops': len(scored),
            'candidate_support': round(support, 4),
            'mean_candidate_score': round(mean_score, 4),
            'roles': dict(roles),
            'dominant_role': roles.most_common(1)[0][0] if roles else ROLE_NONE,
            'persistent_evidence': bool(candidate_scores) and support >= support_ratio
                                   and mean_score >= mean_score_floor,
        },
    }
    if known == 0:
        return record
    team, wins = sorted(votes.items(), key=lambda kv: (-kv[1], kv[0]))[0]
    majority = wins / known
    record['majority_ratio'] = round(majority, 4)
    record['winning_votes'] = wins
    if known < min_votes:
        record['evidence_status'] = TEAM_STATUS_INSUFFICIENT_VOTES
    elif majority < min_majority:
        record['evidence_status'] = TEAM_STATUS_CONFLICTING_VOTES
    else:
        record['team_id'] = normalize_team_label(team)
        record['evidence_status'] = TEAM_STATUS_FRAGMENT_MAJORITY
        record['score'] = round(float(np.mean(
            [p.confidence for p in predictions if p.team_id == team])), 4)
    if record['candidate_evidence']['persistent_evidence'] and record['team_id'] != TEAM_UNKNOWN:
        record['team_id_before_candidate_exclusion'] = record['team_id']
        record['score_before_candidate_exclusion'] = record['score']
        record['score'] = 0.0
        record['team_id'] = TEAM_UNKNOWN
        record['evidence_status'] = TEAM_STATUS_ROLE_CANDIDATE
    return record


def reliable_fragment_team(predictions, separation, calibration, *, min_votes, min_majority):
    proposal = aggregate_fragment_team(predictions, min_votes=min_votes, min_majority=min_majority)
    filtered = []
    diagnostics = []
    for prediction in predictions:
        reason = None
        if prediction.team_id != TEAM_UNKNOWN and calibration is not None:
            thresholds = calibration['thresholds']
            if prediction.confidence <= 0 or prediction.confidence < thresholds['min_crop_score']:
                reason = 'weak_crop_score'
            elif prediction.margin is None or prediction.margin < thresholds['min_crop_margin']:
                reason = 'ambiguous_crop_distance'
        final = replace(prediction, team_id=TEAM_UNKNOWN, confidence=0.0, reason=reason) if reason else prediction
        filtered.append(final)
        diagnostics.append({'proposal_team_id': prediction.team_id, 'proposal_score': prediction.confidence,
                            'distance': prediction.distance, 'distance_margin': prediction.margin,
                            'accepted_crop_team_id': final.team_id, 'reason': final.reason,
                            'role_candidate': asdict(prediction.candidate) if prediction.candidate else None})
    record = aggregate_fragment_team(filtered, min_votes=min_votes, min_majority=min_majority)
    record['proposal'] = {k: proposal[k] for k in ('team_id', 'score', 'evidence_status', 'crop_votes')}
    record['crop_predictions'] = diagnostics
    record['separation_decision'] = separation['decision']
    reason = None
    if record['team_id'] != TEAM_UNKNOWN:
        if len(proposal['crop_votes']) > 1:
            reason = 'mixed_fragment_appearance'
        elif separation['decision'] != 'accepted':
            reason = separation['reason']
    if reason:
        record['team_id_before_reliability'] = record['team_id']
        record['score_before_reliability'] = record['score']
        record.update(team_id=TEAM_UNKNOWN, score=0.0, evidence_status=reason)
    return record


def classify_fragment_teams(fragments, *, policy, box_stats, crop_stats, collection_seconds,
                            min_votes, min_majority, max_per_fragment_fit,
                            candidate_mode, candidate_action, classifier=None, keep_crops=False,
                            calibration=None):
    """Fit the kit clusters once and label every fragment. Returns the report dict.

    Failure to cluster is not an error: the run continues with UNKNOWN labels and the
    reason is recorded, because UNKNOWN is a valid team outcome. Only the classifier's
    documented insufficient-data ValueError is caught; anything else propagates.
    """
    started = time.perf_counter()
    classifier = classifier or build_team_classifier(
        candidate_mode=candidate_mode, candidate_action=candidate_action)
    with_crops = [f for f in fragments if f.get('_team_crops')]
    fit_crops, fit_ids = balance_team_fit_set(fragments, max_per_fragment_fit)
    stored_crops = sum(len(f.get('_team_crops') or []) for f in fragments)

    fit_report, fit_error, fitted = None, None, False
    if len(fit_crops) < TEAM_MIN_SAMPLES:
        fit_error = (f'{len(fit_crops)} balanced fitting crops < min_samples {TEAM_MIN_SAMPLES}; '
                     f'{stored_crops} crops were collected in total')
        abstain_status = TEAM_STATUS_INSUFFICIENT_CROPS
    else:
        try:
            fit_report = classifier.fit(fit_crops, crop_ids=fit_ids)
            fitted = True
        except ValueError as exc:
            fit_error = str(exc)
            abstain_status = TEAM_STATUS_CLUSTERING_REJECTED
    fingerprint = fit_report['cluster_fingerprint'] if fitted else None
    separation = assess_separation(fit_report['separation'], calibration) if fitted else {
        'decision': 'inconclusive', 'reason': abstain_status, 'metrics': None, 'checks': {}}

    labels, statuses, proposals = Counter(), Counter(), Counter()
    for f in fragments:
        stored = f.pop('_team_crops', [])
        if fitted:
            predictions = [classifier.predict(entry['crop']) for entry in stored]
            record = reliable_fragment_team(predictions, separation, calibration,
                                            min_votes=min_votes, min_majority=min_majority)
            record['cluster_fingerprint'] = fingerprint
            proposals[record['proposal']['team_id']] += 1
        else:
            predictions = [TeamPrediction(reason=abstain_status, candidate=(
                evaluate_role_candidate(entry['crop'], mode=candidate_mode)
                if candidate_mode == 'multi' else None)) for entry in stored]
            record = reliable_fragment_team(predictions, separation, None,
                                            min_votes=min_votes, min_majority=min_majority)
            record['evidence_status'] = abstain_status
        for entry, diagnostic in zip(stored, record['crop_predictions']):
            diagnostic['detection_id'] = entry['detection_id']
            diagnostic['frame'] = entry['frame']
        record['crops_collected'] = len(stored)
        record['roster_inherited_team_id'] = f.get('team_id')
        if keep_crops:
            record['_crops'] = stored
        f['team_classification'] = record
        # The independent label owns team_id in the emitted rows and in fragments.json.
        f['team_id'] = record['team_id']
        labels[record['team_id']] += 1
        statuses[record['evidence_status']] += 1

    detections = Counter()
    for f in fragments:
        detections[f['team_id']] += int(f.get('appearance_evidence', {}).get('detections', 0))

    report = {
        'enabled': True,
        'classifier': {
            'module': 'team_classifier.TeamAppearanceClassifier',
            'min_samples': TEAM_MIN_SAMPLES,
            'min_cluster_samples': TEAM_MIN_CLUSTER_SAMPLES,
            'max_iterations': classifier.max_iterations,
            'random_state': classifier.random_state,
            'outlier_multiplier': classifier.outlier_multiplier,
            'minimum_radius': classifier.minimum_radius,
            'use_grass_mask': classifier.use_grass_mask,
            'min_foreground_ratio': classifier.min_foreground_ratio,
            'candidate_mode': classifier.candidate_mode,
            'candidate_action': classifier.candidate_action,
            'candidate_threshold': classifier.candidate_threshold,
        },
        'filter_policy': policy.to_dict(),
        'crop_collection': {
            'fragments_total': len(fragments),
            'fragments_with_crops': len(with_crops),
            'crops_stored': stored_crops,
            'box_filter': box_stats.to_dict(),
            'torso_filter': crop_stats.to_dict(),
        },
        'fit': {
            'fitted': fitted,
            'fitting_crops': len(fit_crops),
            'max_crops_per_fit_fragment': max_per_fragment_fit,
            'error': fit_error,
            'cluster_fingerprint': fingerprint,
            'valid_samples': fit_report['valid_samples'] if fitted else None,
            'fitting_set_candidate_gated': fit_report['fitting_set_candidate_gated'] if fitted else None,
            'cluster_sizes': fit_report['cluster_sizes'] if fitted else None,
            'mean_within_cluster_distance': fit_report['mean_within_cluster_distance'] if fitted else None,
            'outlier_radii': fit_report['outlier_radii'] if fitted else None,
            'labeling_rule': fit_report['labeling_rule'] if fitted else None,
            'anchor_crop_id': fit_report['anchor_crop_id'] if fitted else None,
        },
        'aggregation': {
            'min_votes': min_votes,
            'min_majority': min_majority,
            'candidate_support_ratio': TRACK_CANDIDATE_SUPPORT_RATIO,
            'candidate_mean_score_floor': TRACK_CANDIDATE_MEAN_SCORE,
            'score_kind': TEAM_SCORE_KIND,
        },
        'fragment_label_distribution': dict(sorted(labels.items())),
        'fragment_evidence_status_distribution': dict(sorted(statuses.items())),
        'detection_label_distribution': dict(sorted(detections.items())),
        'fragments_excluded_as_role_candidates': sum(
            1 for f in fragments
            if f['team_classification']['evidence_status'] == TEAM_STATUS_ROLE_CANDIDATE),
        'cluster_separation_check': separation,
        'uncalibrated_proposal_distribution': dict(sorted(proposals.items())),
        'calibration': calibration,
        'mixed_fragments': sum(f['team_classification']['evidence_status'] == 'mixed_fragment_appearance'
                               for f in fragments),
        'runtime_seconds': {
            'crop_collection': round(collection_seconds, 4),
            'fit_and_predict': round(time.perf_counter() - started, 4),
        },
        'semantics': TEAM_SEMANTICS_NOTE,
        'accuracy_claim': None,
    }
    return report


def team_row_fields(fragment):
    """The team fields copied onto each detection row of a fragment."""
    record = fragment['team_classification']
    return {'team_id': record['team_id'],
            'team_evidence_status': record['evidence_status'],
            'team_score': record['score'],
            'team_score_kind': record['score_kind'],
            'team_cluster_fingerprint': record['cluster_fingerprint']}


def write_team_crop_index(out, fragments, *, video, frame_wh, frames, rows_read, sampled, policy):
    """Save accepted torso crops plus an index in the crop_index.json shape.

    The index is consumable by the existing run_team_clustering.py and review tools, so
    contact sheets need no new code. Off by default: ordinary runs keep no crop images.
    """
    crops_dir = out/'team_crops'
    crops_dir.mkdir(parents=True, exist_ok=True)
    width, height = frame_wh
    samples, unwritten = [], 0
    for f in sorted(fragments, key=lambda x: x['fragment_id']):
        record = f['team_classification']
        for i, entry in enumerate(record.get('_crops', [])):
            name = f"fragment_{f['fragment_id']}_d{i:02d}_frame_{entry['frame']:06d}.jpg"
            path = crops_dir/name
            if not cv2.imwrite(str(path), entry['crop']):
                unwritten += 1
                continue
            samples.append({'file': str(path), 'fragment_id': f['fragment_id'],
                            'detection_id': entry['detection_id'], 'track_id': int(f['track_id']),
                            'frame': entry['frame'], 'timestamp_s': entry['timestamp_s'],
                            'detector_confidence': entry['detector_confidence'],
                            'box_width': entry['box_width'], 'box_height': entry['box_height'],
                            'crop_width': int(entry['crop'].shape[1]),
                            'crop_height': int(entry['crop'].shape[0]),
                            'team_id': record['team_id'],
                            'team_evidence_status': record['evidence_status']})
    index = {'video': video, 'video_width': width, 'video_height': height, 'frames': frames,
             'detections_read': rows_read, 'sampled_detections': sampled,
             'filter_policy': policy.to_dict(), 'samples': samples,
             'crop_write_failures': unwritten, 'schema_note':
                 'Same sample shape as review_team_crops_v2 crop_index.json, plus fragment_id '
                 'and the fragment-level team label.'}
    (out/'team_crop_index.json').write_text(json.dumps(index, indent=2))
    return index


def release_team_crops(fragments):
    """Drop retained crop arrays so fragments.json stays serializable and memory is freed."""
    for f in fragments:
        f['team_classification'].pop('_crops', None)


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
    ap.add_argument('--team-sample-every', type=int, default=5,
                    help='sample every Nth detection of each fragment for team torso crops (independent of --sample-every)')
    ap.add_argument('--team-max-crops-per-fragment', type=int, default=8,
                    help='maximum torso crops retained per fragment (memory bound)')
    ap.add_argument('--team-max-crops-per-fit-fragment', type=int, default=2,
                    help='evenly spaced crops per fragment used to fit the two kit clusters')
    ap.add_argument('--team-min-votes', type=int, default=2,
                    help='minimum non-UNKNOWN crop votes before a fragment may take a team label')
    ap.add_argument('--team-min-majority', type=float, default=.60,
                    help='minimum agreement ratio among the non-UNKNOWN crop votes')
    ap.add_argument('--team-candidate-mode', choices=list(CANDIDATE_MODES),
                    default=TEAM_CANDIDATE_MODE_DEFAULT,
                    help='signals used for referee/goalkeeper/odd-kit candidates')
    ap.add_argument('--team-candidate-action', choices=list(CANDIDATE_ACTIONS),
                    default=TEAM_CANDIDATE_ACTION_DEFAULT,
                    help='crop-level consequence of a candidate flag; fragment-level exclusion applies either way')
    ap.add_argument('--team-calibration', help='human-reviewed validation policy; absent means team abstention')
    ap.add_argument('--team-save-crops', action='store_true',
                    help='write accepted torso crops and a crop index for the existing review/clustering tools')
    a = ap.parse_args()
    if a.sample_every < 1 or a.fragment_gap < 1 or a.max_fragment_seconds <= 0:
        ap.error('Sampling/gap/duration must be positive')
    if not 1 <= a.min_observations <= a.max_observations:
        ap.error('Require 1 <= --min-observations <= --max-observations')
    if a.team_sample_every < 1 or a.team_max_crops_per_fragment < 1 or a.team_max_crops_per_fit_fragment < 1:
        ap.error('Team sampling and crop bounds must be positive')
    if a.team_min_votes < 1:
        ap.error('--team-min-votes must be at least 1')
    if not .5 <= a.team_min_majority <= 1.:
        ap.error('--team-min-majority must be between 0.50 and 1.00')
    video_digest = hashlib.sha256()
    with Path(a.video).open('rb') as video_file:
        for block in iter(lambda: video_file.read(1024 * 1024), b''):
            video_digest.update(block)
    video_sha256 = video_digest.hexdigest()
    team_calibration = load_team_calibration(a.team_calibration, video_sha256)
    out = Path(a.out)
    if out.exists() and any(out.iterdir()):
        ap.error('Output directory must be new or empty; existing evidence will not be overwritten')
    out.mkdir(parents=True, exist_ok=True)
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
    team_policy = CropFilterPolicy()
    team_box_stats = FilterStats(); team_crop_stats = FilterStats()
    team_sampled = 0; team_collect_seconds = 0.; video_wh = None
    samples_path = out/'boxes.jsonl'
    accounting_path = out/'detection_accounting.jsonl'
    accounting_totals = new_accounting_totals(); accounting_settings = None
    prev_tracker_frame_id = 0
    with samples_path.open('w') as log, accounting_path.open('w') as acct_log:
        while True:
            ok, frame = cap.read()
            if not ok or (a.max_frames and frame_idx >= a.max_frames): break
            if video_wh is None: video_wh = (int(frame.shape[1]), int(frame.shape[0]))
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
                    # Team torso crops reuse the already-decoded frame and the emitted
                    # detection row; no second decode and no extra model inference.
                    if is_sampling_turn(detection_index, a.team_sample_every):
                        team_sampled += 1
                        t0 = time.perf_counter()
                        collect_team_crop(f, frame, xyxy,
                                          {'frame': frame_idx, 'timestamp_s': frame_idx/fps,
                                           'detection_id': row['detection_id'],
                                           'detector_confidence': conf},
                                          policy=team_policy, box_stats=team_box_stats,
                                          crop_stats=team_crop_stats,
                                          max_crops=a.team_max_crops_per_fragment)
                        team_collect_seconds += time.perf_counter() - t0
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
    team_report = classify_fragment_teams(
        fragments, policy=team_policy, box_stats=team_box_stats, crop_stats=team_crop_stats,
        collection_seconds=team_collect_seconds, min_votes=a.team_min_votes,
        min_majority=a.team_min_majority, max_per_fragment_fit=a.team_max_crops_per_fit_fragment,
        candidate_mode=a.team_candidate_mode, candidate_action=a.team_candidate_action,
        keep_crops=a.team_save_crops, calibration=team_calibration)
    team_report['input_video_sha256'] = video_sha256
    team_report['crop_collection']['sampled_detections'] = team_sampled
    if a.team_save_crops:
        crop_index = write_team_crop_index(out, fragments, video=a.video, frame_wh=video_wh or (0, 0),
                                           frames=frame_idx, rows_read=accounting_totals['emitted_detections'],
                                           sampled=team_sampled, policy=team_policy)
        team_report['saved_crop_index'] = {'file': 'team_crop_index.json', 'crops_dir': 'team_crops',
                                           'samples': len(crop_index['samples']),
                                           'crop_write_failures': crop_index['crop_write_failures']}
    release_team_crops(fragments)
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
            row.update(team_row_fields(f))
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
    (out/'team_classification_report.json').write_text(json.dumps(team_report, indent=2))
    print(f"Team classification: fit={team_report['fit']['fitted']} "
          f"fingerprint={team_report['fit']['cluster_fingerprint']} "
          f"fragments={team_report['fragment_label_distribution']} "
          f"detections={team_report['detection_label_distribution']}", flush=True)
    if not team_report['fit']['fitted']:
        print(f"WARNING: no team clusters were fitted: {team_report['fit']['error']}", flush=True)
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
    (out/'run_manifest.json').write_text(json.dumps({'video':a.video,'frames':frame_idx,'fps':fps,'detector':a.weights,'embedding':'torchvision ResNet18 ImageNet pretrained, NOT player-ReID trained','trained_in_this_project':False,'ocr':'EasyOCR pretrained' if reader else None,'threshold':a.threshold,'margin':a.margin,'appearance_sampling':{'rule':'per-fragment detection index','sample_every':a.sample_every,'min_observations':a.min_observations,'max_observations':a.max_observations,'min_crop_height':MIN_CROP_HEIGHT,'min_crop_width':MIN_CROP_WIDTH},'detection_accounting':{'enabled':True,'accounting_version':ACCOUNTING_VERSION,'per_frame_file':'detection_accounting.jsonl','summary_file':'detection_accounting_summary.json','settings':accounting_settings,'raw_detections':accounting_totals['raw_detections'],'emitted_detections':accounting_totals['emitted_detections'],'dropped_detections':accounting_totals['dropped_detections']},'schema_version':SCHEMA_VERSION,'output_contract':schema_descriptor(),'team_classification':team_report,'identity_fields_note':'person_uid is reserved and not emitted; team_id is now an independent kit-cluster classification and does not depend on roster identification (the roster-inherited value is kept per fragment under team_classification.roster_inherited_team_id)','scope':'LTPI-inspired prototype; not official LTPI reproduction'},indent=2))
    print(f'Finished: {len(fragments)} fragments. Outputs in {out}')

if __name__=='__main__': main()
