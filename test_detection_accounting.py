"""Stage 2 detection-accounting tests.

Synthetic BYTETracker drives wherever possible; the only model-backed test runs three
frames of an existing clip and is skipped when that clip is absent.
"""
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
import numpy as np
import torch
from output_schema import (
    ACCOUNTING_FIELDS, validate_accounting_record, validate_accounting_rows,
)
from run import (
    RawDetectionCapture, accumulate_accounting, accounting_consistency,
    build_accounting_record, build_accounting_summary, confidence_bands, new_accounting_totals,
)

HIGH, LOW, FPS = 0.25, 0.1, 30.0
TRACKER_ARGS = Namespace(track_high_thresh=HIGH, track_low_thresh=LOW, new_track_thresh=HIGH,
                         track_buffer=30, match_thresh=.8, fuse_score=True)


def det_boxes(dets):
    """Build the exact object type track.py hands to BYTETracker.update: Boxes.cpu().numpy()."""
    from ultralytics.engine.results import Boxes
    if not dets:
        return None
    data = torch.tensor([[x1, y1, x2, y2, conf, 0.0] for (x1, y1, x2, y2), conf in dets], dtype=torch.float32)
    return Boxes(data, orig_shape=(720, 1280)).cpu().numpy()


def det_boxes_torch(dets):
    """The torch-backed Boxes a pre-association on_predict_postprocess_end callback sees."""
    from ultralytics.engine.results import Boxes
    if not dets:
        return None
    data = torch.tensor([[x1, y1, x2, y2, conf, 0.0] for (x1, y1, x2, y2), conf in dets], dtype=torch.float32)
    return Boxes(data, orig_shape=(720, 1280))


def callback_name(callback):
    target = getattr(callback, 'func', callback)
    return getattr(target, '__name__', type(target).__name__)


def new_tracker():
    from ultralytics.trackers.byte_tracker import BYTETracker
    BYTETracker.reset_id()
    return BYTETracker(args=TRACKER_ARGS, frame_rate=30)


def step(tracker, dets, frame_idx, previous_tracker_frame_id):
    """Reproduce run.py's per-frame accounting path, including track.py's skip on empty detections."""
    boxes = det_boxes(dets)
    scores = [] if boxes is None else [float(c) for c in boxes.conf]
    if boxes is None or len(boxes) == 0:
        tracks = np.empty((0, 8), dtype=np.float32)          # track.py:78 continues before update()
    else:
        tracks = tracker.update(boxes, None)
    ids = [] if len(tracks) == 0 else tracks[:, 4].astype(int).tolist()   # result = xyxy + [id, score, cls, idx]
    record = build_accounting_record(
        frame_idx=frame_idx, fps=FPS, scores=scores, emitted_track_ids=ids,
        tracker_frame_id=tracker.frame_id, active_tracks=len(tracker.tracked_stracks),
        lost_tracks=len(tracker.lost_stracks), high_thresh=HIGH, low_thresh=LOW,
        previous_tracker_frame_id=previous_tracker_frame_id)
    return record, ids


def feed(dets_list):
    tracker, records, previous = new_tracker(), [], 0
    for i, dets in enumerate(dets_list):
        record, _ = step(tracker, dets, i, previous)
        records.append(record)
        previous = tracker.frame_id
    return tracker, records


BOX_A = (100, 100, 140, 180)
BOX_B = (500, 300, 560, 420)


def make_record(**overrides):
    """A complete, internally consistent accounting record with optional overrides."""
    raw = overrides.get('raw_detections', 0)
    emitted = overrides.get('emitted_detections', 0)
    record = {
        'frame': 0, 'timestamp_s': 0.0, 'raw_detections': raw,
        'raw_high_conf': overrides.get('raw_high_conf', raw), 'raw_low_conf': 0, 'raw_below_low': 0,
        'emitted_detections': emitted, 'dropped_detections': raw - emitted,
        'tracker_update_skipped': raw == 0, 'tracker_frame_id': 1, 'active_tracks': emitted,
        'lost_tracks': 0, 'emitted_track_ids': list(range(1, emitted + 1)),
    }
    record.update(overrides)
    return record


class BandTests(unittest.TestCase):
    def test_boundaries_are_exclusive_and_match_the_tracker_split(self):
        bands = confidence_bands([HIGH, HIGH - 1e-9, LOW, LOW + 1e-9, 0.0, 0.9], HIGH, LOW)
        self.assertEqual(bands, {'raw_high_conf': 2, 'raw_low_conf': 2, 'raw_below_low': 2})

    def test_bands_sum_to_raw(self):
        scores = list(np.random.default_rng(0).random(200))
        bands = confidence_bands(scores, HIGH, LOW)
        self.assertEqual(sum(bands.values()), len(scores))

    def test_empty_scores(self):
        self.assertEqual(confidence_bands([], HIGH, LOW),
                         {'raw_high_conf': 0, 'raw_low_conf': 0, 'raw_below_low': 0})


class FrameRecordTests(unittest.TestCase):
    def test_empty_frame_skips_the_tracker_update(self):
        tracker, records = feed([[(BOX_A, .9)]])
        record, _ = step(tracker, [], 1, tracker.frame_id)
        self.assertEqual(record['raw_detections'], 0)
        self.assertEqual(record['emitted_detections'], 0)
        self.assertEqual(record['dropped_detections'], 0)
        self.assertTrue(record['tracker_update_skipped'])
        self.assertEqual(record['tracker_frame_id'], 1)

    def test_single_high_confidence_detection_is_emitted(self):
        _, records = feed([[(BOX_A, .9)]])
        self.assertEqual(records[0]['raw_detections'], 1)
        self.assertEqual(records[0]['raw_high_conf'], 1)
        self.assertEqual(records[0]['emitted_detections'], 1)
        self.assertEqual(records[0]['dropped_detections'], 0)
        self.assertFalse(records[0]['tracker_update_skipped'])
        self.assertEqual(records[0]['active_tracks'], 1)

    def test_low_confidence_detection_matched_to_existing_track_is_emitted(self):
        _, records = feed([[(BOX_A, .9)], [(BOX_A, .15)]])
        second = records[1]
        self.assertEqual((second['raw_detections'], second['raw_low_conf']), (1, 1))
        self.assertEqual(second['emitted_detections'], 1)
        self.assertEqual(second['dropped_detections'], 0)

    def test_low_confidence_detection_without_a_track_is_dropped(self):
        _, records = feed([[(BOX_A, .15)]])
        record = records[0]
        self.assertEqual((record['raw_detections'], record['raw_low_conf']), (1, 1))
        self.assertEqual(record['emitted_detections'], 0)
        self.assertEqual(record['dropped_detections'], 1)

    def test_overlapping_detections_are_not_deduplicated_by_the_tracker(self):
        """Measured ByteTrack behaviour: it creates a track per detection, it does not NMS them."""
        _, records = feed([[(BOX_A, .9), ((101, 101, 141, 181), .85)]])
        record = records[0]
        self.assertEqual(record['raw_detections'], 2)
        self.assertEqual(record['emitted_detections'], 2)
        self.assertEqual(record['dropped_detections'], 0)
        self.assertEqual(len(set(record['emitted_track_ids'])), 2)

    def test_track_loss_and_removal_never_leak_into_detection_counters(self):
        frames = [[(BOX_A, .9), (BOX_B, .9)]] + [[(BOX_B, .9)]] * 34
        tracker, records = feed(frames)
        self.assertTrue(all(r['raw_detections'] == r['emitted_detections'] + r['dropped_detections'] for r in records))
        self.assertGreater(max(r['lost_tracks'] for r in records), 0)
        self.assertGreater(len(tracker.removed_stracks), 0)
        for record in records:
            self.assertNotIn('removed_tracks', record)
            self.assertEqual(record['emitted_detections'], len(record['emitted_track_ids']))
        summary = build_accounting_summary(
            totals=summarise(records), frames=len(records), fps=FPS, video='v', settings={},
            final_tracker_frame_id=tracker.frame_id, boxes_rows=sum(r['emitted_detections'] for r in records),
            identified_rows=sum(r['emitted_detections'] for r in records))
        self.assertIn('removed_tracks_total', summary['not_measured'])

    def test_consecutive_zero_detection_frames(self):
        tracker, records = feed([[(BOX_A, .9)], [], [], []])
        self.assertEqual([r['raw_detections'] for r in records[1:]], [0, 0, 0])
        self.assertTrue(all(r['tracker_update_skipped'] for r in records[1:]))
        self.assertEqual([r['tracker_frame_id'] for r in records[1:]], [1, 1, 1])
        self.assertEqual(tracker.frame_id, 1)

    def test_per_frame_arithmetic_over_a_random_sequence(self):
        rng = np.random.default_rng(7)
        frames = [[(tuple(float(v) for v in rng.integers(0, 400, 4) + [0, 0, 60, 90]), float(c))
                   for c in rng.random(n)] for n in rng.integers(0, 6, 40)]
        _, records = feed(frames)
        for record in records:
            self.assertEqual(validate_accounting_record(record), [])
            self.assertEqual(record['dropped_detections'],
                             record['raw_detections'] - record['emitted_detections'])
            self.assertLessEqual(record['emitted_detections'], record['raw_detections'])

    def test_record_contains_exactly_the_contract_fields(self):
        _, records = feed([[(BOX_A, .9)]])
        self.assertEqual(set(records[0]), set(ACCOUNTING_FIELDS))


def summarise(records):
    totals = new_accounting_totals()
    for record in records:
        accumulate_accounting(totals, record)
    return totals


class AggregateTests(unittest.TestCase):
    def test_totals_rates_and_histogram(self):
        records = [make_record(frame=0, raw_detections=4, emitted_detections=3),
                   make_record(frame=1, raw_detections=0, emitted_detections=0),
                   make_record(frame=2, raw_detections=2, emitted_detections=2)]
        totals = summarise(records)
        self.assertEqual(totals['raw_detections'], 6)
        self.assertEqual(totals['emitted_detections'], 5)
        self.assertEqual(totals['dropped_detections'], 1)
        self.assertEqual(totals['frames_raw_zero'], 1)
        self.assertEqual(totals['frames_with_drop'], 1)
        self.assertEqual(totals['max_dropped_detections'], 1)
        self.assertEqual(dict(totals['drop_histogram']), {1: 1, 0: 2})
        summary = build_accounting_summary(
            totals=totals, frames=3, fps=FPS, video='v', settings={'nms_conf': .3},
            final_tracker_frame_id=2, boxes_rows=5, identified_rows=5)
        self.assertAlmostEqual(summary['rates']['emission_rate'], 5 / 6)
        self.assertAlmostEqual(summary['rates']['drop_rate'], 1 / 6)
        self.assertTrue(summary['consistency']['all_hold'])
        self.assertEqual(summary['consistency']['violations'], [])

    def test_rates_are_none_without_raw_detections(self):
        totals = summarise([make_record()])
        summary = build_accounting_summary(totals=totals, frames=1, fps=FPS, video='v', settings={},
                                           final_tracker_frame_id=0, boxes_rows=0, identified_rows=0)
        self.assertIsNone(summary['rates']['emission_rate'])
        self.assertIsNone(summary['rates']['drop_rate'])

    def test_consistency_flags_a_row_count_mismatch(self):
        totals = summarise([make_record(raw_detections=2, emitted_detections=1)])
        result = accounting_consistency(totals, frames=1, boxes_rows=5, identified_rows=5,
                                        final_tracker_frame_id=1)
        self.assertFalse(result['all_hold'])
        self.assertIn('emitted_equals_boxes_rows', result['violations'])

    def test_summary_documents_unmeasurable_counters(self):
        totals = summarise([make_record(raw_detections=1, emitted_detections=0)])
        summary = build_accounting_summary(totals=totals, frames=1, fps=FPS, video='v', settings={},
                                           final_tracker_frame_id=1, boxes_rows=0, identified_rows=0)
        self.assertEqual(set(summary['not_measured']),
                         {'per_detection_drop_reason', 'removed_tracks_total', 'low_confidence_second_association'})


class InvalidRecordTests(unittest.TestCase):
    def base(self):
        return make_record(raw_detections=3, emitted_detections=2)

    def assert_error(self, mutate, needle):
        record = self.base()
        mutate(record)
        errors = validate_accounting_record(record)
        self.assertTrue(any(needle in e for e in errors), f'{needle!r} not in {errors}')

    def test_bands_not_summing(self):
        self.assert_error(lambda r: r.update(raw_low_conf=1), 'confidence bands')

    def test_emitted_exceeds_raw(self):
        self.assert_error(lambda r: r.update(emitted_detections=5, emitted_track_ids=[1, 2, 3, 4, 5]), 'exceeds raw')

    def test_dropped_not_raw_minus_emitted(self):
        self.assert_error(lambda r: r.update(dropped_detections=0), 'dropped 0 != raw 3 - emitted 2')

    def test_ids_count_mismatch(self):
        self.assert_error(lambda r: r.update(emitted_track_ids=[1]), 'emitted_track_ids has 1 entries')

    def test_duplicate_ids(self):
        self.assert_error(lambda r: r.update(emitted_track_ids=[1, 1]), 'duplicates')

    def test_missing_field(self):
        self.assert_error(lambda r: r.pop('lost_tracks'), 'missing field lost_tracks')

    def test_negative_count(self):
        self.assert_error(lambda r: r.update(raw_high_conf=-1), 'must be >= 0')

    def test_non_bool_skip_flag(self):
        self.assert_error(lambda r: r.update(tracker_update_skipped='no'), 'must be a bool')

    def test_skip_flag_must_match_tracker_counter(self):
        record = self.base()
        errors = validate_accounting_record(record, previous_tracker_frame_id=1)
        self.assertTrue(any('tracker_update_skipped' in e for e in errors), errors)

    def test_build_rejects_inconsistent_record(self):
        with self.assertRaises(RuntimeError):
            build_accounting_record(frame_idx=0, fps=FPS, scores=[.9, .9], emitted_track_ids=[1, 2, 3],
                                    tracker_frame_id=1, active_tracks=1, lost_tracks=0,
                                    high_thresh=HIGH, low_thresh=LOW, previous_tracker_frame_id=0)


class RowsFileTests(unittest.TestCase):
    def test_sequential_frames_and_totals(self):
        rows = [make_record(raw_detections=2, emitted_detections=1),
                make_record(frame=1, timestamp_s=1 / FPS, tracker_update_skipped=True)]
        result = validate_accounting_rows(rows, expected_frames=2, boxes_rows=1, identified_rows=1)
        self.assertTrue(result['valid'], result['errors'])
        self.assertEqual(result['totals']['frames_tracker_update_skipped'], 1)

    def test_non_sequential_frame_index(self):
        rows = [make_record(frame=5, tracker_update_skipped=False)]
        result = validate_accounting_rows(rows, expected_frames=1)
        self.assertFalse(result['valid'])
        self.assertTrue(any('expected 0' in e for e in result['errors']), result['errors'])

    def test_wrong_record_count_and_row_totals(self):
        result = validate_accounting_rows([], expected_frames=3, boxes_rows=2, identified_rows=2)
        self.assertEqual(result['records'], 0)
        self.assertTrue(any('expected one per processed frame (3)' in e for e in result['errors']))
        self.assertTrue(any('!= boxes.jsonl rows 2' in e for e in result['errors']))


class CallbackOrderTests(unittest.TestCase):
    class Predictor:
        def __init__(self, results):
            self.results = results

    class Result:
        def __init__(self, boxes):
            self.boxes = boxes

    def test_capture_sees_detections_before_association_rewrites_them(self):
        boxes = det_boxes_torch([(BOX_A, .9), (BOX_B, .8)])
        capture = RawDetectionCapture()
        predictor = self.Predictor([self.Result(boxes)])

        def tracker_callback(p):        # stands in for ultralytics trackers/track.py
            p.results[0] = self.Result(det_boxes_torch([(BOX_A, .9)]))

        for callback in (capture, tracker_callback):    # registration order, as ultralytics does
            callback(predictor)
        self.assertEqual(len(capture.confidences), 2)
        self.assertEqual(sorted(round(c, 5) for c in capture.confidences), [.8, .9])

    def test_capture_records_no_ids_before_association(self):
        boxes = det_boxes_torch([(BOX_A, .9)])
        capture = RawDetectionCapture()
        capture(self.Predictor([self.Result(boxes)]))
        self.assertIsNotNone(capture.confidences)
        self.assertIsNone(boxes.id)

    def test_capture_with_no_results(self):
        capture = RawDetectionCapture()
        capture(self.Predictor([]))
        self.assertEqual(capture.confidences, [])

    def test_capture_with_no_boxes(self):
        capture = RawDetectionCapture()
        capture(self.Predictor([self.Result(None)]))
        self.assertEqual(capture.confidences, [])


@unittest.skipUnless(Path('datasets/infactory_soccer_events/data/3.mp4').exists()
                     and Path('yolov8n.pt').exists(), 'clip or weights unavailable')
class RealPipelineTests(unittest.TestCase):
    def test_callback_is_registered_ahead_of_the_tracker_callback(self):
        from ultralytics import YOLO
        import cv2
        model = YOLO('yolov8n.pt')
        capture = RawDetectionCapture()
        model.add_callback('on_predict_postprocess_end', capture)
        cap = cv2.VideoCapture('datasets/infactory_soccer_events/data/3.mp4')
        try:
            ok, frame = cap.read()
            self.assertTrue(ok)
            result = model.track(frame, persist=True, tracker='bytetrack.yaml', classes=[0], conf=.3,
                                 imgsz=960, device='cpu', verbose=False)[0]
            names = [callback_name(cb) for cb in model.predictor.callbacks['on_predict_postprocess_end']]
            capture_index = [i for i, cb in enumerate(model.predictor.callbacks['on_predict_postprocess_end'])
                             if cb is capture][0]
            tracker_indices = [i for i, name in enumerate(names) if name == 'on_predict_postprocess_end']
            self.assertLess(capture_index, max(tracker_indices))
            self.assertIsNotNone(capture.confidences)
            emitted = 0 if result.boxes is None or result.boxes.id is None else len(result.boxes)
            self.assertGreaterEqual(len(capture.confidences), emitted)
            self.assertGreater(len(capture.confidences), 0)
        finally:
            cap.release()


class ValidatorReportTests(unittest.TestCase):
    """analyse_detection_accounting() is the report-only checker required by the Stage 2 contract."""
    def setUp(self):
        from validate_tracking_outputs import analyse_detection_accounting
        self.analyse = analyse_detection_accounting
        self.dets = [[(BOX_A, .9), (BOX_B, .2)],                     # one high, one low -> drop risk
                     [(BOX_A, .8), (BOX_B, .7)],
                     [],
                     [(BOX_A, .95)]]
        tracker, records = feed(self.dets)
        self.records = records
        self.boxes_rows = sum(r['emitted_detections'] for r in records)
        self.summary = build_accounting_summary(
            totals=summarise(records), frames=len(records), fps=FPS, video='x.mp4',
            settings={'track_high_thresh': HIGH, 'track_low_thresh': LOW},
            final_tracker_frame_id=tracker.frame_id,
            boxes_rows=self.boxes_rows, identified_rows=self.boxes_rows)

    def write(self, directory, records=None, summary=None):
        (directory / 'detection_accounting.jsonl').write_text(
            ''.join(json.dumps(r) + '\n' for r in (self.records if records is None else records)))
        (directory / 'detection_accounting_summary.json').write_text(
            json.dumps(self.summary if summary is None else summary))

    def test_all_ten_checks_pass_on_a_consistent_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            self.write(directory)
            result = self.analyse(directory, boxes_rows=self.boxes_rows,
                                  identified_rows=self.boxes_rows, expected_frames=len(self.records))
            self.assertEqual(sorted(result['checks']), sorted([
                'bands_are_mutually_exclusive_and_sum_to_raw', 'dropped_equals_raw_minus_emitted',
                'emitted_equals_boxes_rows', 'emitted_equals_identified_rows',
                'emitted_ids_match_emitted_count', 'emitted_le_raw', 'frame_indices_sequential',
                'one_record_per_processed_frame', 'skipped_equals_zero_raw_frames',
                'summary_agrees_with_records']))
            self.assertEqual(result['check_count_failed'], 0, result['errors'])
            self.assertTrue(result['valid'], result['errors'])

    def test_missing_accounting_file_is_reported_not_repaired(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self.analyse(Path(tmp), boxes_rows=0, identified_rows=0, expected_frames=4)
            self.assertFalse(result['exists'])
            self.assertIn('missing', result['errors'][0])

    def test_bad_arithmetic_and_summary_disagreement_are_both_flagged(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            records = [dict(r) for r in self.records]
            records[0]['dropped_detections'] += 1
            summary = json.loads(json.dumps(self.summary))
            summary['totals']['max_dropped_detections'] = 99
            self.write(directory, records=records, summary=summary)
            before = (directory / 'detection_accounting.jsonl').read_text()
            result = self.analyse(directory, boxes_rows=self.boxes_rows,
                                  identified_rows=self.boxes_rows, expected_frames=len(records))
            self.assertFalse(result['checks']['dropped_equals_raw_minus_emitted'])
            self.assertFalse(result['checks']['summary_agrees_with_records'])
            self.assertTrue(any('dropped 2 != raw' in e for e in result['errors']), result['errors'])
            self.assertTrue(any('max_dropped_detections' in e for e in result['errors']), result['errors'])
            self.assertEqual((directory / 'detection_accounting.jsonl').read_text(), before)

    def test_emitted_total_must_equal_output_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            self.write(directory)
            result = self.analyse(directory, boxes_rows=self.boxes_rows + 3,
                                  identified_rows=self.boxes_rows, expected_frames=len(self.records))
            self.assertFalse(result['checks']['emitted_equals_boxes_rows'])
            self.assertTrue(result['checks']['emitted_equals_identified_rows'])
            self.assertTrue(any('!= boxes.jsonl rows' in e for e in result['errors']), result['errors'])

    def test_record_count_is_checked_against_processed_frames(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            self.write(directory, records=self.records[:-1])
            result = self.analyse(directory, boxes_rows=0, identified_rows=0,
                                  expected_frames=len(self.records))
            self.assertFalse(result['checks']['one_record_per_processed_frame'])


if __name__ == '__main__':
    unittest.main()
