"""Stage 6 integration tests: run.py main() end to end on a SYNTHETIC clip.

The detector, tracker and appearance encoder are scripted stand-ins, so these
tests need no model weights and say nothing about detection, tracking or
identity accuracy. They check wiring, schema versions, provenance, abstention,
validator acceptance and that the identity step is purely additive.
"""

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import cv2
import numpy as np
import torch
from ultralytics.engine.results import Boxes

import run
from identity_association import EXTENSION_FIELDS, person_uid
from validate_pillar1_regression import compare_runs

WIDTH, HEIGHT, FPS, FRAMES = 320, 240, 15.0, 90
# Synthetic script: (track_id, first_frame, last_frame, x, colour BGR).
# Track 1 and track 3 are drawn as the same synthetic person (red) at different times;
# track 2 (blue) is visible for the whole clip. Every fragment has >= 31 detections so
# appearance sampling (every 15th detection, >= 3 crops) yields an embedding.
SCRIPT = [(1, 0, 34, 40, (0, 0, 220)), (2, 0, 89, 200, (220, 0, 0)), (3, 50, 89, 60, (0, 0, 220))]


def boxes_at(frame_idx):
    out = []
    for tid, first, last, x, colour in SCRIPT:
        if first <= frame_idx <= last:
            out.append((tid, [float(x), 60.0, float(x + 40), 180.0], 0.9, colour))
    return out


def write_video(path):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT))
    for i in range(FRAMES):
        frame = np.full((HEIGHT, WIDTH, 3), (40, 140, 40), dtype=np.uint8)
        for _, box, _, colour in boxes_at(i):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, -1)
        writer.write(frame)
    writer.release()


class FakeTracker:
    def __init__(self):
        self.args = SimpleNamespace(track_high_thresh=0.25, track_low_thresh=0.1, new_track_thresh=0.25,
                                    match_thresh=0.8, track_buffer=30, fuse_score=True)
        self.frame_id = 0
        self.tracked_stracks, self.lost_stracks = [], []


class FakeYOLO:
    """Scripted stand-in for ultralytics.YOLO: replays SCRIPT, runs the raw-detection hook."""

    def __init__(self, weights):
        self.names = {0: "person"}
        self.callbacks = []
        self.calls = 0
        self.predictor = SimpleNamespace(args=SimpleNamespace(conf=0.3, iou=0.7, max_det=300, imgsz=960),
                                         trackers=[FakeTracker()], results=None)

    def add_callback(self, event, callback):
        assert event == "on_predict_postprocess_end"
        self.callbacks.append(callback)

    def track(self, frame, **kwargs):
        dets = boxes_at(self.calls)
        self.calls += 1
        raw = torch.tensor([[*b, c, 0.0] for _, b, c, _ in dets], dtype=torch.float32).reshape(-1, 6)
        self.predictor.results = [SimpleNamespace(boxes=Boxes(raw, orig_shape=(HEIGHT, WIDTH)))]
        for callback in self.callbacks:
            callback(self.predictor)
        tracker = self.predictor.trackers[0]
        if dets:  # ultralytics skips tracker.update() on frames without detections
            tracker.frame_id += 1
        tracker.tracked_stracks = list(range(len(dets)))
        tracked = torch.tensor([[*b, tid, c, 0.0] for tid, b, c, _ in dets], dtype=torch.float32).reshape(-1, 7)
        return [SimpleNamespace(boxes=Boxes(tracked, orig_shape=(HEIGHT, WIDTH)) if dets else None)]


class FakeEncoder:
    """Deterministic colour-mean 'embedding'; stands in for ImageNet ResNet18."""

    def __init__(self, device):
        pass

    def __call__(self, crop):
        vector = np.concatenate([crop.reshape(-1, 3).mean(axis=0), [1.0]]).astype(np.float32)
        return vector / np.linalg.norm(vector)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


class PipelineIdentityIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.video = cls.root / "synthetic.mp4"
        write_video(cls.video)
        cls.plain = cls.root / "no_review"
        cls.run_pipeline(cls.plain)
        cls.boxes_sha = sha256(cls.plain / "boxes.jsonl")
        cls.review = cls.root / "synthetic_review.json"
        cls.review.write_text(json.dumps(cls.review_document(cls.boxes_sha)), encoding="utf-8")
        cls.reviewed = cls.root / "with_review"
        cls.run_pipeline(cls.reviewed, "--identity-review", str(cls.review))

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    @classmethod
    def run_pipeline(cls, out, *extra):
        argv = ["run.py", "--video", str(cls.video), "--out", str(out), "--render", *extra]
        with mock.patch.object(run, "YOLO", FakeYOLO), mock.patch.object(run, "Encoder", FakeEncoder), \
                mock.patch.object(sys, "argv", argv):
            run.main()

    @classmethod
    def review_document(cls, boxes_sha, video_sha=None):
        fragments = [{"fragment_id": fid, "single_person": True, "full_span_reviewed": True,
                      "shot_id": "synthetic-shot-1"} for fid in ("F000001", "F000002", "F000003")]
        return {
            "kind": "human_fragment_identity_review",
            "video_sha256": video_sha or sha256(cls.video), "boxes_sha256": boxes_sha,
            "reviewer": "Synthetic fixture (not a real human review)",
            "reviewed_at": "2026-11-15",
            "fragments": fragments,
            "pairs": [
                {"left": "F000001", "right": "F000003", "decision": "same",
                 "reason": "Synthetic: scripted as the same drawn person."},
                {"left": "F000001", "right": "F000002", "decision": "different",
                 "reason": "Synthetic: scripted as different drawn people."},
            ],
        }

    def fragments(self, run_dir):
        return {f["fragment_id"]: f for f in json.loads((run_dir / "fragments.json").read_text())}

    def validate_cli(self, run_dir):
        out = self.root / f"{run_dir.name}_validation_{len(list(self.root.glob('*_validation_*')))}.json"
        result = subprocess.run(
            [sys.executable, "-B", str(Path(run.__file__).with_name("validate_tracking_outputs.py")),
             "--video", str(self.video), "--run-dir", str(run_dir), "--out", str(out), "--require-render"],
            capture_output=True, text=True, check=False)
        return result, json.loads(out.read_text())

    # Scripted scenario sanity -------------------------------------------------
    def test_scenario_produces_three_embedded_fragments(self):
        fragments = self.fragments(self.plain)
        self.assertEqual(sorted(fragments), ["F000001", "F000002", "F000003"])
        for fragment in fragments.values():
            self.assertEqual(fragment["appearance_evidence"]["status"], "ok")

    # No review: abstention -------------------------------------------------------
    def test_no_review_keeps_every_person_uid_null(self):
        report = json.loads((self.plain / "identity_association_report.json").read_text())
        self.assertEqual(report["review_status"], "not_supplied")
        self.assertIn("no reviewed association was performed", report["review_note"])
        self.assertEqual(report["counts"]["groups"], 0)
        for fragment in self.fragments(self.plain).values():
            self.assertIsNone(fragment["person_uid"])
            self.assertEqual(fragment["association_status"], "unresolved")
            self.assertEqual(fragment["association_reason"], "no_human_review")
            self.assertEqual(fragment["association_method"], "abstain")

    def test_raw_boxes_stay_schema_1_0_and_enriched_rows_carry_1_1_fields(self):
        for row in read_jsonl(self.plain / "boxes.jsonl"):
            self.assertNotIn("schema_version", row)
            self.assertNotIn("person_uid", row)
        rows = read_jsonl(self.plain / "identified_boxes.jsonl")
        self.assertTrue(rows)
        for row in rows:
            self.assertTrue(set(EXTENSION_FIELDS) <= set(row))
            self.assertEqual(row["schema_version"], "1.1")

    def test_manifest_records_versions_and_provenance(self):
        manifest = json.loads((self.plain / "run_manifest.json").read_text())
        self.assertEqual(manifest["schema_version"], "1.1")
        self.assertEqual(manifest["artifact_schema_versions"], run.ARTIFACT_SCHEMA_VERSIONS)
        self.assertEqual(manifest["input_video_sha256"], sha256(self.video))
        self.assertEqual(manifest["boxes_sha256"], self.boxes_sha)
        self.assertEqual(manifest["raw_output_contract"]["schema_version"], "1.0")
        identity = manifest["identity_association"]
        self.assertEqual(identity["review_status"], "not_supplied")
        self.assertIsNone(identity["review_sha256"])
        self.assertTrue(identity["cross_file_validation"]["valid"], identity["cross_file_validation"])
        self.assertIsNone(identity["accuracy_claim"])

    def test_validator_accepts_no_review_run(self):
        result, report = self.validate_cli(self.plain)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue(report["acceptance"]["passed"], report["acceptance"]["failures"])
        self.assertTrue(report["identity_association"]["applicable"])

    # Reviewed association -------------------------------------------------------
    def test_review_links_only_the_accepted_pair(self):
        fragments = self.fragments(self.reviewed)
        expected = person_uid(sha256(self.video), self.boxes_sha, ["F000001", "F000003"])
        self.assertEqual(fragments["F000001"]["person_uid"], expected)
        self.assertEqual(fragments["F000003"]["person_uid"], expected)
        self.assertIsNone(fragments["F000002"]["person_uid"])
        report = json.loads((self.reviewed / "identity_association_report.json").read_text())
        self.assertEqual(report["review_status"], "applied")
        self.assertEqual(report["provenance"]["review"]["sha256"], sha256(self.review))
        pairs = {(p["left"], p["right"]): p for p in report["pairs"]}
        self.assertTrue(pairs[("F000001", "F000003")]["accepted"])
        rejected = pairs[("F000001", "F000002")]
        self.assertFalse(rejected["accepted"])
        self.assertIn("review_decision_different", rejected["reasons"])
        self.assertIn("inclusive_frame_overlap", rejected["reasons"])

    def test_identity_fields_untouched_by_association(self):
        plain, reviewed = self.fragments(self.plain), self.fragments(self.reviewed)
        for fid, before in plain.items():
            after = reviewed[fid]
            for key in ("player_id", "team_id", "jersey", "embedding", "start_frame", "end_frame",
                        "track_id", "appearance_evidence", "team_classification"):
                self.assertEqual(before[key], after[key], (fid, key))

    def test_reviewed_run_is_additive_against_the_no_review_run(self):
        for name in ("boxes.jsonl", "detection_accounting.jsonl", "detection_accounting_summary.json",
                     "players.json"):
            self.assertEqual(sha256(self.plain / name), sha256(self.reviewed / name), name)
        report = compare_runs(self.plain, self.reviewed, allow_render_change=True)
        # Both runs carry the 1.1 extension fields, so the only permitted differences
        # between them are those five fields; every other check must pass.
        for kind in ("detections", "fragments"):
            changed = {change["field"] for change in report[kind]["field_changes"]}
            self.assertTrue(changed, kind)
            self.assertLessEqual(changed, set(EXTENSION_FIELDS), kind)
            self.assertEqual(report[kind]["identity_changes"], [], kind)
        other_failures = [name for name, passed in report["checks"].items()
                          if not passed and not name.endswith(".original_fields")]
        self.assertEqual(other_failures, [])

    def test_validator_accepts_reviewed_run(self):
        result, report = self.validate_cli(self.reviewed)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue(report["acceptance"]["passed"], report["acceptance"]["failures"])
        self.assertEqual(report["identity_association"]["associated_fragments"], 2)

    def test_render_shows_person_uid_only_for_associated_rows(self):
        row = {"fragment_id": "F000001", "player_id": "UNK", "association_status": "associated",
               "person_uid": "P" + "a" * 32}
        self.assertEqual(run.render_label(row, True), "F000001 UNK P" + "a" * 8)
        self.assertEqual(run.render_label(row, False), "F000001 UNK")
        unresolved = dict(row, association_status="unresolved", person_uid=None)
        self.assertEqual(run.render_label(unresolved, True), "F000001 UNK")

    # Fail closed -----------------------------------------------------------------
    def test_stale_review_fails_closed(self):
        stale = self.root / "stale_review.json"
        stale.write_text(json.dumps(self.review_document("c" * 64)), encoding="utf-8")
        out = self.root / "stale_run"
        with self.assertRaises(SystemExit) as caught:
            self.run_pipeline(out, "--identity-review", str(stale))
        self.assertIn("Identity review rejected", str(caught.exception))
        self.assertFalse((out / "fragments.json").exists())
        self.assertFalse((out / "identity_association_report.json").exists())

    def test_missing_review_file_is_a_usage_error(self):
        with self.assertRaises(SystemExit):
            self.run_pipeline(self.root / "missing_review_run", "--identity-review",
                              str(self.root / "absent.json"))


if __name__ == "__main__":
    unittest.main()
