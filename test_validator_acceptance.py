"""Stage 5B: aggregate acceptance gate of validate_tracking_outputs (synthetic only)."""

import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from validate_tracking_outputs import assess_acceptance


def good_report(render=True):
    report = {
        "schema": {
            "boxes.jsonl": {"exists": True, "strict_level": {"valid": True}},
            "identified_boxes.jsonl": {"exists": True, "strict_level": {"valid": True}},
            "run_manifest.json": {"exists": True},
        },
        "output_consistency": {
            "row_count_equal": True, "detection_ids_match": True, "fragment_ids_unique": True,
            "detection_id_unique_in_boxes": True, "detection_id_unique_in_identified": True,
            "fragment_ids_referenced_by_boxes_not_in_fragments": [],
            "spatiotemporal_field_drift_between_files": {},
        },
        "timestamps": {key: 0 for key in (
            "missing_timestamp_s", "non_numeric_timestamp_s", "negative_timestamp_s",
            "non_finite_timestamp_s", "timestamp_not_equal_to_frame_divided_by_fps",
            "non_monotonic_within_same_frame")},
        "fragments": {"inverted_fragments": 0, "unique_fragment_ids": 3, "fragments": 3},
        "detection_accounting": {"valid": True},
        "identity_association": {"valid": True},
        "render": ({"exists": True, "opened": True, "matches_frames_processed": True,
                    "geometry_matches_source": True} if render else {"exists": False}),
    }
    return report


class AcceptanceGateTests(unittest.TestCase):
    def test_consistent_report_passes(self):
        result = assess_acceptance(good_report())
        self.assertTrue(result["passed"], result["failures"])
        self.assertEqual(result["failures"], [])
        self.assertIn("not tracking or identity accuracy", result["note"])

    def test_each_failure_is_reported_by_name(self):
        cases = [
            (("schema", "boxes.jsonl", "strict_level", "valid"), False, "schema.boxes.jsonl.strict_valid"),
            (("schema", "identified_boxes.jsonl", "exists"), False, "schema.identified_boxes.jsonl.exists"),
            (("output_consistency", "row_count_equal"), False, "consistency.row_count_equal"),
            (("output_consistency", "spatiotemporal_field_drift_between_files"), {"bbox_xyxy": 2},
             "consistency.no_field_drift"),
            (("output_consistency", "fragment_ids_referenced_by_boxes_not_in_fragments"), ["F9"],
             "consistency.no_unknown_fragment_references"),
            (("timestamps", "negative_timestamp_s"), 1, "timestamps.negative_timestamp_s_zero"),
            (("fragments", "inverted_fragments"), 1, "fragments.no_inverted"),
            (("fragments", "unique_fragment_ids"), 2, "fragments.ids_unique"),
            (("detection_accounting", "valid"), False, "detection_accounting.valid"),
            (("identity_association", "valid"), False, "identity_association.valid"),
            (("render", "matches_frames_processed"), False, "render.frames_match"),
        ]
        for path, value, name in cases:
            with self.subTest(name=name):
                report = copy.deepcopy(good_report())
                target = report
                for key in path[:-1]:
                    target = target[key]
                target[path[-1]] = value
                result = assess_acceptance(report)
                self.assertFalse(result["passed"])
                self.assertIn(name, result["failures"])

    def test_missing_sections_fail_closed(self):
        for section in ("schema", "output_consistency", "timestamps", "fragments",
                        "detection_accounting", "identity_association"):
            with self.subTest(section=section):
                report = good_report()
                del report[section]
                self.assertFalse(assess_acceptance(report)["passed"])

    def test_render_optional_unless_required(self):
        self.assertTrue(assess_acceptance(good_report(render=False))["passed"])
        result = assess_acceptance(good_report(render=False), require_render=True)
        self.assertFalse(result["passed"])
        self.assertEqual(result["failures"], ["render.present_if_required"])


class AcceptanceCliTests(unittest.TestCase):
    def test_incomplete_run_writes_report_and_exits_nonzero(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "synthetic.mp4"
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 15, (64, 48))
            for _ in range(3):
                writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
            writer.release()
            run = root / "run"
            run.mkdir()
            (run / "run_manifest.json").write_text(json.dumps({"fps": 15, "frames": 3}), encoding="utf-8")
            out = root / "validation.json"
            result = subprocess.run(
                [sys.executable, "-B", str(Path(__file__).with_name("validate_tracking_outputs.py")),
                 "--video", str(video), "--run-dir", str(run), "--out", str(out)],
                capture_output=True, text=True, check=False)
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertIn("Acceptance: FAIL", result.stdout)
            report = json.loads(out.read_text(encoding="utf-8"))
            self.assertFalse(report["acceptance"]["passed"])
            self.assertIn("schema.boxes.jsonl.exists", report["acceptance"]["failures"])
            self.assertIn("detection_accounting.valid", report["acceptance"]["failures"])


if __name__ == "__main__":
    unittest.main()
