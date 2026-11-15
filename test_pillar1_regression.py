import copy
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import validate_pillar1_regression as validator
from validate_pillar1_regression import EXACT_FILES, REQUIRED_FILES, compare_runs


class Pillar1RegressionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.baseline, self.candidate = self.root / "baseline", self.root / "candidate"
        self.baseline.mkdir()
        self.candidate.mkdir()
        self.rows = []
        self.fragments = []
        for i in range(2):
            identity = {"player_id": "UNK", "jersey": None, "team_id": "UNKNOWN",
                        "identification_confidence": 0.0, "confidence": 0.0,
                        "identification_reason": "insufficient_appearance_evidence",
                        "reason": "insufficient_appearance_evidence"}
            self.rows.append({"detection_id": f"D{i}", "fragment_id": f"F{i}",
                              "frame": i, "timestamp_s": i / 15, "track_id": i + 1,
                              "bbox_xyxy": [0.0, 1.0, 20.0, 40.0],
                              "detector_confidence": 0.9, "class_id": 0, **identity})
            self.fragments.append({"fragment_id": f"F{i}", "track_id": i + 1,
                                   "start_frame": i, "end_frame": i,
                                   "start_s": i / 15, "end_s": i / 15, "numbers": [],
                                   "embedding": [0.1, 0.2],
                                   "appearance_evidence": {"detections": 1, "samples_used": 1,
                                                           "reason": "insufficient_observations"},
                                   **identity})
        for directory in (self.baseline, self.candidate):
            self.write_records("detections", self.rows, directory)
            self.write_records("fragments", self.fragments, directory)
            boxes = [{key: value for key, value in row.items()
                      if key in {"detection_id", "fragment_id", "track_id", "frame",
                                 "timestamp_s", "bbox_xyxy", "detector_confidence", "class_id"}}
                     for row in self.rows]
            (directory / "boxes.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in boxes), encoding="utf-8")
            (directory / "detection_accounting.jsonl").write_text(
                '{"frame": 0, "raw": 1, "emitted": 1, "dropped": 0}\n', encoding="utf-8")
            (directory / "detection_accounting_summary.json").write_text(
                '{"raw_detections": 2, "emitted_detections": 2, "dropped_detections": 0}',
                encoding="utf-8")
            (directory / "players.json").write_text('{"players": []}', encoding="utf-8")
            (directory / "identified.mp4").write_bytes(b"synthetic render bytes")
            (directory / "run_manifest.json").write_text('{"frames": 2}', encoding="utf-8")
        candidate_rows = copy.deepcopy(self.rows)
        candidate_fragments = copy.deepcopy(self.fragments)
        for row in candidate_rows:
            row.update(team_score=0.0, team_evidence_status="insufficient_crops",
                       team_score_kind="distance_to_radius_heuristic_not_probability")
        for fragment in candidate_fragments:
            fragment["team_classification"] = {
                "team_id": "UNKNOWN", "score": 0.0, "evidence_status": "insufficient_crops",
                "score_kind": "distance_to_radius_heuristic_not_probability",
                "rejection_reasons": ["too_few_votes"], "score_before_candidate_exclusion": 0.8}
        self.write_records("detections", candidate_rows)
        self.write_records("fragments", candidate_fragments)

    def write_records(self, kind, rows, directory=None):
        path = (directory or self.candidate) / validator.RECORD_FILES[kind][0]
        text = ("".join(json.dumps(row) + "\n" for row in rows) if kind == "detections"
                else json.dumps(rows))
        path.write_text(text, encoding="utf-8")

    def records(self, kind):
        text = (self.candidate / validator.RECORD_FILES[kind][0]).read_text(encoding="utf-8")
        return [json.loads(line) for line in text.splitlines()] if kind == "detections" else json.loads(text)

    def compare(self, **kwargs):
        return compare_runs(self.baseline, self.candidate, **kwargs)

    def cli(self, *extra, out=None, baseline=None):
        return subprocess.run(
            [sys.executable, "-B", str(Path(validator.__file__).resolve()),
             "--baseline", str(baseline or self.baseline), "--candidate", str(self.candidate),
             "--out", str(out or self.root / "report.json"), *extra],
            capture_output=True, text=True, check=False)

    def assert_passed(self, report):
        self.assertTrue(report["passed"], report["errors"])

    def test_legacy_baseline_and_new_diagnostics_pass(self):
        report = self.compare()
        self.assert_passed(report)
        self.assertEqual(report["detections"]["team_distributions"],
                         {"baseline": {"UNKNOWN": 2}, "candidate": {"UNKNOWN": 2}})
        self.assertIn("team_classification", report["fragments"]["new_fields"])
        self.assertEqual(report["render"]["status"], "identical")

    def test_changed_labels_and_scores_are_not_identity_changes(self):
        for kind in validator.RECORD_FILES:
            rows = self.records(kind)
            rows[0]["team_id"] = "TEAM_B"
            if kind == "detections":
                rows[0].update(team_score=0.8, team_evidence_status="appearance_votes")
            else:
                rows[0]["team_classification"].update(
                    team_id="TEAM_B", score=0.8, evidence_status="appearance_votes")
            self.write_records(kind, rows)
        report = self.compare()
        self.assert_passed(report)
        for kind in validator.RECORD_FILES:
            section = report[kind]
            self.assertEqual(section["identity_changes"], [])
            self.assertEqual(section["field_changes"], [])
            self.assertEqual(section["team_distributions"]["candidate"], {"TEAM_B": 1, "UNKNOWN": 1})
            change = next(change for change in section["team_changes"] if change["label_changed"])
            self.assertTrue(change["score_changed"])
            self.assertEqual(change["baseline"]["score"], None)
            self.assertEqual(change["candidate"]["score"], 0.8)
            self.assertIn("appearance_votes", json.dumps(change["candidate_evidence"]))
        self.assertEqual(report["fragments"]["team_changes"][0]["candidate_evidence"]["rejection_reasons"],
                         ["too_few_votes"])

    def test_score_only_changes_are_reported(self):
        report = self.compare()
        for kind in validator.RECORD_FILES:
            self.assertEqual(len(report[kind]["team_changes"]), 2)
            self.assertTrue(all(change["score_changed"] and not change["label_changed"]
                                for change in report[kind]["team_changes"]))

    def test_record_order_is_irrelevant(self):
        for kind in validator.RECORD_FILES:
            self.write_records(kind, list(reversed(self.records(kind))))
        self.assert_passed(self.compare())

    def test_new_fields_are_allowed(self):
        for kind in validator.RECORD_FILES:
            rows = self.records(kind)
            rows[0]["new_diagnostic"] = {"reason": "independent team evidence", "votes": [1, 2]}
            self.write_records(kind, rows)
        self.assert_passed(self.compare())

    def test_missing_baseline_directory_fails_clearly(self):
        report = compare_runs(self.root / "absent", self.candidate)
        self.assertFalse(report["passed"])
        self.assertIn("baseline.hash_before: run directory does not exist", "\n".join(report["errors"]))
        self.assertFalse(report["checks"]["baseline.required_files"])

    def test_every_required_file_is_mandatory_on_both_sides(self):
        for directory in (self.baseline, self.candidate):
            for name in REQUIRED_FILES:
                with self.subTest(side=directory.name, name=name):
                    path = directory / name
                    contents = path.read_bytes()
                    path.unlink()
                    try:
                        report = self.compare(allow_render_change=True)
                        self.assertFalse(report["passed"])
                        self.assertFalse(report["checks"][f"{directory.name}.required_files"])
                        self.assertIn(name, "\n".join(report["errors"]))
                    finally:
                        path.write_bytes(contents)

    def test_exact_files_reject_even_whitespace_changes(self):
        for name in EXACT_FILES:
            with self.subTest(name=name):
                path = self.candidate / name
                contents = path.read_bytes()
                path.write_bytes(contents + b" ")
                report = self.compare()
                self.assertFalse(report["passed"])
                self.assertFalse(report["exact_files"][name]["equal"])
                path.write_bytes(contents)

    def test_geometry_appearance_identity_and_accounting_changes_fail(self):
        mutations = {
            "detections": {"bbox_xyxy": [0, 0, 99, 99], "timestamp_s": 9.0, "frame": 99,
                           "detector_confidence": 0.1, "track_id": 99, "fragment_id": "F99",
                           "class_id": 2, "player_id": "P9", "jersey": 9,
                           "identification_confidence": 0.9, "confidence": 0.9,
                           "identification_reason": "changed", "reason": "changed"},
            "fragments": {"embedding": [0.1, 0.3], "numbers": [7], "track_id": 99,
                          "start_frame": 10, "end_frame": 10, "start_s": 1.0, "end_s": 1.0,
                          "appearance_evidence": {"detections": 99}, "player_id": "P9",
                          "jersey": 9, "identification_confidence": 0.9, "confidence": 0.9,
                          "identification_reason": "changed", "reason": "changed"},
        }
        for kind, fields in mutations.items():
            original = self.records(kind)
            for field, value in fields.items():
                with self.subTest(kind=kind, field=field):
                    rows = copy.deepcopy(original)
                    rows[0][field] = value
                    self.write_records(kind, rows)
                    report = self.compare()
                    self.assertFalse(report["passed"])
                    self.assertEqual(report[kind]["field_changes"][0]["field"], field)
                    self.assertEqual(bool(report[kind]["identity_changes"]),
                                     field in validator.IDENTITY_FIELDS)
            self.write_records(kind, original)
        path = self.candidate / "detection_accounting_summary.json"
        path.write_text('{"raw_detections": 999}', encoding="utf-8")
        self.assertFalse(self.compare()["checks"]["exact.detection_accounting_summary.json"])

    def test_missing_original_fields_fail_including_null_values(self):
        for kind in validator.RECORD_FILES:
            original = self.records(kind)
            for field in ("jersey", "player_id"):
                with self.subTest(kind=kind, field=field):
                    rows = copy.deepcopy(original)
                    del rows[0][field]
                    self.write_records(kind, rows)
                    report = self.compare()
                    self.assertFalse(report["passed"])
                    self.assertTrue(report[kind]["field_changes"][0]["missing"])
            self.write_records(kind, original)

    def test_original_nested_appearance_data_is_not_exempt(self):
        rows = self.records("fragments")
        rows[0]["appearance_evidence"]["reason"] = "changed"
        self.write_records("fragments", rows)
        self.assertFalse(self.compare()["checks"]["fragments.original_fields"])

    def test_counts_and_ids_are_not_silently_truncated(self):
        for kind, (_, id_key) in validator.RECORD_FILES.items():
            original = self.records(kind)
            for operation in ("remove", "add", "replace", "duplicate", "missing", "invalid"):
                with self.subTest(kind=kind, operation=operation):
                    rows = copy.deepcopy(original)
                    if operation == "remove":
                        rows.pop()
                    elif operation == "add":
                        rows.append({**rows[-1], id_key: "EXTRA"})
                    elif operation == "missing":
                        del rows[-1][id_key]
                    else:
                        rows[-1][id_key] = {"replace": "OTHER", "duplicate": rows[0][id_key],
                                           "invalid": []}[operation]
                    self.write_records(kind, rows)
                    report = self.compare()
                    self.assertFalse(report["passed"])
                    self.assertFalse(report["checks"][f"{kind}.matching_ids"])
                    if operation in {"remove", "add"}:
                        self.assertFalse(report["checks"][f"{kind}.counts"])
                    if operation in {"duplicate", "missing", "invalid"}:
                        self.assertTrue(report[kind]["id_errors"]["candidate"])
            self.write_records(kind, original)

    def test_duplicate_baseline_ids_are_rejected(self):
        for kind, (_, id_key) in validator.RECORD_FILES.items():
            rows = copy.deepcopy(self.rows if kind == "detections" else self.fragments)
            rows[1][id_key] = rows[0][id_key]
            self.write_records(kind, rows, self.baseline)
        report = self.compare()
        for kind in validator.RECORD_FILES:
            self.assertFalse(report["checks"][f"{kind}.baseline_ids"])

    def test_invalid_scores_and_unknown_nonzero_are_rejected(self):
        scores = [None, True, "0", -0.1, 1.1, float("nan"), float("inf"), float("-inf"), 0.01]
        for kind in validator.RECORD_FILES:
            original = self.records(kind)
            for score in scores:
                with self.subTest(kind=kind, score=score):
                    rows = copy.deepcopy(original)
                    if kind == "detections":
                        rows[0]["team_score"] = score
                    else:
                        rows[0]["team_classification"]["score"] = score
                    self.write_records(kind, rows)
                    report = self.compare()
                    self.assertFalse(report["passed"])
                    self.assertFalse(report["checks"][f"{kind}.final_scores"])
                    self.assertTrue(report[kind]["score_errors"])
                    json.dumps(report, allow_nan=False)
            self.write_records(kind, original)

    def test_missing_candidate_scores_are_rejected(self):
        self.write_records("detections", self.rows)
        self.write_records("fragments", self.fragments)
        report = self.compare()
        for kind in validator.RECORD_FILES:
            self.assertFalse(report["checks"][f"{kind}.final_scores"])

    def test_known_team_score_endpoints_are_allowed(self):
        for kind in validator.RECORD_FILES:
            rows = self.records(kind)
            for i, row in enumerate(rows):
                row["team_id"] = "TEAM_A"
                if kind == "detections":
                    row["team_score"] = float(i)
                else:
                    row["team_classification"].update(team_id="TEAM_A", score=float(i))
            self.write_records(kind, rows)
        self.assert_passed(self.compare())

    def test_fragment_final_label_must_match_score_record(self):
        rows = self.records("fragments")
        rows[0]["team_classification"].update(team_id="TEAM_A", score=0.8)
        self.write_records("fragments", rows)
        report = self.compare()
        self.assertFalse(report["passed"])
        self.assertIn("must match team_id", "\n".join(report["fragments"]["score_errors"]))

    def test_render_difference_requires_explicit_allowance(self):
        (self.candidate / "identified.mp4").write_bytes(b"intentional overlay change")
        report = self.compare()
        self.assertFalse(report["passed"])
        self.assertEqual(report["render"]["status"], "unexpected_difference")
        report = self.compare(allow_render_change=True)
        self.assert_passed(report)
        self.assertTrue(report["render"]["changed"])
        self.assertEqual(report["render"]["status"], "intentional_difference")
        (self.candidate / "boxes.jsonl").write_text("changed", encoding="utf-8")
        self.assertFalse(self.compare(allow_render_change=True)["passed"])

    def test_malformed_records_fail_with_a_report(self):
        for kind, (name, _) in validator.RECORD_FILES.items():
            path = self.candidate / name
            original = path.read_bytes()
            for contents in ("{bad json", "42", "null"):
                with self.subTest(kind=kind, contents=contents):
                    path.write_text(contents, encoding="utf-8")
                    report = self.compare()
                    self.assertFalse(report["passed"])
                    self.assertTrue(report["errors"])
            path.write_bytes(original)

    def test_all_original_files_remain_unchanged_and_hashes_are_auditable(self):
        (self.candidate / "extra").mkdir()
        (self.candidate / "extra" / "diagnostic.json").write_text("{}", encoding="utf-8")
        originals = {path: path.read_bytes() for directory in (self.baseline, self.candidate)
                     for path in directory.rglob("*") if path.is_file()}
        report = self.compare()
        self.assert_passed(report)
        self.assertEqual(originals, {path: path.read_bytes() for path in originals})
        for side, directory in (("baseline", self.baseline), ("candidate", self.candidate)):
            audit = report["read_only"][side]
            expected = {path.relative_to(directory).as_posix(): hashlib.sha256(data).hexdigest()
                        for path, data in originals.items() if path.is_relative_to(directory)}
            self.assertEqual(audit["before"], expected)
            self.assertEqual(audit["after"], expected)
            self.assertTrue(audit["unchanged"])

    def test_concurrent_input_mutation_fails_read_only_gate(self):
        load = validator._load_records

        def mutate_then_load(path, report, name):
            if name == "detections.candidate.load":
                (self.candidate / "players.json").write_text("changed concurrently", encoding="utf-8")
            return load(path, report, name)

        with patch.object(validator, "_load_records", side_effect=mutate_then_load):
            report = self.compare()
        self.assertFalse(report["passed"])
        self.assertFalse(report["read_only"]["candidate"]["unchanged"])

    def test_cli_success_failure_and_render_flag(self):
        result = self.cli()
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        self.assertTrue(json.loads((self.root / "report.json").read_text())["passed"])
        (self.candidate / "identified.mp4").write_bytes(b"new overlay")
        result = self.cli(out=self.root / "failed.json")
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertFalse(json.loads((self.root / "failed.json").read_text())["passed"])
        result = self.cli("--allow-render-change", out=self.root / "allowed.json")
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)

    def test_cli_refuses_existing_output_and_source_output_paths(self):
        for path in (self.root / "existing.json", self.candidate / "players.json",
                     self.baseline / "identified.mp4"):
            if not path.exists():
                path.write_bytes(b"existing report")
            original = path.read_bytes()
            result = self.cli(out=path)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("refusing to overwrite", result.stderr)
            self.assertEqual(path.read_bytes(), original)

    def test_cli_new_report_cannot_change_input_inventory(self):
        for directory in (self.baseline, self.candidate):
            path = directory / 'new_report.json'
            result = self.cli(out=path)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('outside both run directories', result.stderr)
            self.assertFalse(path.exists())

    def test_cli_missing_baseline_returns_failure_report(self):
        result = self.cli(baseline=self.root / "absent")
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("run directory does not exist", result.stdout)
        self.assertFalse(json.loads((self.root / "report.json").read_text())["passed"])

    def test_cli_nonfinite_score_writes_valid_failure_json(self):
        rows = self.records("detections")
        rows[0]["team_score"] = float("nan")
        self.write_records("detections", rows)
        result = self.cli()
        self.assertEqual(result.returncode, 1, result.stderr)
        report = json.loads((self.root / "report.json").read_text())
        self.assertFalse(report["passed"])
        json.dumps(report, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
