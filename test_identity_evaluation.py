"""Behavioral tests for the annotated-pair evaluator; synthetic labels only.

Nothing here is football ground truth: outcomes and denominators verify the
scoring policy, not identity accuracy.
"""

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from evaluate_identity_associations import evaluate_pairs, evaluate_run, main


VIDEO_SHA = hashlib.sha256(b"synthetic-video").hexdigest()


def fragment(fid, uid):
    return {"fragment_id": fid, "person_uid": uid}


def annotation(left, right, identity, reason="Synthetic policy label, not real accuracy."):
    return {"left": left, "right": right, "identity": identity, "reason": reason}


class EvaluatePairsTests(unittest.TestCase):
    def test_outcome_matrix_covers_every_label_and_prediction_state(self):
        uid_a, uid_b = "P" + "a" * 32, "P" + "b" * 32
        fragments = [fragment("A", uid_a), fragment("B", uid_a), fragment("C", uid_b),
                     fragment("D", None), fragment("E", None)]
        annotations = [
            annotation("A", "B", "different"),   # same non-null UID -> false_merge
            annotation("A", "C", "same"),        # distinct non-null UIDs -> false_split
            annotation("A", "D", "same"),        # one unresolved -> unresolved_same_pair
            annotation("D", "E", "same"),        # both unresolved -> unresolved_same_pair
            annotation("B", "C", "different"),   # distinct non-null UIDs -> kept_separate
            annotation("A", "E", "different"),   # one unresolved -> kept_separate
            annotation("C", "E", "uncertain"),   # not_scorable
            annotation("D", "C", "uncertain"),   # not_scorable
        ]
        report = evaluate_pairs(fragments, annotations)
        self.assertEqual(report["counts"], {
            "false_merge": 1, "false_split": 1, "unresolved_same_pair": 2,
            "confirmed_link": 0, "kept_separate": 2, "not_scorable": 2,
        })
        self.assertFalse(any(pair["outcome"] == "confirmed_link" for pair in report["pairs"]))
        outcomes = {(pair["left"], pair["right"]): pair["outcome"] for pair in report["pairs"]}
        self.assertEqual(outcomes[("A", "D")], "unresolved_same_pair")
        self.assertEqual(outcomes[("D", "E")], "unresolved_same_pair")
        self.assertEqual(outcomes[("A", "E")], "kept_separate")

    def test_confirmed_link_requires_two_nonnull_equal_uids(self):
        uid = "P" + "c" * 32
        fragments = [fragment("A", uid), fragment("B", uid), fragment("C", None),
                     fragment("D", None)]
        report = evaluate_pairs(fragments, [annotation("A", "B", "same"),
                                            annotation("C", "D", "same")])
        self.assertEqual(report["counts"]["confirmed_link"], 1)
        self.assertEqual(report["counts"]["unresolved_same_pair"], 1)
        self.assertEqual(report["pairs"][1]["prediction"], "both_unresolved")

    def test_denominators_are_explicit_and_never_substituted(self):
        fragments = [fragment("A", "P" + "a" * 32), fragment("B", "P" + "a" * 32),
                     fragment("C", None), fragment("D", None), fragment("E", "P" + "e" * 32)]
        annotations = [annotation("A", "B", "same"), annotation("C", "D", "same"),
                       annotation("A", "E", "different"), annotation("B", "E", "uncertain")]
        report = evaluate_pairs(fragments, annotations)
        denominators = report["denominators"]
        self.assertEqual(denominators["total_fragments"], 5)
        self.assertEqual(denominators["assigned_fragments"], 3)
        self.assertEqual(denominators["unresolved_fragments"], 2)
        self.assertEqual(denominators["annotated_fragments"], 5)
        self.assertEqual(denominators["unannotated_fragments"], 0)
        self.assertEqual(denominators["possible_unordered_pairs"], 10)
        self.assertEqual(denominators["annotated_pairs"], 4)
        self.assertEqual(denominators["unannotated_pairs"], 6)
        # Uncertain GT is excluded from the scorable denominator, and
        # scorable splits into both-assigned and with-unresolved parts.
        self.assertEqual(denominators["scorable_pairs"], 3)
        self.assertEqual(denominators["scorable_both_assigned_pairs"], 2)
        self.assertEqual(denominators["scorable_with_unresolved_pairs"], 1)
        self.assertEqual(denominators["by_identity"],
                         {"same": 2, "different": 1, "uncertain": 1})
        self.assertEqual(sum(denominators["by_prediction"].values()), 4)
        cross = denominators["by_identity_and_prediction"]
        self.assertEqual(cross["same"]["same_nonnull_uid"], 1)
        self.assertEqual(cross["same"]["both_unresolved"], 1)
        self.assertEqual(cross["different"]["distinct_nonnull_uids"], 1)
        json.dumps(report)

    def test_pair_and_reason_records_are_returned_verbatim(self):
        report = evaluate_pairs([fragment("A", None), fragment("B", None)],
                                [annotation("A", "B", "different", "kept apart on purpose")])
        pair = report["pairs"][0]
        self.assertEqual(pair["reason"], "kept apart on purpose")
        self.assertEqual(pair["prediction"], "both_unresolved")
        self.assertEqual(pair["unresolved_members"], ["A", "B"])
        self.assertEqual(pair["outcome"], "kept_separate")

    def test_invalid_inputs_raise(self):
        uid = "P" + "a" * 32
        good = [fragment("A", uid), fragment("B", uid)]
        for fragments, annotations, note in (
            ([fragment("A", uid)], [annotation("A", "missing", "same")], "unknown ID"),
            ([fragment("A", uid), fragment("B", uid)], [annotation("A", "A", "same")], "self-pair"),
            ([fragment("A", uid), fragment("B", uid)],
             [annotation("A", "B", "same"), annotation("B", "A", "same")], "duplicate pair"),
            ([{"fragment_id": "A"}], [], "missing explicit person_uid"),
            ([{"fragment_id": "A", "person_uid": ""}], [], "empty uid"),
            ([{"fragment_id": ""}], [], "empty fragment id"),
            ([fragment("A", uid), fragment("B", uid)],
             [annotation("A", "B", "maybe")], "invalid identity"),
            ([fragment("A", uid), fragment("B", uid)],
             [{"left": "A", "right": "B", "identity": "same", "reason": "  "}], "blank reason"),
            ("not-a-list", [], "fragments must be a list"),
            ([fragment("A", uid)], "not-a-list", "annotations must be a list"),
        ):
            with self.subTest(case=note):
                self.assertRaises(ValueError, evaluate_pairs, fragments, annotations)

    def test_evaluate_pairs_does_not_mutate_inputs(self):
        fragments = [fragment("A", "P" + "a" * 32), fragment("B", None)]
        annotations = [annotation("A", "B", "same")]
        before = json.dumps([fragments, annotations])
        evaluate_pairs(fragments, annotations)
        self.assertEqual(json.dumps([fragments, annotations]), before)


class EvaluateRunTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "run"
        self.root.mkdir()
        self.uid_a = "P" + "a" * 32
        self.uid_b = "P" + "b" * 32
        self.fragments = [fragment("A", self.uid_a), fragment("B", self.uid_a),
                          fragment("C", self.uid_b), fragment("D", None)]

    def write_run(self, manifest=None, boxes=b"D1\nD2\n", fragments=None, identity_report=None):
        (self.root / "boxes.jsonl").write_bytes(boxes)
        (self.root / "fragments.json").write_text(
            json.dumps(self.fragments if fragments is None else fragments), encoding="utf-8")
        if manifest is None:
            manifest = {"input_video_sha256": VIDEO_SHA}
        (self.root / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        if identity_report is not None:
            (self.root / "identity_association_report.json").write_text(
                json.dumps(identity_report), encoding="utf-8")

    def write_annotations(self, pairs, reviewer="Synthetic annotator", provenance="synthetic",
                          kind="human_identity_evaluation", video_sha=VIDEO_SHA, boxes_sha=None,
                          name="annotations.json"):
        path = Path(self.temp.name) / name
        body = {"kind": kind, "video_sha256": video_sha,
                "boxes_sha256": (boxes_sha if boxes_sha is not None
                                 else hashlib.sha256((self.root / "boxes.jsonl").read_bytes()).hexdigest()),
                "reviewer": reviewer, "provenance": provenance, "pairs": pairs}
        path.write_text(json.dumps(body), encoding="utf-8")
        return path

    def test_annotated_pairs_are_scored_and_bound_to_source_hashes(self):
        self.write_run()
        annotations = [annotation("A", "B", "same"), annotation("A", "C", "different"),
                       annotation("D", "A", "same")]
        path = self.write_annotations(annotations)
        report = evaluate_run(self.root, path)
        self.assertEqual(report["status"], "evaluated_annotated_pairs")
        self.assertEqual(report["counts"]["confirmed_link"], 1)
        self.assertEqual(report["counts"]["kept_separate"], 1)
        self.assertEqual(report["counts"]["unresolved_same_pair"], 1)
        self.assertEqual(report["source_hashes"]["input_video_sha256"], VIDEO_SHA)
        self.assertEqual(report["source_hashes"]["boxes_sha256"],
                         hashlib.sha256((self.root / "boxes.jsonl").read_bytes()).hexdigest())
        self.assertEqual(report["source_hashes"]["annotations_sha256"],
                         hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertEqual(report["annotation_provenance"]["independence"],
                         "declared_not_cryptographically_proven")

    def test_manifest_digest_may_come_from_the_schema_1_0_team_classification_block(self):
        self.write_run(manifest={"schema_version": "1.0",
                                 "team_classification": {"input_video_sha256": VIDEO_SHA}})
        path = self.write_annotations([annotation("A", "B", "same")])
        self.assertEqual(evaluate_run(self.root, path)["source_hashes"]["input_video_sha256"],
                         VIDEO_SHA)

    def test_conflicting_or_missing_manifest_digests_fail_closed(self):
        self.write_run(manifest={"input_video_sha256": VIDEO_SHA,
                                 "team_classification": {"input_video_sha256": "f" * 64}})
        path = self.write_annotations([annotation("A", "B", "same")])
        self.assertRaises(ValueError, evaluate_run, self.root, path)
        self.write_run(manifest={"schema_version": "1.0"})
        self.assertRaises(ValueError, evaluate_run, self.root, path)

    def test_annotation_hashes_must_match_the_run(self):
        self.write_run()
        stale = self.write_annotations([annotation("A", "B", "same")], video_sha="e" * 64)
        self.assertRaises(ValueError, evaluate_run, self.root, stale)
        bad_boxes = self.write_annotations([annotation("A", "B", "same")], boxes_sha="d" * 64)
        self.assertRaises(ValueError, evaluate_run, self.root, bad_boxes)

    def test_wrapper_metadata_is_required(self):
        self.write_run()
        for kwargs in ({"kind": "other"}, {"reviewer": ""}, {"provenance": " "}):
            with self.subTest(**kwargs):
                path = self.write_annotations([annotation("A", "B", "same")], **kwargs)
                self.assertRaises(ValueError, evaluate_run, self.root, path)
        path = self.write_annotations(None)
        self.assertRaises(ValueError, evaluate_run, self.root, path)

    def test_annotations_reusing_the_review_profile_are_rejected(self):
        self.write_run()
        annotations = self.write_annotations([annotation("A", "B", "same")],
                                             name="review_profile.json")
        review_sha = hashlib.sha256(annotations.read_bytes()).hexdigest()
        self.write_run(identity_report={"provenance": {"review": {"sha256": review_sha}}})
        self.assertRaises(ValueError, evaluate_run, self.root, annotations)

    def test_distinct_annotation_file_from_a_recorded_review_is_warned_free(self):
        self.write_run(identity_report={"provenance": {"review": {"sha256": "c" * 64}}})
        path = self.write_annotations([annotation("A", "B", "same")])
        report = evaluate_run(self.root, path)
        self.assertEqual(report["source_hashes"]["review_profile_sha256"], "c" * 64)
        self.assertEqual(report["source_hashes"]["identity_association_report_sha256"],
                         hashlib.sha256((self.root / "identity_association_report.json").read_bytes()).hexdigest())

    def test_missing_identity_report_warns_but_still_scores(self):
        self.write_run()
        path = self.write_annotations([annotation("A", "B", "same")])
        report = evaluate_run(self.root, path)
        self.assertIsNone(report["source_hashes"]["identity_association_report_sha256"])
        self.assertTrue(any("review profile digest" in w for w in report["warnings"]))

    def test_empty_annotation_file_produces_no_annotated_pairs_status(self):
        self.write_run()
        path = self.write_annotations([])
        report = evaluate_run(self.root, path)
        self.assertEqual(report["status"], "no_annotated_pairs")
        self.assertTrue(any("no pairs" in w for w in report["warnings"]))

    def test_legacy_fragments_without_person_uid_fail_closed(self):
        self.write_run(fragments=[{"fragment_id": "A"}, {"fragment_id": "B"}])
        path = self.write_annotations([annotation("A", "B", "same")])
        self.assertRaises(ValueError, evaluate_run, self.root, path)

    def test_evaluate_run_does_not_write_into_the_run_directory(self):
        self.write_run()
        path = self.write_annotations([annotation("A", "B", "same")])
        before = sorted(p.name for p in self.root.iterdir())
        evaluate_run(self.root, path)
        self.assertEqual(sorted(p.name for p in self.root.iterdir()), before)


class CliTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "run"
        self.root.mkdir()
        self.out = Path(self.temp.name) / "reports"
        self.out.mkdir()
        (self.root / "boxes.jsonl").write_bytes(b"D1\n")
        (self.root / "fragments.json").write_text(json.dumps(
            [fragment("A", "P" + "a" * 32), fragment("B", None)]), encoding="utf-8")
        (self.root / "run_manifest.json").write_text(
            json.dumps({"input_video_sha256": VIDEO_SHA}), encoding="utf-8")
        self.annotations = Path(self.temp.name) / "annotations.json"
        self.annotations.write_text(json.dumps({
            "kind": "human_identity_evaluation", "video_sha256": VIDEO_SHA,
            "boxes_sha256": hashlib.sha256(b"D1\n").hexdigest(),
            "reviewer": "Synthetic annotator", "provenance": "synthetic",
            "pairs": [annotation("A", "B", "same")]}), encoding="utf-8")

    def test_cli_writes_a_new_external_report(self):
        import contextlib
        import io
        target = self.out / "pair_evaluation.json"
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(["--run-dir", str(self.root),
                                   "--annotations", str(self.annotations),
                                   "--out", str(target)]), 0)
        report = json.loads(target.read_text(encoding="utf-8"))
        self.assertEqual(report["kind"], "identity_pair_evaluation")
        self.assertEqual(report["counts"]["unresolved_same_pair"], 1)

    def test_cli_refuses_existing_and_internal_output_paths(self):
        existing = self.out / "existing.json"
        existing.write_text("preserve", encoding="utf-8")
        for out in (existing, self.root / "inside.json", self.root):
            with self.subTest(out=out):
                result = subprocess.run(
                    [sys.executable, "-B", "evaluate_identity_associations.py",
                     "--run-dir", str(self.root), "--annotations", str(self.annotations),
                     "--out", str(out)],
                    capture_output=True, text=True, check=False, cwd=str(Path(__file__).parent))
                self.assertEqual(result.returncode, 2)
                self.assertIn("error:", result.stderr)
        self.assertEqual(existing.read_text(encoding="utf-8"), "preserve")
        self.assertFalse((self.root / "inside.json").exists())


if __name__ == "__main__":
    unittest.main()
