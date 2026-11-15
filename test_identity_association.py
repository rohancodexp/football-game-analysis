"""Synthetic policy fixtures ONLY: these labels are not real identity accuracy.

No video, detector, ResNet inference, learned Re-ID or external review is used.
"""

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from identity_association import EXTENSION_FIELDS, associate_fragments


VIDEO = "a" * 64
BOXES = "b" * 64


def fragment(fid, start, end, **updates):
    result = {
        "fragment_id": fid, "track_id": 7, "start_frame": start, "end_frame": end,
        "start_s": start / 30, "end_s": end / 30, "embedding": [1.0, 0.0, 0.0],
        "appearance_evidence": {"status": "ok", "embedding_generated": True,
                                "samples_used": 3, "min_observations": 3},
        "team_id": "TEAM_A", "player_id": "UNK", "jersey": None, "numbers": [],
        "identification_confidence": 0.1, "identification_reason": "existing_resolver_abstention",
        "bbox_xyxy": [1.0, 2.0, 10.0, 20.0], "custom": {"keep": [1, 2, 3]},
    }
    result.update(updates)
    return result


def pair(left, right, decision="same"):
    return {"left": left, "right": right, "decision": decision,
            "reason": "Synthetic identity judgment for policy testing, not real accuracy."}


def review_for(fragments, pairs=None):
    return {
        "kind": "human_fragment_identity_review", "video_sha256": VIDEO, "boxes_sha256": BOXES,
        "reviewer": "Synthetic policy reviewer", "reviewed_at": "2026-11-01T12:00:00Z",
        "fragments": [{"fragment_id": f["fragment_id"], "single_person": True,
                       "full_span_reviewed": True, "shot_id": "synthetic-shot-1"} for f in fragments],
        "pairs": [pair("A", "B")] if pairs is None else pairs,
    }


class SyntheticIdentityAssociationTests(unittest.TestCase):
    """Same/different-player and event names describe synthetic scenarios only."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.review_path = Path(self.temp.name) / "synthetic_review.json"
        self.fragments = [fragment("A", 0, 9), fragment("B", 20, 29)]

    def write_review(self, review):
        self.review_path.write_text(json.dumps(review, allow_nan=False), encoding="utf-8")
        return self.review_path

    def associate(self, fragments=None, review=None, **kwargs):
        fragments = self.fragments if fragments is None else fragments
        if review is None:
            review = review_for(fragments)
        report = associate_fragments(fragments, video_sha256=VIDEO, boxes_sha256=BOXES,
                                     review_path=self.write_review(review), **kwargs)
        # All public reports, including invalid numerical evidence, must be strict JSON.
        self.assertEqual(json.loads(json.dumps(report, allow_nan=False)), report)
        self.assertEqual(len(report["decisions"]), len(fragments))
        for decision in report["decisions"]:
            self.assertEqual(set(decision), {"fragment_id", *EXTENSION_FIELDS})
            self.assertTrue(decision["association_reason"])
        return report

    def assert_veto(self, report, reason):
        self.assertEqual(report["groups"], [])
        self.assertFalse(report["pairs"][0]["accepted"])
        self.assertIn(reason, report["pairs"][0]["reasons"])
        for decision in report["decisions"]:
            self.assertIsNone(decision["person_uid"])
            self.assertEqual(decision["association_status"], "unresolved")
            self.assertEqual(decision["association_method"], "abstain")

    def assert_invalid_review(self, review):
        original = copy.deepcopy(self.fragments)
        self.write_review(review)
        with self.assertRaises(ValueError):
            associate_fragments(self.fragments, video_sha256=VIDEO, boxes_sha256=BOXES,
                                review_path=self.review_path)
        self.assertEqual(self.fragments, original)

    def test_synthetic_same_player_reviewed_fragments_associate(self):
        self.fragments[1]["track_id"] = 99
        report = self.associate()
        self.assertEqual(len(report["groups"]), 1)
        group = report["groups"][0]
        self.assertEqual(group["fragment_ids"], ["A", "B"])
        self.assertRegex(group["person_uid"], r"^P[0-9a-f]{32}$")
        self.assertTrue(report["pairs"][0]["accepted"])
        self.assertTrue(report["pairs"][0]["grouped"])
        for f in self.fragments:
            self.assertEqual(f["schema_version"], "1.1")
            self.assertEqual(f["person_uid"], group["person_uid"])
            self.assertEqual(f["association_status"], "associated")
            self.assertEqual(f["association_method"], "human_reviewed_pair_clique")
        self.assertFalse(report["pairs"][0]["diagnostics"]["same_track_id"])

    def test_synthetic_same_uniform_different_players_never_link(self):
        report = self.associate(review=review_for(self.fragments, [pair("A", "B", "different")]))
        self.assert_veto(report, "review_decision_different")
        self.assertEqual(report["pairs"][0]["diagnostics"]["cosine_similarity"], 1.0)
        self.assertTrue(report["pairs"][0]["diagnostics"]["same_track_id"])

    def test_uncertain_review_abstains(self):
        report = self.associate(review=review_for(self.fragments, [pair("A", "B", "uncertain")]))
        self.assert_veto(report, "review_decision_uncertain")

    def test_overlap_including_touching_endpoint_vetoes(self):
        for start, end in ((5, 15), (9, 15), (0, 9), (2, 3)):
            with self.subTest(interval=(start, end)):
                fs = [fragment("A", 0, 9), fragment("B", start, end)]
                report = self.associate(fs)
                self.assert_veto(report, "inclusive_frame_overlap")
                self.assertTrue(report["pairs"][0]["diagnostics"]["intervals_overlap"])
                self.assertLessEqual(report["pairs"][0]["diagnostics"]["temporal_gap_frames"], 0)

    def test_adjacent_inclusive_intervals_are_disjoint(self):
        self.fragments[1].update(start_frame=10, end_frame=19)
        report = self.associate()
        self.assertTrue(report["pairs"][0]["accepted"])
        self.assertEqual(report["pairs"][0]["diagnostics"]["temporal_gap_frames"], 1)

    def test_synthetic_occlusion_reappearance_requires_direct_review(self):
        self.fragments[1].update(track_id=101, start_frame=300, end_frame=309)
        before = associate_fragments(self.fragments, video_sha256=VIDEO, boxes_sha256=BOXES)
        self.assertEqual(before["groups"], [])
        after = self.associate()
        self.assertEqual(after["groups"][0]["fragment_ids"], ["A", "B"])

    def test_synthetic_loss_without_reappearance_stays_unresolved(self):
        fs = [fragment("lost", 0, 9)]
        report = self.associate(fs, review_for(fs, []))
        self.assertEqual(report["groups"], [])
        self.assertEqual(report["pairs"], [])
        self.assertIsNone(fs[0]["person_uid"])
        self.assertEqual(fs[0]["association_reason"], "no_reviewed_pairs_for_fragment")

    def test_long_gap_not_a_numerical_acceptance_threshold(self):
        for start in (10, 1000, 10**12):
            with self.subTest(start=start):
                fs = [fragment("A", 0, 9), fragment("B", start, start + 9)]
                report = self.associate(fs)
                self.assertTrue(report["pairs"][0]["accepted"])
                self.assertEqual(report["pairs"][0]["diagnostics"]["temporal_gap_frames"], start - 9)
        review = review_for(self.fragments)
        review["fragments"][1]["shot_id"] = None
        self.assert_veto(self.associate(review=review), "unknown_shot_continuity")

    def test_cut_or_replay_in_different_reviewed_shot_vetoes(self):
        review = review_for(self.fragments)
        review["fragments"][1]["shot_id"] = "synthetic-cut-or-replay"
        self.assert_veto(self.associate(review=review), "different_reviewed_shots")

    def test_viewpoint_change_with_unknown_shot_vetoes(self):
        for shots in ((None, "shot"), ("shot", None), (None, None)):
            with self.subTest(shots=shots):
                review = review_for(self.fragments)
                for entry, shot in zip(review["fragments"], shots):
                    entry["shot_id"] = shot
                self.assert_veto(self.associate(review=review), "unknown_shot_continuity")

    def test_full_span_single_person_confirmation_required_on_both(self):
        for side, prefix in ((0, "left"), (1, "right")):
            for field, reason in (("single_person", "single_person_not_confirmed"),
                                  ("full_span_reviewed", "full_span_not_reviewed")):
                with self.subTest(side=side, field=field):
                    review = review_for(self.fragments)
                    review["fragments"][side][field] = False
                    self.assert_veto(self.associate(review=review), f"{prefix}:{reason}")

    def test_missing_zero_nonfinite_and_malformed_embeddings_veto(self):
        cases = [
            (None, "missing_embedding"), ([], "invalid_embedding_shape"),
            ([0, 0, 0], "zero_embedding"), ([float("nan"), 1, 0], "nonfinite_embedding"),
            ([float("inf"), 0, 0], "nonfinite_embedding"),
            ([float("-inf"), 0, 0], "nonfinite_embedding"),
            ([[1, 0, 0]], "invalid_embedding_shape"), (1.0, "invalid_embedding_shape"),
            (["1", "0", "0"], "invalid_embedding_shape"), ([True, 0, 0], "invalid_embedding_shape"),
            ({"x": 1}, "invalid_embedding_shape"), ("100", "invalid_embedding_shape"),
            ({0, 1}, "invalid_embedding_shape"),
            (np.ones((3, 1)), "invalid_embedding_shape"),
            (np.array(1.0), "invalid_embedding_shape"),
            (np.array([1 + 0j, 0, 0]), "invalid_embedding_shape"),
        ]
        for value, reason in cases:
            for side, prefix in ((0, "left"), (1, "right")):
                with self.subTest(value=repr(value), side=side):
                    fs = copy.deepcopy(self.fragments)
                    fs[side]["embedding"] = value
                    self.assert_veto(self.associate(fs), f"{prefix}:{reason}")
        del self.fragments[0]["embedding"]
        self.assert_veto(self.associate(), "left:missing_embedding")

    def test_incompatible_embedding_lengths_veto(self):
        self.fragments[1]["embedding"] = [1, 0]
        report = self.associate()
        self.assert_veto(report, "incompatible_embedding_shapes")
        self.assertIsNone(report["pairs"][0]["diagnostics"]["cosine_similarity"])

    def test_appearance_status_and_generated_flag_both_required(self):
        for side, prefix in ((0, "left"), (1, "right")):
            for evidence, reason in (
                (None, "appearance_evidence_not_ok"), ({}, "appearance_evidence_not_ok"),
                ({"status": "insufficient_observations", "embedding_generated": True}, "appearance_evidence_not_ok"),
                ({"status": "ok", "embedding_generated": False}, "embedding_not_generated"),
                ({"status": "ok", "embedding_generated": 1}, "embedding_not_generated"),
                ({"status": "ok"}, "embedding_not_generated"),
            ):
                with self.subTest(side=side, evidence=evidence):
                    fs = copy.deepcopy(self.fragments)
                    fs[side]["appearance_evidence"] = evidence
                    self.assert_veto(self.associate(fs), f"{prefix}:{reason}")

    def test_cosine_is_descriptive_even_for_opposite_vectors(self):
        for vector, expected in (([1, 0, 0], 1), ([0, 1, 0], 0), ([-1, 0, 0], -1)):
            with self.subTest(vector=vector):
                self.fragments[1]["embedding"] = vector
                report = self.associate()
                self.assertTrue(report["pairs"][0]["accepted"])
                self.assertEqual(report["pairs"][0]["diagnostics"]["cosine_similarity"], expected)

    def test_finite_nonzero_embeddings_have_no_epsilon_or_overflow_gate(self):
        for scale in (1e-320, 1e308):
            with self.subTest(scale=scale):
                self.fragments[0]["embedding"] = [scale, scale, 0]
                self.fragments[1]["embedding"] = [scale, scale, 0]
                report = self.associate()
                self.assertTrue(report["pairs"][0]["accepted"])
                self.assertAlmostEqual(report["pairs"][0]["diagnostics"]["cosine_similarity"], 1)

    def test_known_team_conflict_vetoes(self):
        self.fragments[1]["team_id"] = "TEAM_B"
        self.assert_veto(self.associate(), "conflicting_known_team_ids")

    def test_final_team_classification_conflict_not_hidden_by_unknown_field(self):
        self.fragments[1].update(team_id="UNKNOWN", team_classification={"team_id": "TEAM_B"})
        self.assert_veto(self.associate(), "conflicting_known_team_ids")

    def test_internal_known_team_conflict_vetoes(self):
        self.fragments[0]["team_classification"] = {"team_id": "TEAM_B"}
        self.assert_veto(self.associate(), "conflicting_known_team_ids")

    def test_resolved_roster_conflict_vetoes(self):
        self.fragments[0]["player_id"] = "roster-player-1"
        self.fragments[1]["player_id"] = "roster-player-2"
        self.assert_veto(self.associate(), "conflicting_resolved_player_ids")

    def test_unknown_team_and_roster_are_not_conflicting_or_filled_in(self):
        self.fragments[0].update(team_id="UNKNOWN", player_id="UNK")
        self.fragments[1].update(team_id="TEAM_A", player_id="roster-player-1")
        report = self.associate()
        self.assertTrue(report["pairs"][0]["accepted"])
        self.assertEqual(self.fragments[0]["team_id"], "UNKNOWN")
        self.assertEqual(self.fragments[0]["player_id"], "UNK")

    def test_mixed_fragment_status_and_hidden_proposal_votes_veto(self):
        cases = [
            {"team_classification": {"evidence_status": "mixed_fragment_appearance"}},
            {"team_classification": {"proposal": {"evidence_status": "mixed_fragment_appearance"}}},
            {"team_classification": {"team_id": "UNKNOWN", "evidence_status": "uncalibrated",
                                     "proposal": {"crop_votes": {"TEAM_A": 10, "TEAM_B": 1}}}},
            {"team_classification": {"crop_votes": {"TEAM_A": 2, "TEAM_B": 2}}},
            {"team_evidence_status": "mixed_fragment_appearance"},
            {"mixed_fragment_appearance": True},
        ]
        for update in cases:
            with self.subTest(update=update):
                fs = copy.deepcopy(self.fragments)
                fs[0].update(update)
                self.assert_veto(self.associate(fs), "left:mixed_fragment_appearance")

    def test_persistent_role_candidates_veto_even_when_team_already_unknown(self):
        cases = [
            {"team_classification": {"evidence_status": "persistent_role_candidate"}},
            {"team_classification": {"proposal": {"evidence_status": "persistent_role_candidate"}}},
            {"team_classification": {"team_id": "UNKNOWN", "evidence_status": "insufficient_fragment_votes",
                                     "candidate_evidence": {"persistent_evidence": True}}},
            {"persistent_role_candidate": True},
        ]
        for update in cases:
            with self.subTest(update=update):
                fs = copy.deepcopy(self.fragments)
                fs[1].update(update)
                self.assert_veto(self.associate(fs), "right:persistent_role_candidate")

    def test_no_review_never_links_identical_track_uniform_embedding_or_roster(self):
        for f in self.fragments:
            f["player_id"] = "same-roster-id"
        report = associate_fragments(self.fragments, video_sha256=VIDEO, boxes_sha256=BOXES)
        json.dumps(report, allow_nan=False)
        self.assertEqual(report["pairs"], [])
        self.assertEqual(report["groups"], [])
        self.assertEqual(report["counts"]["unreviewed_pairs"], 1)
        self.assertIsNone(report["provenance"]["review"])
        for f in self.fragments:
            self.assertIsNone(f["person_uid"])
            self.assertEqual(f["association_reason"], "no_human_review")
            self.assertEqual(f["association_method"], "abstain")

    def test_review_can_be_a_subset_and_unreviewed_fragments_abstain(self):
        review = review_for(self.fragments)
        self.fragments.append(fragment("C", 100, 109))
        report = self.associate(review=review)
        self.assertEqual(len(report["pairs"]), 1)
        self.assertEqual(report["counts"]["reviewed_fragments"], 2)
        self.assertEqual(report["counts"]["unreviewed_pairs"], 2)
        self.assertIsNone(self.fragments[2]["person_uid"])
        self.assertIn("unreviewed_pairs_abstain", report["assessment_scope"])

    def test_empty_review_pairs_do_not_associate(self):
        report = self.associate(review=review_for(self.fragments, []))
        self.assertEqual(report["groups"], [])
        self.assertEqual(report["pairs"], [])
        self.assertEqual(report["counts"]["fragments_unresolved"], 2)

    def test_all_reviewed_pairs_and_counts_are_reported(self):
        self.fragments.append(fragment("C", 100, 109))
        reviews = [pair("A", "B"), pair("A", "C", "different"), pair("B", "C", "uncertain")]
        report = self.associate(review=review_for(self.fragments, reviews))
        self.assertEqual([(p["left"], p["right"], p["decision"], p["accepted"]) for p in report["pairs"]],
                         [("A", "B", "same", True), ("A", "C", "different", False),
                          ("B", "C", "uncertain", False)])
        self.assertEqual(report["counts"], {
            "fragments_total": 3, "fragments_associated": 2, "fragments_unresolved": 1, "groups": 1,
            "reviewed_fragments": 3, "reviewed_pairs": 3, "accepted_pairs": 1,
            "rejected_pairs": 2, "blocked_accepted_pairs": 0, "unreviewed_pairs": 0,
        })
        for p in report["pairs"]:
            self.assertTrue(p["reasons"])
            self.assertTrue(p["review_reason"])
            self.assertTrue(p["diagnostics"]["descriptive_only"])

    def test_ab_bc_without_ac_never_implies_transitive_identity(self):
        self.fragments.append(fragment("C", 100, 109))
        review = review_for(self.fragments, [pair("B", "C"), pair("A", "B")])
        report = self.associate(review=review)
        self.assertEqual([g["fragment_ids"] for g in report["groups"]], [["A", "B"]])
        self.assertEqual(report["counts"]["accepted_pairs"], 2)
        self.assertEqual(report["counts"]["blocked_accepted_pairs"], 1)
        self.assertFalse(report["pairs"][1]["grouped"])
        blocked = report["blocked_links"][0]
        self.assertEqual((blocked["left"], blocked["right"]), ("B", "C"))
        self.assertEqual(blocked["reason"], "clique_condition_not_met")
        self.assertEqual(blocked["blocking_pairs"], [{"left": "A", "right": "C", "decision": None,
                                                     "reasons": ["unreviewed_pair"]}])
        self.assertIsNone(self.fragments[2]["person_uid"])
        self.assertIn("accepted_pair_blocked_by_clique_condition", self.fragments[2]["association_reason"])

    def test_explicit_different_or_uncertain_ac_blocks_clique(self):
        self.fragments.append(fragment("C", 100, 109))
        for decision in ("different", "uncertain"):
            with self.subTest(decision=decision):
                review = review_for(self.fragments, [pair("A", "B"), pair("B", "C"), pair("A", "C", decision)])
                report = self.associate(review=review)
                self.assertEqual([g["fragment_ids"] for g in report["groups"]], [["A", "B"]])
                self.assertEqual(report["blocked_links"][0]["blocking_pairs"][0]["decision"], decision)

    def test_complete_three_member_clique_forms_one_group(self):
        self.fragments.append(fragment("C", 100, 109))
        review = review_for(self.fragments, [pair("C", "B"), pair("C", "A"), pair("B", "A")])
        report = self.associate(review=review)
        self.assertEqual([g["fragment_ids"] for g in report["groups"]], [["A", "B", "C"]])
        self.assertEqual(report["blocked_links"], [])
        self.assertTrue(all(p["grouped"] for p in report["pairs"]))

    def test_two_cliques_merge_only_when_every_cross_pair_is_accepted(self):
        self.fragments.extend([fragment("C", 40, 49), fragment("D", 60, 69)])
        reviews = [pair("A", "B"), pair("A", "C"), pair("B", "D"), pair("C", "D")]
        report = self.associate(review=review_for(self.fragments, reviews))
        self.assertEqual([g["fragment_ids"] for g in report["groups"]], [["A", "B"], ["C", "D"]])
        members = [fid for g in report["groups"] for fid in g["fragment_ids"]]
        self.assertEqual(len(members), len(set(members)))
        self.assertEqual(len(report["blocked_links"]), 2)
        reviews.extend([pair("A", "D"), pair("B", "C")])
        report = self.associate(review=review_for(self.fragments, reviews))
        self.assertEqual([g["fragment_ids"] for g in report["groups"]], [["A", "B", "C", "D"]])

    def test_endpoint_displacement_is_temporally_oriented_descriptive_only(self):
        self.fragments = [fragment("A", 20, 29), fragment("B", 0, 9)]
        endpoints = {
            "B": {"start_bbox": [0, 0, 2, 2], "end_bbox": [0, 0, 2, 2]},
            "A": {"start_bbox": [3, 4, 5, 6], "end_bbox": [100, 100, 102, 102]},
        }
        original = copy.deepcopy(endpoints)
        report = self.associate(endpoints=endpoints)
        diag = report["pairs"][0]["diagnostics"]
        self.assertEqual(diag["endpoint_displacement_pixels"], 5.0)
        self.assertEqual(diag["endpoint_direction"], {"from": "B", "to": "A"})
        self.assertEqual(endpoints, original)
        endpoints["A"]["start_bbox"] = [1e12, 1e12, 1e12 + 2, 1e12 + 2]
        report = self.associate(endpoints=endpoints)
        self.assertTrue(report["pairs"][0]["accepted"])
        self.assertGreater(report["pairs"][0]["diagnostics"]["endpoint_displacement_pixels"], 1e12)

    def test_missing_invalid_nonfinite_endpoints_are_safe_and_do_not_veto(self):
        cases = [None, {}, {"A": {}},
                 {"A": {"end_bbox": [0, 0, 1, 1]}, "B": {"start_bbox": [float("nan"), 0, 2, 2]}},
                 {"A": {"end_bbox": [1, 0, 0, 1]}, "B": {"start_bbox": [0, 0, 2, 2]}},
                 {"A": {"end_bbox": [0, 0, 1]}, "B": {"start_bbox": [0, 0, 2, 2]}},
                 {"A": {"end_bbox": [-1.7e308, 0, -1.6e308, 1]},
                  "B": {"start_bbox": [1.6e308, 0, 1.7e308, 1]}}]
        for endpoints in cases:
            with self.subTest(endpoints=endpoints):
                report = self.associate(endpoints=endpoints)
                self.assertTrue(report["pairs"][0]["accepted"])
                self.assertIsNone(report["pairs"][0]["diagnostics"]["endpoint_displacement_pixels"])

    def test_invalid_frame_intervals_veto_without_coercion(self):
        for start, end in ((None, 9), (0, None), (-1, 9), (10, 9), (True, 9), (0.0, 9), ("0", 9)):
            with self.subTest(interval=(start, end)):
                fs = copy.deepcopy(self.fragments)
                fs[0].update(start_frame=start, end_frame=end)
                self.assert_veto(self.associate(fs), "left:invalid_frame_interval")

    def test_numpy_fragments_and_report_are_json_safe_without_input_conversion(self):
        for f in self.fragments:
            f["embedding"] = np.array([1, 0, 0], dtype=np.float32)
            f["start_frame"] = np.int64(f["start_frame"])
            f["end_frame"] = np.int64(f["end_frame"])
            f["track_id"] = np.int64(f["track_id"])
        report = self.associate()
        self.assertTrue(report["pairs"][0]["accepted"])
        self.assertIsInstance(self.fragments[0]["embedding"], np.ndarray)
        self.assertIsInstance(self.fragments[0]["start_frame"], np.int64)

    def test_only_five_extension_fields_are_mutated(self):
        self.fragments[0]["embedding"] = np.array([1, 0, 0], dtype=np.float32)
        objects = [{key: value for key, value in f.items()} for f in self.fragments]
        originals = copy.deepcopy(self.fragments)
        order = [id(f) for f in self.fragments]
        report = self.associate()
        self.assertEqual([id(f) for f in self.fragments], order)
        for f, original, old_objects, decision in zip(self.fragments, originals, objects, report["decisions"]):
            self.assertEqual(set(f) - set(original), set(EXTENSION_FIELDS))
            for key in original:
                self.assertIs(f[key], old_objects[key])
                if isinstance(original[key], np.ndarray):
                    np.testing.assert_array_equal(f[key], original[key])
                else:
                    self.assertEqual(f[key], original[key])
            self.assertEqual({k: f[k] for k in EXTENSION_FIELDS},
                             {k: decision[k] for k in EXTENSION_FIELDS})

    def test_report_does_not_expose_mutable_input_objects(self):
        report = self.associate()
        report["pairs"][0]["diagnostics"]["left_interval"][0] = 12345
        report["decisions"][0]["association_reason"] = "changed only the report"
        self.assertEqual(self.fragments[0]["start_frame"], 0)
        self.assertNotEqual(self.fragments[0]["association_reason"], "changed only the report")

    def test_uid_hashes_exact_inputs_and_sorted_member_ids(self):
        report = self.associate()
        payload = json.dumps([VIDEO, BOXES, ["A", "B"]], ensure_ascii=True, separators=(",", ":")).encode("utf-8")
        self.assertEqual(report["groups"][0]["person_uid"], "P" + hashlib.sha256(payload).hexdigest()[:32])
        original_uid = report["groups"][0]["person_uid"]
        for key in ("video_sha256", "boxes_sha256"):
            review = review_for(self.fragments)
            review[key] = "c" * 64
            kwargs = {"video_sha256": VIDEO, "boxes_sha256": BOXES, key: "c" * 64}
            changed = associate_fragments(self.fragments, review_path=self.write_review(review), **kwargs)
            self.assertNotEqual(original_uid, changed["groups"][0]["person_uid"])
        fs = [fragment("A", 0, 9), fragment("C", 20, 29)]
        changed = self.associate(fs, review_for(fs, [pair("A", "C")]))
        self.assertNotEqual(original_uid, changed["groups"][0]["person_uid"])

    def test_deterministic_input_reordering_and_repeated_invocation(self):
        self.fragments.append(fragment("C", 100, 109))
        review = review_for(self.fragments, [pair("B", "C"), pair("A", "B")])
        first = self.associate(review=review)
        self.fragments.reverse()
        second = self.associate(review=review)
        self.assertEqual(first, second)
        self.assertEqual(second, self.associate(review=review))

    def test_review_reordering_preserves_decisions_but_retains_exact_source_hash(self):
        self.fragments.append(fragment("C", 100, 109))
        review = review_for(self.fragments, [pair("B", "C"), pair("A", "B")])
        first = self.associate(review=review)
        review["fragments"].reverse()
        review["pairs"].reverse()
        for p in review["pairs"]:
            p["left"], p["right"] = p["right"], p["left"]
        second = self.associate(review=review)
        self.assertNotEqual(first["provenance"]["review"]["sha256"], second["provenance"]["review"]["sha256"])
        first["provenance"]["review"].pop("sha256")
        second["provenance"]["review"].pop("sha256")
        self.assertEqual(first, second)

    def test_review_sha256_source_path_and_reviewer_provenance(self):
        report = self.associate()
        provenance = report["provenance"]["review"]
        self.assertEqual(provenance["sha256"], hashlib.sha256(self.review_path.read_bytes()).hexdigest())
        self.assertEqual(provenance["source_path"], str(self.review_path.resolve()))
        self.assertEqual(provenance["reviewer"], "Synthetic policy reviewer")
        self.assertEqual(provenance["reviewed_at"], "2026-11-01T12:00:00Z")
        self.assertEqual(provenance["video_sha256"], VIDEO)
        self.assertEqual(provenance["boxes_sha256"], BOXES)
        self.assertIn("not real identity accuracy", " ".join(report["limitations"]))

    def test_iso_dates_and_timestamps_are_valid(self):
        for stamp in ("2026-11-01", "2026-11-01T12:00:00Z", "2026-11-01T12:00:00+00:00",
                      "2026-11-01T12:00:00.123456-05:00", "2026-11-01 12:00"):
            with self.subTest(stamp=stamp):
                review = review_for(self.fragments)
                review["reviewed_at"] = stamp
                self.assertTrue(self.associate(review=review)["pairs"][0]["accepted"])

    def test_malformed_review_root_metadata_rejected_atomically(self):
        for value in (None, [], "review"):
            with self.subTest(root=value):
                self.assert_invalid_review(value)
        invalid = {
            "kind": (None, "other"), "reviewer": (None, "", "  ", 1),
            "reviewed_at": (None, "", "yesterday", "2026-02-30", "2026-11-01T25:00:00Z", "20261101"),
            "fragments": ({}, None), "pairs": ({}, None),
        }
        for key, values in invalid.items():
            for value in values:
                with self.subTest(key=key, value=value):
                    review = review_for(self.fragments)
                    review[key] = value
                    self.assert_invalid_review(review)
        for key in review_for(self.fragments):
            with self.subTest(missing=key):
                review = review_for(self.fragments)
                del review[key]
                self.assert_invalid_review(review)

    def test_malformed_fragment_review_rejected_atomically(self):
        for key, values in {
            "fragment_id": (None, "", "missing", []), "single_person": (1, "true", None),
            "full_span_reviewed": (0, "false", None), "shot_id": ("", "  ", 1, False),
        }.items():
            for value in values:
                with self.subTest(key=key, value=value):
                    review = review_for(self.fragments)
                    review["fragments"][0][key] = value
                    self.assert_invalid_review(review)
        for key in review_for(self.fragments)["fragments"][0]:
            review = review_for(self.fragments)
            del review["fragments"][0][key]
            self.assert_invalid_review(review)

    def test_malformed_pair_review_rejected_atomically(self):
        for key, values in {
            "left": (None, "missing", []), "right": (None, "missing", 1),
            "decision": (None, True, "Same", "yes", []), "reason": (None, "", "  ", 1),
        }.items():
            for value in values:
                with self.subTest(key=key, value=value):
                    review = review_for(self.fragments)
                    review["pairs"][0][key] = value
                    self.assert_invalid_review(review)
        for key in pair("A", "B"):
            review = review_for(self.fragments)
            del review["pairs"][0][key]
            self.assert_invalid_review(review)

    def test_duplicate_fragments_and_reversed_pairs_rejected(self):
        review = review_for(self.fragments)
        review["fragments"].append(copy.deepcopy(review["fragments"][0]))
        self.assert_invalid_review(review)
        for extra in (pair("A", "B"), pair("B", "A"), pair("B", "A", "different")):
            review = review_for(self.fragments)
            review["pairs"].append(extra)
            self.assert_invalid_review(review)

    def test_self_pairs_unknown_ids_and_missing_pair_fragment_entries_rejected(self):
        for p in (pair("A", "A"), pair("A", "missing")):
            self.assert_invalid_review(review_for(self.fragments, [p]))
        review = review_for(self.fragments)
        review["fragments"].pop()
        self.assert_invalid_review(review)

    def test_stale_and_malformed_hashes_rejected_with_exact_case_comparison(self):
        for key in ("video_sha256", "boxes_sha256"):
            for value in ("c" * 64, None, "x" * 64, "a" * 63, "a" * 65,
                          (VIDEO if key == "video_sha256" else BOXES).upper()):
                with self.subTest(key=key, value=value):
                    review = review_for(self.fragments)
                    review[key] = value
                    self.assert_invalid_review(review)

    def test_invalid_json_duplicate_keys_and_nonfinite_json_rejected(self):
        valid = json.dumps(review_for(self.fragments))
        raws = [b"{", b"\xff", b"NaN", b"Infinity", b"-Infinity",
                valid.replace('"reviewer":', '"reviewer": "duplicate", "reviewer":', 1).encode(),
                valid.replace('"single_person": true', '"single_person": false, "single_person": true', 1).encode()]
        for raw in raws:
            with self.subTest(raw=raw[:50]):
                before = copy.deepcopy(self.fragments)
                self.review_path.write_bytes(raw)
                with self.assertRaises(ValueError):
                    associate_fragments(self.fragments, video_sha256=VIDEO, boxes_sha256=BOXES,
                                        review_path=self.review_path)
                self.assertEqual(self.fragments, before)

    def test_missing_review_file_fails_without_mutation(self):
        original = copy.deepcopy(self.fragments)
        with self.assertRaises(FileNotFoundError):
            associate_fragments(self.fragments, video_sha256=VIDEO, boxes_sha256=BOXES,
                                review_path=self.review_path)
        self.assertEqual(self.fragments, original)

    def test_unknown_review_fields_rejected_instead_of_silently_ignored(self):
        for location in ("root", "fragment", "pair"):
            review = review_for(self.fragments)
            target = review if location == "root" else review["fragments" if location == "fragment" else "pairs"][0]
            target["unexpected"] = True
            self.assert_invalid_review(review)

    def test_invalid_input_fragment_ids_and_hashes_rejected_without_mutation(self):
        for fs in ([fragment("A", 0, 9), fragment("A", 20, 29)], [{}], [None],
                   [fragment("", 0, 9)], [fragment(123, 0, 9)]):
            with self.subTest(fragments=fs):
                before = copy.deepcopy(fs)
                with self.assertRaises(ValueError):
                    associate_fragments(fs, video_sha256=VIDEO, boxes_sha256=BOXES)
                self.assertEqual(fs, before)
        for kwargs in ({"video_sha256": "bad", "boxes_sha256": BOXES},
                       {"video_sha256": VIDEO, "boxes_sha256": None}):
            with self.assertRaises(ValueError):
                associate_fragments(self.fragments, **kwargs)
        with self.assertRaises(ValueError):
            associate_fragments(self.fragments, video_sha256=VIDEO, boxes_sha256=BOXES, endpoints=[])
        self.assertNotIn("person_uid", self.fragments[0])

    def test_empty_input_has_empty_json_safe_report(self):
        report = associate_fragments([], video_sha256=VIDEO, boxes_sha256=BOXES)
        json.dumps(report, allow_nan=False)
        for key in ("pairs", "groups", "decisions", "blocked_links"):
            self.assertEqual(report[key], [])
        self.assertTrue(all(value == 0 for value in report["counts"].values()))

    def test_no_review_does_not_build_quadratic_pair_matrix(self):
        fs = [fragment(f"F{i:06d}", i * 10, i * 10 + 9) for i in range(1000)]
        report = associate_fragments(fs, video_sha256=VIDEO, boxes_sha256=BOXES)
        self.assertEqual(report["pairs"], [])
        self.assertEqual(report["counts"]["unreviewed_pairs"], 1000 * 999 // 2)
        self.assertEqual(report["counts"]["fragments_unresolved"], 1000)


if __name__ == "__main__":
    unittest.main()
