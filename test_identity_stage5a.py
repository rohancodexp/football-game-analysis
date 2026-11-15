"""Stage 5A completion tests (synthetic policy fixtures only, not identity accuracy).

Covers the three gaps recorded in STAGE_1_TO_4B_AUDIT.md: digest case
(G2), the public deterministic UID helper (G3) and the reviewer-flagged
tracker-switch veto (G1).
"""

import json
import tempfile
import unittest
from pathlib import Path

import identity_association as ia
from identity_association import associate_fragments, person_uid
from test_identity_association import BOXES, VIDEO, fragment, pair, review_for


class Stage5ATests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.review_path = Path(self.temp.name) / "synthetic_review.json"
        self.fragments = [fragment("A", 0, 9), fragment("B", 20, 29)]

    def write(self, review):
        self.review_path.write_text(json.dumps(review, allow_nan=False), encoding="utf-8")
        return self.review_path

    def associate(self, review, video=VIDEO, boxes=BOXES):
        return associate_fragments(self.fragments, video_sha256=video, boxes_sha256=boxes,
                                   review_path=self.write(review))

    # G2 -----------------------------------------------------------------
    def test_upper_case_input_digests_fail_closed_before_any_mutation(self):
        for video, boxes in ((VIDEO.upper(), BOXES), (VIDEO, BOXES.upper())):
            with self.subTest(video=video[:4], boxes=boxes[:4]):
                with self.assertRaisesRegex(ValueError, "lower-case"):
                    self.associate(review_for(self.fragments), video=video, boxes=boxes)
                for f in self.fragments:
                    self.assertNotIn("person_uid", f)

    def test_lower_case_digests_still_accepted(self):
        report = self.associate(review_for(self.fragments))
        self.assertEqual(report["counts"]["groups"], 1)

    # G3 -----------------------------------------------------------------
    def test_public_uid_matches_legacy_name_and_report(self):
        report = self.associate(review_for(self.fragments))
        expected = person_uid(VIDEO, BOXES, ["A", "B"])
        self.assertIs(ia._person_uid, ia.person_uid)
        self.assertEqual(report["groups"][0]["person_uid"], expected)
        self.assertRegex(expected, r"^P[0-9a-f]{32}$")

    def test_uid_is_order_independent_and_input_bound(self):
        self.assertEqual(person_uid(VIDEO, BOXES, ["B", "A"]), person_uid(VIDEO, BOXES, ["A", "B"]))
        self.assertNotEqual(person_uid(VIDEO, BOXES, ["A", "B"]), person_uid("c" * 64, BOXES, ["A", "B"]))
        self.assertNotEqual(person_uid(VIDEO, BOXES, ["A", "B"]), person_uid(VIDEO, "c" * 64, ["A", "B"]))
        self.assertNotEqual(person_uid(VIDEO, BOXES, ["A", "B"]), person_uid(VIDEO, BOXES, ["A", "C"]))

    def test_validator_uses_public_helper(self):
        import validate_tracking_outputs as vto
        self.assertIs(vto.person_uid, ia.person_uid)

    # G1 -----------------------------------------------------------------
    def test_reviewer_flagged_tracker_switch_vetoes_on_either_side(self):
        for side, prefix in ((0, "left"), (1, "right")):
            with self.subTest(side=side):
                review = review_for(self.fragments)
                review["fragments"][side]["tracker_switch_suspected"] = True
                report = self.associate(review)
                self.assertFalse(report["pairs"][0]["accepted"])
                self.assertIn(f"{prefix}:tracker_switch_suspected", report["pairs"][0]["reasons"])
                self.assertEqual(report["counts"]["groups"], 0)
                self.assertTrue(all(d["person_uid"] is None for d in report["decisions"]))

    def test_explicit_false_and_absent_flag_keep_existing_behaviour(self):
        review = review_for(self.fragments)
        review["fragments"][0]["tracker_switch_suspected"] = False
        self.assertTrue(self.associate(review)["pairs"][0]["accepted"])
        self.assertTrue(self.associate(review_for(self.fragments))["pairs"][0]["accepted"])

    def test_non_boolean_flag_and_unknown_keys_rejected(self):
        for value in ("yes", 1, None, []):
            with self.subTest(value=value):
                review = review_for(self.fragments)
                review["fragments"][0]["tracker_switch_suspected"] = value
                with self.assertRaises(ValueError):
                    self.associate(review)
        review = review_for(self.fragments)
        review["fragments"][0]["unexpected"] = True
        with self.assertRaises(ValueError):
            self.associate(review)

    def test_flag_blocks_transitive_group_through_flagged_fragment(self):
        self.fragments = [fragment("A", 0, 9), fragment("B", 20, 29), fragment("C", 40, 49)]
        review = review_for(self.fragments, [pair("A", "B"), pair("B", "C"), pair("A", "C")])
        review["fragments"][1]["tracker_switch_suspected"] = True
        report = self.associate(review)
        accepted = {(p["left"], p["right"]) for p in report["pairs"] if p["accepted"]}
        self.assertEqual(accepted, {("A", "C")})
        self.assertEqual([g["fragment_ids"] for g in report["groups"]], [["A", "C"]])
        by_id = {d["fragment_id"]: d for d in report["decisions"]}
        self.assertIsNone(by_id["B"]["person_uid"])


if __name__ == "__main__":
    unittest.main()
