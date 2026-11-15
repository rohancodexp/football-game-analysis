import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import assign_persistent_ids as api


def vec(*head):
    v = [0.0] * 8
    for i, x in enumerate(head):
        v[i] = x
    return v


def frag(fid, start, end, emb=None, proposal=None, team="UNKNOWN"):
    return {"fragment_id": fid, "start_frame": start, "end_frame": end, "embedding": emb,
            "player_id": "UNK", "team_id": team, "person_uid": None,
            "team_classification": {"team_id_before_reliability": proposal}}


def ends_for(*specs):
    """specs: (fid, first_box, last_box, detections)"""
    return {fid: {"first": (0, first), "last": (0, last), "detections": n} for fid, first, last, n in specs}


BOX = [100, 100, 140, 200]          # height 100
NEAR = [110, 105, 150, 205]         # centre moves ~11 px
FAR = [900, 100, 940, 200]


class AssignTests(unittest.TestCase):
    def test_continuity_links_reacquired_track(self):
        frags = [frag("F1", 0, 10), frag("F2", 20, 30)]
        a, groups, links, *_ = api.assign(frags, ends_for(("F1", BOX, BOX, 11), ("F2", NEAR, NEAR, 11)))
        self.assertEqual(a["F1"]["persistent_id"], a["F2"]["persistent_id"])
        self.assertEqual(links[0]["kind"], "continuity")
        self.assertEqual(len(groups), 1)

    def test_far_or_late_reappearance_without_appearance_stays_split(self):
        frags = [frag("F1", 0, 10), frag("F2", 20, 30), frag("F3", 200, 210)]
        a, *_ = api.assign(frags, ends_for(("F1", BOX, BOX, 1), ("F2", FAR, FAR, 1), ("F3", BOX, BOX, 1)))
        self.assertEqual(len({a[f]["persistent_id"] for f in a}), 3)

    def test_appearance_links_long_gap(self):
        frags = [frag("F1", 0, 10, vec(1, 0.1)), frag("F2", 300, 310, vec(1, 0.12))]
        a, _, links, *_ = api.assign(frags, ends_for(("F1", BOX, BOX, 1), ("F2", FAR, FAR, 1)))
        self.assertEqual(a["F1"]["persistent_id"], a["F2"]["persistent_id"])
        self.assertEqual(links[0]["kind"], "appearance")

    def test_low_cosine_does_not_link(self):
        frags = [frag("F1", 0, 10, vec(1, 0)), frag("F2", 300, 310, vec(0, 1))]
        a, *_ = api.assign(frags, ends_for(("F1", BOX, BOX, 1), ("F2", FAR, FAR, 1)))
        self.assertNotEqual(a["F1"]["persistent_id"], a["F2"]["persistent_id"])

    def test_simultaneous_fragments_never_share_an_id(self):
        same = vec(1, 0)
        frags = [frag("F1", 0, 50, same), frag("F2", 40, 90, same)]
        a, *_ = api.assign(frags, ends_for(("F1", BOX, BOX, 1), ("F2", NEAR, NEAR, 1)))
        self.assertNotEqual(a["F1"]["persistent_id"], a["F2"]["persistent_id"])

    def test_group_level_overlap_blocks_transitive_merge(self):
        # F1 and F3 overlap; F2 links to both by appearance; F2 may join only one of them.
        same = vec(1, 0)
        frags = [frag("F1", 0, 10, same), frag("F3", 5, 15, same), frag("F2", 100, 110, same)]
        a, _, _, refused, *_ = api.assign(frags, ends_for(("F1", BOX, BOX, 1), ("F3", FAR, FAR, 1), ("F2", BOX, BOX, 1)))
        self.assertNotEqual(a["F1"]["persistent_id"], a["F3"]["persistent_id"])
        self.assertEqual(refused.get("temporal_overlap"), 1)

    def test_team_conflict_blocks_link(self):
        same = vec(1, 0)
        frags = [frag("F1", 0, 10, same, "TEAM_A"), frag("F2", 20, 30, same, "TEAM_B")]
        a, *_ = api.assign(frags, ends_for(("F1", BOX, BOX, 1), ("F2", NEAR, NEAR, 1)))
        self.assertNotEqual(a["F1"]["persistent_id"], a["F2"]["persistent_id"])
        self.assertEqual(a["F1"]["persistent_team_id"], "TEAM_A")
        self.assertEqual(a["F2"]["persistent_team_id"], "TEAM_B")

    def test_weak_chain_is_refused(self):
        # F1~F2 and F2~F3 pass, F1~F3 fails: all three may not form one appearance group.
        e1, e2, e3 = vec(1, 0), vec(0.96, 0.28), vec(0.85, 0.53)
        frags = [frag("F1", 0, 10, e1), frag("F2", 100, 110, e2), frag("F3", 200, 210, e3)]
        a, *_ = api.assign(frags, ends_for(("F1", BOX, BOX, 1), ("F2", FAR, FAR, 1), ("F3", BOX, [500, 100, 540, 200], 1)),
                           {"min_cosine": 0.9})
        a2, _, _, refused, *_ = api.assign(frags, ends_for(("F1", BOX, BOX, 1), ("F2", FAR, FAR, 1), ("F3", BOX, [500, 100, 540, 200], 1)),
                                          {"min_cosine": 0.9})
        self.assertEqual(refused.get("weak_appearance_pair"), 1)
        self.assertFalse(a["F1"]["persistent_id"] == a["F2"]["persistent_id"] == a["F3"]["persistent_id"])

    def test_team_majority_and_status(self):
        frags = [frag("F1", 0, 10, None, "TEAM_A"), frag("F2", 20, 30)]
        a, groups, *_ = api.assign(frags, ends_for(("F1", BOX, BOX, 5), ("F2", NEAR, NEAR, 5)))
        self.assertEqual(a["F2"]["persistent_team_id"], "TEAM_A")
        self.assertEqual(groups[0]["team_status"], "uncalibrated_proposal")

    def test_calibrated_team_preferred(self):
        frags = [frag("F1", 0, 10, None, "TEAM_B", team="TEAM_A")]
        a, groups, *_ = api.assign(frags, ends_for(("F1", BOX, BOX, 1)))
        self.assertEqual(a["F1"]["persistent_team_id"], "TEAM_A")
        self.assertEqual(groups[0]["team_status"], "calibrated")

    def test_every_fragment_gets_exactly_one_id_in_start_order(self):
        frags = [frag("F2", 100, 110), frag("F1", 0, 10), frag("F3", 0, 5)]
        a, groups, *_ = api.assign(frags, ends_for(("F1", BOX, BOX, 1), ("F2", FAR, FAR, 1), ("F3", FAR, FAR, 1)))
        self.assertEqual(set(a), {"F1", "F2", "F3"})
        self.assertEqual([g["persistent_id"] for g in groups], ["ID001", "ID002", "ID003"])
        self.assertEqual(a["F1"]["persistent_id"], "ID001")

    def test_deterministic_and_input_not_mutated(self):
        frags = [frag("F1", 0, 10, vec(1, 0.1), "TEAM_A"), frag("F2", 300, 310, vec(1, 0.1)), frag("F3", 20, 30)]
        ends = ends_for(("F1", BOX, BOX, 1), ("F2", FAR, FAR, 1), ("F3", NEAR, NEAR, 1))
        before = json.dumps(frags, sort_keys=True)
        first = api.assign(frags, ends)
        second = api.assign(list(reversed(frags)), ends)
        self.assertEqual(first[:3], second[:3])
        self.assertEqual(json.dumps(frags, sort_keys=True), before)

    def test_rejects_bad_input(self):
        with self.assertRaises(api.InputError):
            api.assign([frag("F1", 0, 1), frag("F1", 5, 6)], {})
        with self.assertRaises(api.InputError):
            api.assign([frag("F1", 9, 1)], {})
        with self.assertRaises(api.InputError):
            api.assign([frag("F1", -1, 1)], {})
        with self.assertRaises(api.InputError):
            api.assign([{"start_frame": 0, "end_frame": 1}], {})

    def test_zero_or_bad_embedding_is_ignored(self):
        frags = [frag("F1", 0, 10, [0.0] * 8), frag("F2", 300, 310, [float("nan")] * 8)]
        a, *_ = api.assign(frags, ends_for(("F1", BOX, BOX, 1), ("F2", FAR, FAR, 1)))
        self.assertNotEqual(a["F1"]["persistent_id"], a["F2"]["persistent_id"])


class ReviewFollowUpTests(unittest.TestCase):
    """PR #6 review: parameter checks, malformed inputs, crowded scenes, cuts, mixed team evidence."""

    def test_params_are_validated(self):
        self.assertEqual(api.validate_params(None), {**api.DEFAULTS, "max_center_dist": 1.0,
                                                     "max_height_ratio": 1.5, "min_cosine": 0.88})
        bad = [{"max_gap": 0}, {"max_gap": -5}, {"max_gap": 1.5}, {"max_gap": True},
               {"max_center_dist": 0}, {"max_center_dist": -1}, {"max_center_dist": float("nan")},
               {"max_center_dist": float("inf")}, {"max_height_ratio": 0.9}, {"min_cosine": 0},
               {"min_cosine": 1.5}, {"min_cosine": -0.2}, {"min_cosine": float("nan")}, {"surprise": 1}]
        for params in bad:
            with self.subTest(params=params), self.assertRaises(ValueError):
                api.validate_params(params)

    def test_nearby_simultaneous_players_continue_separately(self):
        # F1 and F2 are on screen together, side by side; F3 appears where F1 vanished.
        frags = [frag("F1", 0, 10), frag("F2", 0, 10), frag("F3", 15, 25)]
        side = [190, 100, 230, 200]
        a, _, _, refused, *_ = api.assign(frags, ends_for(("F1", BOX, BOX, 1), ("F2", side, side, 1), ("F3", NEAR, NEAR, 1)))
        self.assertEqual(a["F1"]["persistent_id"], a["F3"]["persistent_id"])
        self.assertNotEqual(a["F2"]["persistent_id"], a["F3"]["persistent_id"])
        self.assertGreaterEqual(refused.get("temporal_overlap", 0), 1)

    def test_nearby_players_of_different_teams_are_not_linked(self):
        frags = [frag("F1", 0, 10, None, "TEAM_A"), frag("F2", 12, 20, None, "TEAM_B")]
        a, *_ = api.assign(frags, ends_for(("F1", BOX, BOX, 1), ("F2", NEAR, NEAR, 1)))
        self.assertNotEqual(a["F1"]["persistent_id"], a["F2"]["persistent_id"])

    def test_camera_cut_without_appearance_starts_new_id(self):
        # Same centre as BOX but 4x taller (cut to a close-up): height ratio 4 > 1.5 blocks continuity.
        same_centre_close_up = [70, -50, 170, 350]
        # Normal size but the other side of the frame right after the cut.
        elsewhere = [900, 300, 940, 400]
        frags = [frag("F1", 0, 100), frag("F2", 101, 150), frag("F3", 0, 100), frag("F4", 101, 150)]
        a, *_ = api.assign(frags, ends_for(("F1", BOX, BOX, 1), ("F2", same_centre_close_up, same_centre_close_up, 1),
                                           ("F3", [600, 300, 640, 400], [600, 300, 640, 400], 1),
                                           ("F4", elsewhere, elsewhere, 1)))
        self.assertNotEqual(a["F1"]["persistent_id"], a["F2"]["persistent_id"])
        self.assertNotEqual(a["F3"]["persistent_id"], a["F4"]["persistent_id"])
        self.assertNotEqual(a["F1"]["persistent_id"], a["F4"]["persistent_id"])

    def test_missing_embeddings_still_allow_continuity_only(self):
        frags = [frag("F1", 0, 10), frag("F2", 15, 25), frag("F3", 400, 410, vec(1, 0))]
        a, _, links, *_ = api.assign(frags, ends_for(("F1", BOX, BOX, 1), ("F2", NEAR, NEAR, 1), ("F3", FAR, FAR, 1)))
        self.assertEqual(a["F1"]["persistent_id"], a["F2"]["persistent_id"])
        self.assertNotEqual(a["F2"]["persistent_id"], a["F3"]["persistent_id"])
        self.assertEqual([l["kind"] for l in links], ["continuity"])

    def test_mixed_calibrated_and_uncalibrated_status(self):
        frags = [frag("F1", 0, 10, None, None, team="TEAM_A"), frag("F2", 15, 25, None, "TEAM_A"), frag("F3", 0, 5)]
        a, groups, *_ = api.assign(frags, ends_for(("F1", BOX, BOX, 1), ("F2", NEAR, NEAR, 1), ("F3", FAR, FAR, 1)))
        self.assertEqual(a["F2"]["persistent_team_status"], "mixed_calibrated_and_uncalibrated")
        self.assertEqual(a["F2"]["persistent_team_id"], "TEAM_A")
        g = next(g for g in groups if "F1" in g["fragments"])
        self.assertEqual((g["calibrated_fragments"], g["uncalibrated_fragments"]), (1, 1))
        self.assertEqual(a["F3"]["persistent_team_status"], "none")
        self.assertEqual(a["F3"]["persistent_team_id"], "UNKNOWN")

    def test_calibrated_and_uncalibrated_disagreeing_are_not_linked(self):
        frags = [frag("F1", 0, 10, None, None, team="TEAM_A"), frag("F2", 15, 25, None, "TEAM_B")]
        a, *_ = api.assign(frags, ends_for(("F1", BOX, BOX, 1), ("F2", NEAR, NEAR, 1)))
        self.assertNotEqual(a["F1"]["persistent_id"], a["F2"]["persistent_id"])
        self.assertEqual(a["F1"]["persistent_team_status"], "calibrated")
        self.assertEqual(a["F2"]["persistent_team_status"], "uncalibrated_proposal")

    def test_malformed_embeddings_are_ignored_and_counted(self):
        good = vec(1, 0)
        frags = [frag("F1", 0, 10, good), frag("F2", 300, 310, good[:4]), frag("F3", 600, 610, ["x"] * 8),
                 frag("F4", 900, 910, good)]
        a, _, _, _, _, issues = api.assign(frags, ends_for(("F1", BOX, BOX, 1), ("F2", FAR, FAR, 1),
                                                           ("F3", BOX, BOX, 1), ("F4", FAR, FAR, 1)))
        self.assertEqual(issues["embedding_dimension"], 8)
        self.assertEqual(issues["embeddings_ignored"], {"wrong_dimension": 1, "not_a_numeric_list": 1})
        self.assertEqual(a["F1"]["persistent_id"], a["F4"]["persistent_id"])
        self.assertNotEqual(a["F1"]["persistent_id"], a["F2"]["persistent_id"])
        self.assertNotEqual(a["F1"]["persistent_id"], a["F3"]["persistent_id"])

    def test_invalid_boxes_are_not_used_for_continuity(self):
        rows = [{"frame": 10, "fragment_id": "F1", "bbox_xyxy": BOX},
                {"frame": 11, "fragment_id": "F1", "bbox_xyxy": [150, 100, 120, 200]},     # x2 < x1
                {"frame": 15, "fragment_id": "F2", "bbox_xyxy": [float("nan"), 0, 1, 1]},
                {"frame": 16, "fragment_id": "F2", "bbox_xyxy": [1, 2, 3]}]
        ends, invalid = api.fragment_endpoints(rows)
        self.assertEqual(invalid, 3)
        self.assertEqual(ends["F1"]["last"], (10, BOX))
        self.assertIsNone(ends["F2"]["first"])
        a, *_ = api.assign([frag("F1", 10, 11), frag("F2", 15, 16)], ends)
        self.assertNotEqual(a["F1"]["persistent_id"], a["F2"]["persistent_id"])


class CliTests(unittest.TestCase):
    def make_run(self, root):
        run = root / "run"
        run.mkdir()
        frags = [frag("F1", 0, 1, vec(1, 0)), frag("F2", 3, 4)]
        (run / "fragments.json").write_text(json.dumps(frags))
        rows = [{"frame": f, "fragment_id": fid, "bbox_xyxy": b, "player_id": "UNK", "team_id": "UNKNOWN"}
                for f, fid, b in ((0, "F1", BOX), (1, "F1", BOX), (3, "F2", NEAR), (4, "F2", NEAR))]
        (run / "identified_boxes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        return run

    def run_cli(self, argv):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return api.main(argv)

    def test_cli_writes_new_files_and_leaves_run_untouched(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = self.make_run(root)
            before = {p.name: p.read_bytes() for p in run.iterdir()}
            self.assertEqual(self.run_cli(["--run-dir", str(run), "--out", str(root / "ids")]), 0)
            self.assertEqual({p.name: p.read_bytes() for p in run.iterdir()}, before)
            report = json.loads((root / "ids" / "persistent_ids.json").read_text())
            self.assertIsNone(report["accuracy_claim"])
            self.assertEqual(report["counts"]["persistent_ids"], 1)
            rows = [json.loads(l) for l in (root / "ids" / "persistent_boxes.jsonl").read_text().splitlines()]
            self.assertEqual({r["persistent_id"] for r in rows}, {"ID001"})
            self.assertTrue(all(r["player_id"] == "UNK" and r["team_id"] == "UNKNOWN" for r in rows))

    def test_cli_refuses_nonempty_out_and_same_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = self.make_run(root)
            (root / "busy").mkdir()
            (root / "busy" / "x").write_text("keep")
            for out in (root / "busy", run):
                with self.assertRaises(SystemExit):
                    self.run_cli(["--run-dir", str(run), "--out", str(out)])
            self.assertEqual((root / "busy" / "x").read_text(), "keep")

    def test_render_requires_video(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = self.make_run(root)
            with self.assertRaises(SystemExit):
                self.run_cli(["--run-dir", str(run), "--out", str(root / "o"), "--render"])

    def test_cli_rejects_bad_parameters_without_writing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = self.make_run(root)
            for extra in (["--max-gap", "-3"], ["--max-height-ratio", "0.5"], ["--min-cosine", "1.2"],
                          ["--max-center-dist", "0"]):
                with self.subTest(extra=extra), self.assertRaises(SystemExit):
                    self.run_cli(["--run-dir", str(run), "--out", str(root / "o")] + extra)
                self.assertFalse((root / "o").exists())

    def test_cli_malformed_inputs_fail_without_writing(self):
        cases = {
            "bad_json": lambda run: (run / "fragments.json").write_text("{not json"),
            "unknown_fragment": lambda run: (run / "identified_boxes.jsonl").write_text(
                json.dumps({"frame": 0, "fragment_id": "F9", "bbox_xyxy": BOX}) + "\n"),
            "string_frame": lambda run: (run / "identified_boxes.jsonl").write_text(
                json.dumps({"frame": "0", "fragment_id": "F1", "bbox_xyxy": BOX}) + "\n"),
            "bad_row_json": lambda run: (run / "identified_boxes.jsonl").write_text("[1, 2\n"),
        }
        for name, corrupt in cases.items():
            with self.subTest(case=name), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                run = self.make_run(root)
                corrupt(run)
                with self.assertRaises(SystemExit) as ctx:
                    self.run_cli(["--run-dir", str(run), "--out", str(root / "o")])
                self.assertIn("nothing written", str(ctx.exception.code))
                self.assertFalse((root / "o").exists())

    def test_cli_output_is_frame_ordered_and_fully_assigned(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = self.make_run(root)
            lines = (run / "identified_boxes.jsonl").read_text().splitlines()
            (run / "identified_boxes.jsonl").write_text("\n".join(reversed(lines)) + "\n")
            self.assertEqual(self.run_cli(["--run-dir", str(run), "--out", str(root / "ids")]), 0)
            rows = [json.loads(l) for l in (root / "ids" / "persistent_boxes.jsonl").read_text().splitlines()]
            self.assertEqual([r["frame"] for r in rows], sorted(r["frame"] for r in rows))
            self.assertEqual(len(rows), len(lines))
            report = json.loads((root / "ids" / "persistent_ids.json").read_text())
            checks = report["output_checks"]
            self.assertTrue(checks["every_row_has_assignment"] and checks["every_fragment_has_one_id"]
                            and checks["output_rows_frame_ordered"])
            self.assertFalse(checks["input_rows_frame_ordered"])
            self.assertEqual(report["identity_type"], "anonymous_unvalidated")
            self.assertIn("not a verified real-world player identity", report["semantics"])


if __name__ == "__main__":
    unittest.main()
