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
        a, groups, links, _, _ = api.assign(frags, ends_for(("F1", BOX, BOX, 11), ("F2", NEAR, NEAR, 11)))
        self.assertEqual(a["F1"]["persistent_id"], a["F2"]["persistent_id"])
        self.assertEqual(links[0]["kind"], "continuity")
        self.assertEqual(len(groups), 1)

    def test_far_or_late_reappearance_without_appearance_stays_split(self):
        frags = [frag("F1", 0, 10), frag("F2", 20, 30), frag("F3", 200, 210)]
        a, *_ = api.assign(frags, ends_for(("F1", BOX, BOX, 1), ("F2", FAR, FAR, 1), ("F3", BOX, BOX, 1)))
        self.assertEqual(len({a[f]["persistent_id"] for f in a}), 3)

    def test_appearance_links_long_gap(self):
        frags = [frag("F1", 0, 10, vec(1, 0.1)), frag("F2", 300, 310, vec(1, 0.12))]
        a, _, links, _, _ = api.assign(frags, ends_for(("F1", BOX, BOX, 1), ("F2", FAR, FAR, 1)))
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
        a, _, _, refused, _ = api.assign(frags, ends_for(("F1", BOX, BOX, 1), ("F3", FAR, FAR, 1), ("F2", BOX, BOX, 1)))
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
        a2, _, _, refused, _ = api.assign(frags, ends_for(("F1", BOX, BOX, 1), ("F2", FAR, FAR, 1), ("F3", BOX, [500, 100, 540, 200], 1)),
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
        with self.assertRaises(ValueError):
            api.assign([frag("F1", 0, 1), frag("F1", 5, 6)], {})
        with self.assertRaises(ValueError):
            api.assign([frag("F1", 9, 1)], {})

    def test_zero_or_bad_embedding_is_ignored(self):
        frags = [frag("F1", 0, 10, [0.0] * 8), frag("F2", 300, 310, [float("nan")] * 8)]
        a, *_ = api.assign(frags, ends_for(("F1", BOX, BOX, 1), ("F2", FAR, FAR, 1)))
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


if __name__ == "__main__":
    unittest.main()
