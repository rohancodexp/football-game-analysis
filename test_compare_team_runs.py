import unittest

from compare_team_runs import (
    choose_alignment,
    compare,
    confidence_distributions,
    crop_key,
)


def _report(index, tracks, crops, with_distributions=False):
    report = {
        "input_index": index,
        "fit_settings": {"outlier_multiplier": 1.35},
        "fit_report": {"cluster_sizes": {"TEAM_A": 1, "TEAM_B": 1}},
        "tracks": {
            str(tid): {"team_id": team, "confidence": 0.5} for tid, team in tracks.items()
        },
        "samples": [
            {
                "track_id": tid,
                "frame": frame,
                "crop_prediction": team,
                "crop_confidence": conf,
            }
            for (tid, frame), (team, conf) in crops.items()
        ],
    }
    if with_distributions:
        report["crop_confidence_distributions"] = confidence_distributions(
            {"samples": report["samples"]}
        )
    return report


class CompareTeamRunsTests(unittest.TestCase):
    def setUp(self):
        self.index = r"results\crop_index.json"
        self.identity = _report(
            self.index,
            {1: "TEAM_A", 2: "TEAM_B", 3: "UNKNOWN"},
            {(1, 10): ("TEAM_A", 0.9), (2, 20): ("TEAM_B", 0.4)},
        )
        self.flipped = _report(
            self.index,
            {1: "TEAM_B", 2: "TEAM_A", 3: "UNKNOWN"},
            {(1, 10): ("TEAM_B", 0.9), (2, 20): ("TEAM_A", 0.4)},
        )

    def test_swapped_vocabulary_is_detected(self):
        name, alias = choose_alignment(self.identity, self.flipped)
        self.assertEqual(name, "swap")
        self.assertEqual(alias["TEAM_A"], "TEAM_B")

    def test_identical_run_aligns_to_identity(self):
        name, _ = choose_alignment(self.identity, self.identity)
        self.assertEqual(name, "identity")

    def test_anonymous_label_flip_is_not_reported_as_a_team_change(self):
        summary = compare(self.identity, self.flipped)
        self.assertEqual(summary["track_predictions"]["changed_count"], 0)
        self.assertEqual(summary["crop_predictions"]["changed_count"], 0)

    def test_real_change_survives_alignment(self):
        moved = _report(
            self.index,
            {1: "TEAM_A", 2: "TEAM_A", 3: "UNKNOWN"},
            {(1, 10): ("TEAM_A", 0.9), (2, 20): ("TEAM_A", 0.1)},
        )
        summary = compare(self.identity, moved)
        self.assertEqual(summary["track_predictions"]["changed_tracks"], ["2"])

    def test_unknown_is_preserved_as_unknown(self):
        both_unknown = _report(
            self.index,
            {1: "TEAM_A", 2: "TEAM_B", 3: "UNKNOWN"},
            {(1, 10): ("TEAM_A", 0.9), (2, 20): ("TEAM_B", 0.4)},
        )
        summary = compare(self.identity, both_unknown)
        self.assertEqual(summary["track_predictions"]["reference"]["UNKNOWN"]["count"], 1)
        self.assertEqual(summary["track_predictions"]["candidate"]["UNKNOWN"]["count"], 1)
        self.assertEqual(summary["track_predictions"]["changed_count"], 0)

    def test_confidence_distributions_fall_back_to_samples(self):
        summary = compare(self.identity, self.flipped)
        dist = summary["confidence_distributions"]["crop"]
        self.assertEqual(dist["TEAM_A"]["count"], {"old": 1, "new": 1})
        self.assertEqual(dist["TEAM_A"]["histogram_shift"], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    def test_different_input_dataset_is_flagged(self):
        other = _report(
            r"results\other_video_crop_index.json",
            {1: "TEAM_A", 2: "TEAM_B", 3: "UNKNOWN"},
            {(1, 10): ("TEAM_A", 0.9), (2, 20): ("TEAM_B", 0.4)},
        )
        summary = compare(self.identity, other)
        self.assertFalse(summary["inputs"]["same_input_dataset"])
        self.assertIn("not comparable", summary["inputs"]["comparability_warning"].lower())

    def test_crop_key_is_version_independent(self):
        self.assertEqual(crop_key({"track_id": 7, "frame": 3}), "track7_frame3")
        self.assertEqual(
            crop_key({"track_id": 7, "frame": 3, "detection_id": "D000003_T000007_000"}),
            "track7_frame3",
        )


if __name__ == "__main__":
    unittest.main()
