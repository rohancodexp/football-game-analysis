import unittest

import cv2
import numpy as np

from team_classifier import (
    ROLE_NONE,
    ROLE_OUTFIT_MISMATCH_CANDIDATE,
    ROLE_REFEREE_CANDIDATE,
    TEAM_A,
    TEAM_B,
    UNKNOWN,
    TeamAppearanceClassifier,
    appearance_feature,
    detect_candidate_outlier,
    extract_grass_mask,
    extract_torso_crop,
    get_foreground_stats,
)
from run_team_clustering import aggregate_track_candidates


class TeamClassifierTests(unittest.TestCase):
    def test_torso_crop_is_inside_frame(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        crop = extract_torso_crop(frame, [-20, -10, 80, 90])
        self.assertIsNotNone(crop)
        self.assertEqual(crop.ndim, 3)
        self.assertEqual(crop.shape[2], 3)

    def test_invalid_box_returns_none(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        self.assertIsNone(extract_torso_crop(frame, [20, 20, 20, 80]))

    def test_feature_is_finite(self):
        crop = np.full((40, 30, 3), (220, 220, 220), dtype=np.uint8)
        feature = appearance_feature(crop)
        self.assertIsNotNone(feature)
        self.assertTrue(np.isfinite(feature).all())
        self.assertAlmostEqual(float(np.linalg.norm(feature)), 1.0, places=5)

    def test_extract_grass_mask(self):
        # Green pitch pixel in HSV: Hue ~ 50 (green), Sat ~ 160, Val ~ 140
        green_hsv = np.full((20, 20, 3), (50, 160, 140), dtype=np.uint8)
        mask = extract_grass_mask(green_hsv)
        self.assertEqual(np.count_nonzero(mask), 20 * 20)

        # White pixel in HSV: Hue = 0, Sat = 0, Val = 255 -> NOT grass
        white_hsv = np.full((20, 20, 3), (0, 0, 255), dtype=np.uint8)
        mask_white = extract_grass_mask(white_hsv)
        self.assertEqual(np.count_nonzero(mask_white), 0)

        # Dark navy pixel in HSV: Hue = 110, Sat = 200, Val = 80 -> NOT grass
        navy_hsv = np.full((20, 20, 3), (110, 200, 80), dtype=np.uint8)
        mask_navy = extract_grass_mask(navy_hsv)
        self.assertEqual(np.count_nonzero(mask_navy), 0)

    def test_get_foreground_stats(self):
        # Pitch green BGR: B=35, G=140, R=50 -> converts to green HSV
        pitch_bgr = np.full((40, 40, 3), (35, 140, 50), dtype=np.uint8)
        stats = get_foreground_stats(pitch_bgr, min_foreground_ratio=0.25)
        self.assertGreater(stats["grass_ratio"], 0.95)
        self.assertLess(stats["foreground_ratio"], 0.05)
        self.assertFalse(stats["is_valid_foreground"])

        # White kit BGR: (230, 230, 230)
        white_bgr = np.full((40, 40, 3), (230, 230, 230), dtype=np.uint8)
        stats_w = get_foreground_stats(white_bgr, min_foreground_ratio=0.25)
        self.assertAlmostEqual(stats_w["grass_ratio"], 0.0, places=2)
        self.assertAlmostEqual(stats_w["foreground_ratio"], 1.0, places=2)
        self.assertTrue(stats_w["is_valid_foreground"])

        # Half pitch green, half white kit
        half_half = np.zeros((40, 40, 3), dtype=np.uint8)
        half_half[:20, :] = (35, 140, 50)
        half_half[20:, :] = (230, 230, 230)
        stats_half = get_foreground_stats(half_half, min_foreground_ratio=0.25)
        self.assertAlmostEqual(stats_half["foreground_ratio"], 0.5, places=1)
        self.assertAlmostEqual(stats_half["removed_percentage"], 50.0, places=1)
        self.assertTrue(stats_half["is_valid_foreground"])

    def test_get_foreground_stats_invalid_crop(self):
        stats = get_foreground_stats(None)
        self.assertFalse(stats["is_valid_foreground"])
        self.assertEqual(stats["foreground_ratio"], 0.0)

        empty = np.zeros((0, 0, 3), dtype=np.uint8)
        stats_empty = get_foreground_stats(empty)
        self.assertFalse(stats_empty["is_valid_foreground"])

    def test_appearance_feature_with_grass_masking(self):
        # Mixed crop: 60% white jersey, 40% grass
        crop = np.zeros((50, 40, 3), dtype=np.uint8)
        crop[:20, :] = (35, 140, 50)     # pitch grass
        crop[20:, :] = (230, 230, 230)   # white jersey

        feat_no_mask = appearance_feature(crop, use_grass_mask=False)
        feat_masked = appearance_feature(crop, use_grass_mask=True, min_foreground_ratio=0.25)

        self.assertIsNotNone(feat_no_mask)
        self.assertIsNotNone(feat_masked)
        # Masking out grass should produce a different feature vector
        self.assertFalse(np.allclose(feat_no_mask, feat_masked, atol=1e-3))

    def test_appearance_feature_rejects_mostly_grass_crop(self):
        # 90% pitch grass, 10% player
        crop = np.zeros((50, 40, 3), dtype=np.uint8)
        crop[:45, :] = (35, 140, 50)    # pitch grass
        crop[45:, :] = (230, 230, 230)  # small sliver

        feat = appearance_feature(crop, use_grass_mask=True, min_foreground_ratio=0.25)
        self.assertIsNone(feat)

    def test_detect_candidate_outlier_yellow_kit(self):
        # Bright yellow fluorescent kit: BGR (20, 220, 240)
        yellow_crop = np.full((40, 30, 3), (20, 220, 240), dtype=np.uint8)

        # Disabled check returns False
        flagged_dis, _ = detect_candidate_outlier(yellow_crop, enabled=False)
        self.assertFalse(flagged_dis)

        # Enabled check detects candidate yellow
        flagged_en, reason = detect_candidate_outlier(yellow_crop, enabled=True)
        self.assertTrue(flagged_en)
        self.assertEqual(reason, "candidate_non_outfield_yellow")

    def test_detect_candidate_outlier_regular_kits(self):
        # White kit: BGR (240, 240, 240)
        white_crop = np.full((40, 30, 3), (240, 240, 240), dtype=np.uint8)
        flagged_w, _ = detect_candidate_outlier(white_crop, enabled=True)
        self.assertFalse(flagged_w)

        # Dark navy kit: BGR (80, 25, 15)
        navy_crop = np.full((40, 30, 3), (80, 25, 15), dtype=np.uint8)
        flagged_n, _ = detect_candidate_outlier(navy_crop, enabled=True)
        self.assertFalse(flagged_n)

    def test_two_visual_groups_can_be_fitted(self):
        # Synthetic smoke test only; it does not establish real-video accuracy.
        white = [np.full((40, 30, 3), (225, 225, 225), dtype=np.uint8) for _ in range(8)]
        blue = [np.full((40, 30, 3), (150, 55, 20), dtype=np.uint8) for _ in range(8)]
        model = TeamAppearanceClassifier(min_samples=12, min_cluster_samples=4)
        report = model.fit(white + blue)
        self.assertEqual(set(report["cluster_sizes"]), {"TEAM_A", "TEAM_B"})
        prediction = model.predict(white[0])
        self.assertIn(prediction.team_id, {"TEAM_A", "TEAM_B"})

    def test_classifier_candidate_outlier_prediction(self):
        white = [np.full((40, 30, 3), (225, 225, 225), dtype=np.uint8) for _ in range(8)]
        blue = [np.full((40, 30, 3), (150, 55, 20), dtype=np.uint8) for _ in range(8)]
        model = TeamAppearanceClassifier(
            min_samples=12,
            min_cluster_samples=4,
            detect_candidate_outliers=True,
        )
        model.fit(white + blue)

        yellow_crop = np.full((40, 30, 3), (20, 220, 240), dtype=np.uint8)
        prediction = model.predict(yellow_crop)
        self.assertEqual(prediction.team_id, UNKNOWN)
        self.assertEqual(prediction.reason, "candidate_non_outfield_yellow")

    def test_classifier_insufficient_foreground_rejection(self):
        white = [np.full((40, 30, 3), (225, 225, 225), dtype=np.uint8) for _ in range(8)]
        blue = [np.full((40, 30, 3), (150, 55, 20), dtype=np.uint8) for _ in range(8)]
        model = TeamAppearanceClassifier(
            min_samples=12,
            min_cluster_samples=4,
            use_grass_mask=True,
            min_foreground_ratio=0.30,
        )
        model.fit(white + blue)

        # 90% pitch grass crop
        mostly_grass = np.full((40, 30, 3), (35, 140, 50), dtype=np.uint8)
        prediction = model.predict(mostly_grass)
        self.assertEqual(prediction.team_id, UNKNOWN)
        self.assertEqual(prediction.reason, "insufficient_foreground")

    def test_classifier_outlier_rejection(self):
        # Tight clusters: very consistent white and navy
        white = [np.full((40, 30, 3), (230, 230, 230), dtype=np.uint8) for _ in range(10)]
        blue = [np.full((40, 30, 3), (140, 40, 10), dtype=np.uint8) for _ in range(10)]
        model = TeamAppearanceClassifier(
            min_samples=12,
            min_cluster_samples=4,
            outlier_multiplier=1.2,
            minimum_radius=0.05,
        )
        model.fit(white + blue)

        # A completely different magenta kit: BGR (220, 20, 220)
        magenta = np.full((40, 30, 3), (220, 20, 220), dtype=np.uint8)
        prediction = model.predict(magenta)
        self.assertEqual(prediction.team_id, UNKNOWN)
        self.assertEqual(prediction.reason, "appearance_outlier")

    def test_predict_unfitted_returns_not_fitted(self):
        model = TeamAppearanceClassifier()
        crop = np.full((40, 30, 3), (220, 220, 220), dtype=np.uint8)
        prediction = model.predict(crop)
        self.assertEqual(prediction.team_id, UNKNOWN)
        self.assertEqual(prediction.reason, "not_fitted")


def _noisy_kits(rng, colour, count):
    return [
        np.clip(np.array(colour) + rng.integers(-12, 13, size=(40, 30, 3)), 0, 255).astype(
            np.uint8
        )
        for _ in range(count)
    ]


class TeamLabelStabilityTests(unittest.TestCase):
    """Task 1.3: reproducible anonymous labels and conservative candidates."""

    def setUp(self):
        rng = np.random.default_rng(3)
        self.white = _noisy_kits(rng, (225, 225, 225), 10)
        self.navy = _noisy_kits(rng, (150, 55, 20), 10)
        self.crops = self.white + self.navy
        self.ids = [f"D{i:06d}_T{i:06d}_000" for i in range(len(self.crops))]

    def _fit(self, model, crops, ids=None):
        return model.fit(crops, ids)

    def test_repeated_fit_is_identical(self):
        a = TeamAppearanceClassifier(min_samples=12, min_cluster_samples=4)
        b = TeamAppearanceClassifier(min_samples=12, min_cluster_samples=4)
        report_a = self._fit(a, self.crops, self.ids)
        report_b = self._fit(b, self.crops, self.ids)
        self.assertEqual(report_a, report_b)
        self.assertEqual(
            [a.predict(c).team_id for c in self.crops],
            [b.predict(c).team_id for c in self.crops],
        )

    def test_fit_is_permutation_invariant(self):
        baseline = TeamAppearanceClassifier(min_samples=12, min_cluster_samples=4)
        base_report = self._fit(baseline, self.crops, self.ids)
        base_labels = [baseline.predict(c).team_id for c in self.crops]

        for seed in (1, 2, 3, 4):
            order = np.random.default_rng(seed).permutation(len(self.crops))
            shuffled = [self.crops[i] for i in order]
            shuffled_ids = [self.ids[i] for i in order]

            model = TeamAppearanceClassifier(min_samples=12, min_cluster_samples=4)
            report = self._fit(model, shuffled, shuffled_ids)

            # Same partition, same labels, same anchor crop - only the input
            # order changed.
            self.assertEqual(report["cluster_fingerprint"], base_report["cluster_fingerprint"])
            self.assertEqual(report["anchor_crop_id"], base_report["anchor_crop_id"])
            self.assertEqual(report["cluster_sizes"], base_report["cluster_sizes"])
            self.assertEqual(report["outlier_radii"], base_report["outlier_radii"])
            self.assertEqual(
                [model.predict(c).team_id for c in shuffled],
                [base_labels[i] for i in order],
            )

    def test_anchor_crop_defines_team_a(self):
        model = TeamAppearanceClassifier(min_samples=12, min_cluster_samples=4)
        report = self._fit(model, self.crops, self.ids)
        self.assertEqual(report["labeling_rule"], "anchor_crop")
        anchor = self.crops[report["anchor_sample_index"]]
        self.assertEqual(model.predict(anchor).team_id, "TEAM_A")

    def test_crop_ids_length_must_match(self):
        model = TeamAppearanceClassifier(min_samples=12, min_cluster_samples=4)
        with self.assertRaises(ValueError):
            model.fit(self.crops, ["too", "few"])

    def test_candidate_flag_keeps_the_team_label(self):
        yellow = np.full((40, 30, 3), (20, 220, 240), dtype=np.uint8)

        model = TeamAppearanceClassifier(
            min_samples=12,
            min_cluster_samples=4,
            candidate_mode="multi",
            candidate_action="flag",
        )
        self._fit(model, self.crops, self.ids)
        prediction = model.predict(yellow)

        self.assertTrue(prediction.candidate.is_candidate)
        self.assertEqual(prediction.candidate.role, ROLE_REFEREE_CANDIDATE)
        # Flagging annotates; it must not silently invent or delete a label.
        self.assertIn(prediction.team_id, {TEAM_A, TEAM_B, UNKNOWN})
        self.assertNotEqual(prediction.reason, "role_candidate_rejected")

    def test_legacy_reject_path_matches_v5(self):
        yellow = np.full((40, 30, 3), (20, 220, 240), dtype=np.uint8)
        model = TeamAppearanceClassifier(
            min_samples=12, min_cluster_samples=4, detect_candidate_outliers=True
        )
        report = self._fit(model, self.crops, self.ids)
        self.assertEqual(model.candidate_mode, "legacy")
        self.assertEqual(model.candidate_action, "reject")
        self.assertTrue(report["detect_candidate_outliers"])
        prediction = model.predict(yellow)
        self.assertEqual(prediction.team_id, UNKNOWN)
        self.assertEqual(prediction.reason, "candidate_non_outfield_yellow")

    def test_multi_reject_forces_unknown_with_reason(self):
        yellow = np.full((40, 30, 3), (20, 220, 240), dtype=np.uint8)
        model = TeamAppearanceClassifier(
            min_samples=12,
            min_cluster_samples=4,
            candidate_mode="multi",
            candidate_action="reject",
        )
        self._fit(model, self.crops, self.ids)
        prediction = model.predict(yellow)
        self.assertEqual(prediction.team_id, UNKNOWN)
        self.assertEqual(prediction.reason, "role_candidate_rejected")
        self.assertTrue(prediction.candidate.is_candidate)

    def test_candidate_never_assigns_a_team_to_an_uncertain_crop(self):
        # Magenta is unlike either cluster and unlike a referee strip.
        magenta = np.full((40, 30, 3), (220, 20, 220), dtype=np.uint8)
        model = TeamAppearanceClassifier(
            min_samples=12,
            min_cluster_samples=4,
            outlier_multiplier=1.2,
            minimum_radius=0.05,
            candidate_mode="multi",
        )
        self._fit(model, self.crops, self.ids)
        prediction = model.predict(magenta)
        self.assertEqual(prediction.team_id, UNKNOWN)
        self.assertEqual(prediction.reason, "appearance_outlier")

    def test_off_mode_attaches_no_candidate(self):
        model = TeamAppearanceClassifier(
            min_samples=12, min_cluster_samples=4, candidate_mode="off"
        )
        self._fit(model, self.crops, self.ids)
        prediction = model.predict(self.white[0])
        self.assertIsNone(prediction.candidate)

    def test_invalid_candidate_knobs_raise(self):
        with self.assertRaises(ValueError):
            TeamAppearanceClassifier(candidate_mode="guess")
        with self.assertRaises(ValueError):
            TeamAppearanceClassifier(candidate_action="delete")

    def test_multisignal_detects_a_non_yellow_odd_kit(self):
        # v5's yellow-only rule cannot see this; the outfit-contrast/off-palette
        # signals can. It is flagged, not ejected.
        hooped = np.zeros((40, 30, 3), dtype=np.uint8)
        hooped[:20, :] = (250, 250, 250)   # white shirt
        hooped[20:, :] = (20, 20, 200)     # red shorts
        model = TeamAppearanceClassifier(
            min_samples=12, min_cluster_samples=4, candidate_mode="multi"
        )
        self._fit(model, self.crops, self.ids)
        prediction = model.predict(hooped)
        self.assertIsNotNone(prediction.candidate)

        legacy_model = TeamAppearanceClassifier(
            min_samples=12, min_cluster_samples=4, candidate_mode="legacy"
        )
        self._fit(legacy_model, self.crops, self.ids)
        legacy_prediction = legacy_model.predict(hooped)
        self.assertFalse(legacy_prediction.candidate.is_candidate)
        self.assertTrue(prediction.candidate.is_candidate)


    def test_fit_gate_removes_referees_from_the_model_not_the_output(self):
        yellow = np.full((40, 30, 3), (20, 220, 240), dtype=np.uint8)
        crops = self.crops + [yellow, yellow]

        gated = TeamAppearanceClassifier(
            min_samples=12, min_cluster_samples=4, candidate_mode="multi"
        )
        report = self._fit(gated, crops)
        self.assertEqual(report["fitting_set_candidate_gated"], 2)
        # Gate off: the two referee crops merge into one team cluster, so the
        # split is computed over a contaminated training set (12/10 vs 10/10).
        self.assertEqual(sorted(report["cluster_sizes"].values()), [10, 10])

        ungated = TeamAppearanceClassifier(
            min_samples=12, min_cluster_samples=4, candidate_mode="off"
        )
        ungated_report = self._fit(ungated, crops)
        self.assertEqual(ungated_report["fitting_set_candidate_gated"], 0)
        self.assertEqual(sorted(ungated_report["cluster_sizes"].values()), [10, 12])
        self.assertNotEqual(
            ungated_report["cluster_fingerprint"], report["cluster_fingerprint"]
        )

        prediction = gated.predict(yellow)
        self.assertTrue(prediction.candidate.is_candidate)
        self.assertIn(prediction.team_id, {TEAM_A, TEAM_B, UNKNOWN})


class TrackCandidateEvidenceTests(unittest.TestCase):
    def _sample(self, track_id, role, score):
        return {
            "track_id": track_id,
            "candidate_role": role,
            "candidate_score": score,
            "crop_prediction": TEAM_A if role == ROLE_NONE else UNKNOWN,
        }

    def test_single_odd_crop_does_not_exclude(self):
        samples = [self._sample(7, ROLE_OUTFIT_MISMATCH_CANDIDATE, 0.95)] + [
            self._sample(7, ROLE_NONE, 0.0) for _ in range(5)
        ]
        stats = aggregate_track_candidates(samples)
        self.assertFalse(stats[7]["persistent_evidence"])
        self.assertAlmostEqual(stats[7]["candidate_support"], 1 / 6, places=4)

    def test_persistent_evidence_excludes(self):
        samples = [self._sample(7, ROLE_REFEREE_CANDIDATE, 0.9) for _ in range(5)] + [
            self._sample(7, ROLE_NONE, 0.0) for _ in range(1)
        ]
        stats = aggregate_track_candidates(samples)
        self.assertTrue(stats[7]["persistent_evidence"])
        self.assertEqual(stats[7]["dominant_role"], ROLE_REFEREE_CANDIDATE)

    def test_many_weak_candidates_do_not_exclude(self):
        samples = [self._sample(3, ROLE_REFEREE_CANDIDATE, 0.55) for _ in range(6)]
        stats = aggregate_track_candidates(samples)
        self.assertEqual(stats[3]["candidate_support"], 1.0)
        self.assertFalse(stats[3]["persistent_evidence"])

    def test_unreadable_crops_are_not_scored(self):
        samples = [
            self._sample(9, ROLE_REFEREE_CANDIDATE, 0.9),
            self._sample(9, ROLE_REFEREE_CANDIDATE, 0.9),
            {"track_id": 9, "candidate_role": None, "candidate_score": 0.0},
        ]
        stats = aggregate_track_candidates(samples)
        self.assertEqual(stats[9]["scored_crops"], 2)
        self.assertEqual(stats[9]["candidate_support"], 1.0)


class TrackCandidateExclusionScopeTests(unittest.TestCase):
    """`candidate_action` is crop-level; track exclusion is gated by mode."""

    def _setup(self, persistent=True, team="TEAM_A"):
        track_results = {5: {"team_id": team, "reason": "track_majority_vote"}}
        evidence = {
            5: {
                "persistent_evidence": persistent,
                "candidate_support": 1.0 if persistent else 0.2,
            }
        }
        return track_results, evidence

    def test_flag_action_still_excludes_a_persistent_candidate_track(self):
        """The crop kept its label, but a whole track of them is sufficient evidence."""
        from run_team_clustering import apply_track_candidate_exclusions

        for action in ("flag", "reject"):
            with self.subTest(action=action):
                track_results, evidence = self._setup()
                excluded = apply_track_candidate_exclusions(
                    track_results, evidence, candidate_mode="multi"
                )
                self.assertEqual(excluded, [5])
                self.assertEqual(track_results[5]["team_id"], UNKNOWN)
                self.assertEqual(
                    track_results[5]["team_id_before_candidate_exclusion"], "TEAM_A"
                )
                self.assertEqual(track_results[5]["reason"], "persistent_role_candidate")

    def test_single_odd_crop_never_reaches_exclusion(self):
        from run_team_clustering import apply_track_candidate_exclusions

        track_results, evidence = self._setup(persistent=False)
        self.assertEqual(
            apply_track_candidate_exclusions(
                track_results, evidence, candidate_mode="multi"
            ),
            [],
        )
        self.assertEqual(track_results[5]["team_id"], "TEAM_A")

    def test_candidate_mode_off_disables_exclusion(self):
        from run_team_clustering import apply_track_candidate_exclusions

        track_results, evidence = self._setup()
        self.assertEqual(
            apply_track_candidate_exclusions(
                track_results, evidence, candidate_mode="off"
            ),
            [],
        )
        self.assertEqual(track_results[5]["team_id"], "TEAM_A")

    def test_already_unknown_tracks_are_not_relabelled(self):
        from run_team_clustering import apply_track_candidate_exclusions

        track_results, evidence = self._setup(team=UNKNOWN)
        self.assertEqual(
            apply_track_candidate_exclusions(
                track_results, evidence, candidate_mode="legacy"
            ),
            [],
        )
        self.assertNotIn("team_id_before_candidate_exclusion", track_results[5])

    def setUp(self):
        rng = np.random.default_rng(3)
        self.crops = _noisy_kits(rng, (225, 225, 225), 10) + _noisy_kits(
            rng, (150, 55, 20), 10
        )

    def test_classifier_flag_action_keeps_the_crop_label(self):
        # The crop-level half of the same rule, inside the classifier itself.
        yellow = np.full((40, 30, 3), (20, 220, 240), dtype=np.uint8)
        model = TeamAppearanceClassifier(
            min_samples=12,
            min_cluster_samples=4,
            candidate_mode="legacy",
            candidate_action="flag",
        )
        model.fit(self.crops)
        prediction = model.predict(yellow)
        self.assertTrue(prediction.candidate.is_candidate)
        self.assertNotEqual(prediction.reason, "candidate_non_outfield_yellow")


if __name__ == "__main__":
    unittest.main()
