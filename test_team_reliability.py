"""Controlled synthetic logic fixtures ONLY, not football calibration or labels."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from team_classifier import (
    TEAM_A, TEAM_B, UNKNOWN, TeamAppearanceClassifier, TeamPrediction,
    appearance_feature,
)
from team_reliability import (
    FEATURE_VERSION, assess_separation, load_team_calibration, measure_separation,
)


# Arbitrary synthetic test boundaries; never a real-video policy.
SYNTHETIC_THRESHOLDS = {
    "min_centroid_distance": 0.2,
    "max_within_radius": 0.25,
    "min_cluster_fraction": 0.2,
    "min_mean_margin": 0.1,
    "min_seed_agreement": 0.75,
    "max_outlier_fraction": 0.05,
    "min_crop_score": 0.4,
    "min_crop_margin": 0.1,
}
TARGET_HASH = "a" * 64
VALIDATION_HASH = "b" * 64


def solid(colour):
    return np.full((40, 30, 3), colour, dtype=np.uint8)


def red_blue():
    return [solid((20, 20, 220)) for _ in range(10)] + [
        solid((220, 20, 20)) for _ in range(10)
    ]


def geometry():
    return measure_separation(
        np.array([[0.0, 0.0], [0.0, 0.2], [1.0, 0.0], [1.0, 0.2]]),
        [0, 0, 1, 1], [[0.0, 0.1], [1.0, 0.1]], [0.2, 0.2],
        [[0, 0, 1, 1], [1, 1, 0, 0]],
    )


class SyntheticPolicyFixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.evidence = self.directory / "synthetic_evidence.txt"
        self.evidence.write_bytes(b"Synthetic logic fixture only; no human football labels.\n")
        self.path = self.directory / "synthetic_policy.json"
        self.profile = {
            "kind": "human_reviewed_validation",
            "feature_version": FEATURE_VERSION,
            "reviewer": "Synthetic test actor, not an actual human review",
            "rationale": "Controlled logic fixture ONLY, not football calibration",
            "validation_video_sha256": [VALIDATION_HASH],
            "evidence_report": self.evidence.name,
            "evidence_sha256": hashlib.sha256(self.evidence.read_bytes()).hexdigest(),
            "thresholds": dict(SYNTHETIC_THRESHOLDS),
        }

    def write_profile(self):
        self.path.write_text(json.dumps(self.profile), encoding="utf-8")

    def calibration(self):
        self.write_profile()
        return load_team_calibration(self.path, TARGET_HASH)


class SeparationMeasurementTests(unittest.TestCase):
    def test_exact_geometry_and_cluster_summaries(self):
        metrics = geometry()
        self.assertEqual(metrics["feature_version"], "hsv_lab_v1")
        self.assertEqual(metrics["centroid_distance"], 1.0)
        self.assertEqual(metrics["min_cluster_fraction"], 0.5)
        self.assertAlmostEqual(metrics["max_within_radius"], 0.1)
        self.assertAlmostEqual(metrics["mean_margin"], np.sqrt(1.01) - 0.1)
        self.assertEqual(metrics["min_seed_agreement"], 1.0)
        self.assertEqual(metrics["outlier_fraction"], 0.0)
        self.assertAlmostEqual(metrics["cluster_ball_gap"], 0.8)
        self.assertEqual(metrics["initialization_trials"], 3)
        for k, cluster in enumerate(metrics["clusters"]):
            self.assertEqual(cluster["cluster"], k)
            self.assertEqual(cluster["count"], 2)
            for key, expected in (
                ("within_distances", 0.1), ("other_distances", np.sqrt(1.01)),
                ("margins", np.sqrt(1.01) - 0.1),
            ):
                self.assertAlmostEqual(cluster[key]["rms"], expected)
                self.assertEqual(set(cluster[key]["quantiles"]), {
                    "q0", "q25", "q50", "q75", "q90", "q95", "q100",
                })
                for quantile in cluster[key]["quantiles"].values():
                    self.assertAlmostEqual(quantile, expected)
        json.dumps(metrics, allow_nan=False)

    def test_radius_is_actual_max_not_rejection_radius(self):
        x = [[0.0], [0.2], [0.8], [1.0]]
        labels = [0, 0, 1, 1]
        metrics = measure_separation(x, labels, [[0.1], [0.9]], [0.05, 0.2], [labels])
        self.assertAlmostEqual(metrics["max_within_radius"], 0.1)
        self.assertAlmostEqual(metrics["cluster_ball_gap"], 0.6)
        self.assertEqual(metrics["outlier_fraction"], 0.5)
        wide = measure_separation(x, labels, [[0.1], [0.9]], [1, 1], [labels])
        self.assertEqual(wide["max_within_radius"], metrics["max_within_radius"])
        self.assertEqual(wide["cluster_ball_gap"], metrics["cluster_ball_gap"])
        self.assertEqual(wide["outlier_fraction"], 0.0)

    def test_radius_boundary_is_inclusive(self):
        labels = [0, 0, 1, 1]
        metrics = measure_separation(
            [[0.0], [0.25], [0.75], [1.0]], labels,
            [[0.125], [0.875]], [0.125, 0.125], [labels],
        )
        self.assertEqual(metrics["outlier_fraction"], 0.0)

    def test_seed_agreement_aligns_identity_or_swap_and_takes_minimum(self):
        labels = [0, 0, 1, 1]
        metrics = measure_separation(
            [[0], [0], [1], [1]], labels, [[0], [1]], [0, 0],
            [labels, [1, 1, 0, 0], [1, 0, 1, 1]],
        )
        self.assertEqual(metrics["min_seed_agreement"], 0.75)
        self.assertEqual(metrics["initialization_trials"], 4)
        worst = measure_separation(
            [[0], [0], [1], [1]], labels, [[0], [1]], [0, 0], [[0, 1, 0, 1]],
        )
        self.assertEqual(worst["min_seed_agreement"], 0.5)

    def test_absent_alternates_do_not_claim_stability(self):
        metrics = measure_separation([[0], [1]], [0, 1], [[0], [1]], [0, 0], [])
        self.assertIsNone(metrics["min_seed_agreement"])
        self.assertEqual(metrics["initialization_trials"], 1)

    def test_negative_margin_uses_baseline_labels_not_reassignment(self):
        metrics = measure_separation([[0], [1]], [1, 0], [[0], [1]], [0, 0], [[1, 0]])
        self.assertEqual(metrics["mean_margin"], -1.0)
        self.assertEqual(metrics["cluster_ball_gap"], -1.0)
        self.assertEqual(metrics["outlier_fraction"], 1.0)

    def test_empty_cluster_has_explicit_missing_summary(self):
        metrics = measure_separation([[0], [0]], [0, 0], [[0], [0]], [0, 0], [[0, 0]])
        self.assertEqual(metrics["min_cluster_fraction"], 0.0)
        self.assertIsNone(metrics["cluster_ball_gap"])
        self.assertEqual(metrics["clusters"][1]["count"], 0)
        self.assertEqual(metrics["clusters"][1]["within_distances"], {
            "rms": None, "quantiles": None,
        })
        json.dumps(metrics, allow_nan=False)

    def test_deterministic_under_permutation_repetition_without_mutation(self):
        rng = np.random.default_rng(5)
        x = rng.random((180, 8))
        labels = rng.integers(0, 2, len(x))
        centres = np.stack([x[labels == k].mean(axis=0) for k in range(2)])
        radii = np.array([0.5, 0.6])
        alternates = [1 - labels, rng.integers(0, 2, len(x))]
        originals = [a.copy() for a in [x, labels, centres, radii, *alternates]]
        expected = measure_separation(x, labels, centres, radii, alternates)
        for _ in range(4):
            self.assertEqual(measure_separation(x, labels, centres, radii, alternates), expected)
            order = rng.permutation(len(x))
            self.assertEqual(measure_separation(
                x[order], labels[order], centres, radii, [a[order] for a in alternates],
            ), expected)
        for actual, original in zip([x, labels, centres, radii, *alternates], originals):
            np.testing.assert_array_equal(actual, original)

    def test_replicated_samples_preserve_scalar_geometry(self):
        x = np.array([[0.0], [0.25], [0.75], [1.0]])
        labels = np.array([0, 0, 1, 1])
        baseline = measure_separation(x, labels, [[0.125], [0.875]], [0.2, 0.2], [labels])
        repeated = measure_separation(
            np.repeat(x, 3, axis=0), np.repeat(labels, 3),
            [[0.125], [0.875]], [0.2, 0.2], [np.repeat(labels, 3)],
        )
        for key in baseline.keys() - {"clusters"}:
            self.assertEqual(baseline[key], repeated[key])
        self.assertEqual([c["count"] for c in repeated["clusters"]], [6, 6])

    def test_invalid_inputs_fail_explicitly(self):
        base = dict(x=[[0], [1]], labels=[0, 1], centres=[[0], [1]], radii=[0.2, 0.2],
                    alternate_labels=[[0, 1]])
        for key, value in (
            ("x", []), ("x", [[np.nan], [1]]), ("x", [[np.inf], [1]]),
            ("centres", [[0]]), ("centres", [[0], [np.inf]]),
            ("radii", [0.2]), ("radii", [0.2, -1]), ("radii", [0.2, np.nan]),
            ("labels", [0]), ("labels", [0, 2]), ("labels", [0, np.nan]),
            ("alternate_labels", [[0]]), ("alternate_labels", [[0, 2]]),
        ):
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                measure_separation(**{**base, key: value})


class CalibrationLoaderTests(SyntheticPolicyFixture):
    def test_none_path_means_uncalibrated(self):
        self.assertIsNone(load_team_calibration(None, None))

    def test_loads_relative_evidence_and_records_exact_provenance(self):
        calibration = self.calibration()
        self.assertEqual(calibration["source_path"], str(self.path.resolve()))
        self.assertEqual(calibration["profile_sha256"], hashlib.sha256(self.path.read_bytes()).hexdigest())
        for key, value in self.profile.items():
            self.assertEqual(calibration[key], value)
        self.assertEqual(calibration["thresholds"]["min_crop_score"], 0.4)
        self.assertEqual(calibration["thresholds"]["min_crop_margin"], 0.1)

    def test_rejects_invalid_review_metadata(self):
        for key, value in (
            ("kind", "synthetic"), ("kind", None),
            ("feature_version", "future_version"), ("feature_version", None),
            ("reviewer", ""), ("reviewer", " \t"), ("reviewer", 3),
            ("rationale", ""), ("rationale", None),
            ("validation_video_sha256", []), ("validation_video_sha256", VALIDATION_HASH),
            ("validation_video_sha256", ["B" * 64]), ("validation_video_sha256", ["b" * 63]),
            ("validation_video_sha256", ["z" * 64]), ("validation_video_sha256", [True]),
            ("validation_video_sha256", [VALIDATION_HASH, "invalid"]),
        ):
            with self.subTest(key=key, value=value):
                original = self.profile[key]
                self.profile[key] = value
                with self.assertRaises(ValueError):
                    self.calibration()
                self.profile[key] = original

    def test_missing_required_fields_fail(self):
        for key in list(self.profile):
            with self.subTest(key=key):
                value = self.profile.pop(key)
                with self.assertRaises(ValueError):
                    self.calibration()
                self.profile[key] = value

    def test_target_video_leakage_is_rejected_anywhere_in_validation_list(self):
        self.profile["validation_video_sha256"] = [VALIDATION_HASH, TARGET_HASH]
        with self.assertRaisesRegex(ValueError, "target video"):
            self.calibration()

    def test_invalid_target_digest_cannot_bypass_leakage_check(self):
        self.write_profile()
        for target in (None, "", "A" * 64, "x" * 64, True):
            with self.subTest(target=target), self.assertRaises(ValueError):
                load_team_calibration(self.path, target)

    def test_evidence_must_exist_and_match_exact_bytes(self):
        self.calibration()
        self.evidence.write_bytes(b"Tampered synthetic evidence")
        with self.assertRaisesRegex(ValueError, "mismatch"):
            load_team_calibration(self.path, TARGET_HASH)
        self.evidence.unlink()
        with self.assertRaisesRegex(ValueError, "evidence"):
            load_team_calibration(self.path, TARGET_HASH)

    def test_evidence_paths_must_be_relative_and_point_to_a_file(self):
        for name in ("", "  ", None, str(self.evidence.resolve()), "C:/evidence.txt",
                     "C:evidence.txt", "\\\\host\\share\\evidence.txt", "/evidence.txt", "."):
            with self.subTest(name=name):
                self.profile["evidence_report"] = name
                with self.assertRaises(ValueError):
                    self.calibration()

    def test_evidence_digest_must_be_lower_case_sha256(self):
        for digest in (None, "", "b" * 63, "B" * 64, "x" * 64, True):
            with self.subTest(digest=digest):
                self.profile["evidence_sha256"] = digest
                with self.assertRaises(ValueError):
                    self.calibration()

    def test_threshold_keys_are_exact(self):
        for key in SYNTHETIC_THRESHOLDS:
            with self.subTest(key=key):
                self.profile["thresholds"] = dict(SYNTHETIC_THRESHOLDS)
                del self.profile["thresholds"][key]
                with self.assertRaises(ValueError):
                    self.calibration()
        self.profile["thresholds"] = {**SYNTHETIC_THRESHOLDS, "accuracy": 0.9}
        with self.assertRaises(ValueError):
            self.calibration()
        for thresholds in (None, [], 1):
            self.profile["thresholds"] = thresholds
            with self.assertRaises(ValueError):
                self.calibration()

    def test_thresholds_are_finite_nonbool_numbers_in_legitimate_ranges(self):
        bounds = {
            "min_centroid_distance": (0, 2), "max_within_radius": (0, 2),
            "min_mean_margin": (0, 2), "min_crop_margin": (0, 2),
            "min_cluster_fraction": (0, 0.5), "min_seed_agreement": (0.5, 1),
            "max_outlier_fraction": (0, 1), "min_crop_score": (0, 1),
        }
        for key, (lower, upper) in bounds.items():
            for value in (True, False, None, "0.5", [], {}, float("nan"), float("inf"),
                          -float("inf"), lower - 0.01, upper + 0.01, 10 ** 400):
                with self.subTest(key=key, value=value):
                    self.profile["thresholds"] = {**SYNTHETIC_THRESHOLDS, key: value}
                    with self.assertRaises(ValueError):
                        self.calibration()
            for boundary in (lower, upper):
                with self.subTest(key=key, boundary=boundary):
                    self.profile["thresholds"] = {**SYNTHETIC_THRESHOLDS, key: boundary}
                    self.assertEqual(self.calibration()["thresholds"][key], boundary)

    def test_corrupt_or_ambiguous_json_rejected(self):
        for text in ("{", "null", "[]", '"policy"', '{"kind": 1, "kind": 2}'):
            with self.subTest(text=text):
                self.path.write_text(text, encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_team_calibration(self.path, TARGET_HASH)
        self.write_profile()
        text = self.path.read_text(encoding="utf-8").replace(
            '"min_crop_score": 0.4', '"min_crop_score": 0.4, "min_crop_score": 0.8',
        )
        self.path.write_text(text, encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            load_team_calibration(self.path, TARGET_HASH)
        self.path.write_bytes(b"\xff\xff")
        with self.assertRaises(ValueError):
            load_team_calibration(self.path, TARGET_HASH)

    def test_missing_profile_rejected(self):
        with self.assertRaises(ValueError):
            load_team_calibration(self.path, TARGET_HASH)


class SeparationAssessmentTests(SyntheticPolicyFixture):
    def test_no_calibration_is_always_inconclusive(self):
        for metrics in (geometry(), {}, None, {"centroid_distance": 2}):
            result = assess_separation(metrics)
            self.assertEqual(result["decision"], "inconclusive")
            self.assertEqual(result["reason"], "uncalibrated_team_separation")
            self.assertIsNone(result["calibration"])
            self.assertEqual(result["checks"], {})

    def test_accepted_only_against_synthetic_review_fixture(self):
        calibration = self.calibration()
        metrics = geometry()
        result = assess_separation(metrics, calibration)
        self.assertEqual(result["decision"], "accepted")
        self.assertEqual(result["metrics"], metrics)
        self.assertEqual(result["calibration"], calibration)
        self.assertTrue(all(c["passed"] for c in result["checks"].values()))
        self.assertNotIn("min_crop_score", result["checks"])
        self.assertNotIn("min_crop_margin", result["checks"])

    def test_each_min_max_check_is_inclusive_and_can_reject_independently(self):
        calibration = self.calibration()
        checks = {
            "min_centroid_distance": ("centroid_distance", -1),
            "max_within_radius": ("max_within_radius", 1),
            "min_cluster_fraction": ("min_cluster_fraction", -1),
            "min_mean_margin": ("mean_margin", -1),
            "min_seed_agreement": ("min_seed_agreement", -1),
            "max_outlier_fraction": ("outlier_fraction", 1),
        }
        for key, (metric, direction) in checks.items():
            with self.subTest(key=key):
                metrics = geometry()
                metrics[metric] = calibration["thresholds"][key]
                boundary = assess_separation(metrics, calibration)
                self.assertEqual(boundary["decision"], "accepted")
                metrics[metric] += direction * 0.001
                rejected = assess_separation(metrics, calibration)
                self.assertEqual(rejected["decision"], "rejected")
                self.assertFalse(rejected["checks"][key]["passed"])
                self.assertEqual(rejected["checks"][key]["operator"], ">=" if direction < 0 else "<=")

    def test_missing_nonfinite_or_boolean_metrics_cannot_pass(self):
        calibration = self.calibration()
        keys = ("centroid_distance", "max_within_radius", "min_cluster_fraction",
                "mean_margin", "min_seed_agreement", "outlier_fraction")
        for key in keys:
            for value in (None, True, False, "1", float("nan"), float("inf"), -float("inf")):
                with self.subTest(key=key, value=value):
                    metrics = {**geometry(), key: value}
                    self.assertEqual(assess_separation(metrics, calibration)["decision"], "inconclusive")
            metrics = geometry()
            del metrics[key]
            self.assertEqual(assess_separation(metrics, calibration)["decision"], "inconclusive")

    def test_degenerate_centres_and_empty_clusters_cannot_pass_zero_policy_bounds(self):
        self.profile["thresholds"].update(min_centroid_distance=0, min_cluster_fraction=0)
        calibration = self.calibration()
        for key in ("centroid_distance", "min_cluster_fraction"):
            result = assess_separation({**geometry(), key: 0.0}, calibration)
            self.assertEqual(result["decision"], "rejected")
            self.assertEqual(result["reason"], "degenerate_team_clusters")

    def test_invalid_policy_cannot_pass(self):
        for calibration in ({}, [], {"thresholds": {}}, {"thresholds": SYNTHETIC_THRESHOLDS.keys()}):
            self.assertEqual(assess_separation(geometry(), calibration)["decision"], "inconclusive")

    def test_crop_thresholds_are_loaded_but_not_separation_checks(self):
        self.profile["thresholds"].update(min_crop_score=1.0, min_crop_margin=2.0)
        result = assess_separation(geometry(), self.calibration())
        self.assertEqual(result["decision"], "accepted")
        self.assertNotIn("min_crop_score", result["checks"])
        self.assertNotIn("min_crop_margin", result["checks"])

    def test_actual_imbalance_radius_and_seed_geometry_fail_synthetic_policy(self):
        calibration = self.calibration()
        imbalanced = measure_separation(
            [[0]] * 9 + [[1]], [0] * 9 + [1], [[0], [1]], [0, 0], [[0] * 9 + [1]],
        )
        broad = measure_separation(
            [[0], [0.6], [1.4], [2]], [0, 0, 1, 1], [[0.3], [1.7]], [1, 1], [[0, 0, 1, 1]],
        )
        unstable = measure_separation(
            [[0], [0], [1], [1]], [0, 0, 1, 1], [[0], [1]], [0, 0], [[0, 1, 0, 1]],
        )
        for metrics, key in ((imbalanced, "min_cluster_fraction"),
                             (broad, "max_within_radius"), (unstable, "min_seed_agreement")):
            with self.subTest(key=key):
                result = assess_separation(metrics, calibration)
                self.assertEqual(result["decision"], "rejected")
                self.assertFalse(result["checks"][key]["passed"])


class ClassifierReliabilityTests(SyntheticPolicyFixture):
    def test_raw_red_blue_proposals_need_separate_calibrated_assessment(self):
        crops = red_blue()
        model = TeamAppearanceClassifier()
        report = model.fit(crops)
        metrics = report["separation"]
        self.assertEqual(metrics["initialization_trials"], 3)
        self.assertEqual(assess_separation(metrics)["decision"], "inconclusive")
        self.assertEqual(assess_separation(metrics, self.calibration())["decision"], "accepted")
        self.assertNotEqual(model.predict(crops[0]).team_id, model.predict(crops[-1]).team_id)
        self.assertIn(model.predict(crops[0]).team_id, (TEAM_A, TEAM_B))

    def test_near_same_colour_fits_but_fails_synthetic_separation_policy(self):
        crops = [solid((20, 20, 220))] * 10 + [solid((21, 21, 219))] * 10
        model = TeamAppearanceClassifier()
        report = model.fit(crops)
        self.assertEqual(assess_separation(report["separation"])["decision"], "inconclusive")
        assessed = assess_separation(report["separation"], self.calibration())
        self.assertEqual(assessed["decision"], "rejected")
        self.assertFalse(assessed["checks"]["min_centroid_distance"]["passed"])
        # The low-level API still proposes clusters; reliability belongs to run.py.
        self.assertIn(model.predict(crops[0]).team_id, (TEAM_A, TEAM_B))

    def test_single_kit_cannot_leave_a_fitted_classifier(self):
        model = TeamAppearanceClassifier()
        with self.assertRaises(ValueError):
            model.fit([solid((20, 20, 220))] * 20)
        self.assertIsNone(model.centres)
        self.assertIsNone(model.radii)
        self.assertEqual(model.fit_report, {})
        self.assertEqual(model.predict(solid((20, 20, 220))).team_id, UNKNOWN)

    def test_third_kit_outliers_are_measured_and_fail_synthetic_policy(self):
        # A rare third colour must not disappear into a broad q90 radius.
        crops = red_blue() * 3 + [solid((220, 20, 220))] * 2
        model = TeamAppearanceClassifier()
        report = model.fit(crops)
        self.assertGreater(report["separation"]["outlier_fraction"], 0.0)
        self.profile["thresholds"]["max_outlier_fraction"] = 0.0
        result = assess_separation(report["separation"], self.calibration())
        self.assertEqual(result["decision"], "rejected")
        self.assertFalse(result["checks"]["max_outlier_fraction"]["passed"])

    def test_failed_refit_resets_state_before_input_checks(self):
        def broken_iterable():
            raise ValueError("synthetic iterable failure")
            yield None

        cases = (
            lambda m: m.fit([]),
            lambda m: m.fit(red_blue(), ["wrong length"]),
            lambda m: m.fit([solid((20, 20, 220))] * 20),
            lambda m: m.fit([None] * 20),
            lambda m: m.fit(broken_iterable()),
        )
        for refit in cases:
            model = TeamAppearanceClassifier()
            model.fit(red_blue())
            with self.assertRaises(ValueError):
                refit(model)
            self.assertIsNone(model.centres)
            self.assertIsNone(model.radii)
            self.assertIsNone(model.label_order)
            self.assertEqual(model._cluster_to_team, {})
            self.assertEqual(model.fit_report, {})
            prediction = model.predict(red_blue()[0])
            self.assertEqual(prediction.reason, "not_fitted")
            self.assertEqual(prediction.team_id, UNKNOWN)
            self.assertIsNone(prediction.margin)

    def test_alternate_trials_do_not_change_baseline_fit_fingerprint_or_anchor(self):
        crops = red_blue()
        model = TeamAppearanceClassifier(random_state=17)
        x = np.stack(sorted([appearance_feature(c) for c in crops],
                            key=lambda feature: tuple(np.round(feature, 9).tolist())))
        labels, centres = model._kmeans(x, model.max_iterations, 17)
        keys = sorted(tuple(np.round(centre, 8).tolist()) for centre in centres)
        fingerprint = hashlib.sha256(repr(keys).encode("utf-8")).hexdigest()[:16]
        with patch.object(model, "_kmeans", wraps=model._kmeans) as kmeans:
            report = model.fit(crops)
        self.assertEqual([call.args[2] for call in kmeans.call_args_list], [17, 18, 19])
        np.testing.assert_array_equal(model.centres, centres)
        expected_radii = [max(model.minimum_radius, float(np.quantile(
            np.linalg.norm(x[labels == k] - centres[k], axis=1), 0.9,
        )) * model.outlier_multiplier) for k in range(2)]
        np.testing.assert_array_equal(model.radii, np.array(expected_radii, dtype=np.float32))
        self.assertEqual(report["cluster_fingerprint"], fingerprint)
        self.assertEqual(report["anchor_cluster"], int(labels[0]))
        self.assertEqual(model.predict(crops[report["anchor_sample_index"]]).team_id, TEAM_A)

    def test_fit_metrics_are_identical_under_permutation_and_repeated_fit(self):
        rng = np.random.default_rng(21)
        crops = [np.clip(crop.astype(int) + rng.integers(-4, 5, crop.shape), 0, 255).astype(np.uint8)
                 for crop in red_blue()]
        model = TeamAppearanceClassifier()
        expected = model.fit(crops)["separation"]
        self.assertEqual(model.fit(crops)["separation"], expected)
        for _ in range(4):
            order = rng.permutation(len(crops))
            self.assertEqual(model.fit([crops[i] for i in order])["separation"], expected)

    def test_prediction_margin_is_other_minus_nearest_for_each_distance_path(self):
        cases = (
            ({}, solid((20, 20, 220)), "nearest_team_appearance_cluster"),
            ({}, solid((220, 20, 220)), "appearance_outlier"),
            ({"candidate_mode": "multi", "candidate_action": "reject"},
             solid((20, 220, 240)), "role_candidate_rejected"),
            ({"candidate_mode": "legacy", "candidate_action": "flag"},
             solid((20, 220, 240)), "appearance_outlier"),
        )
        for kwargs, crop, reason in cases:
            with self.subTest(kwargs=kwargs, reason=reason):
                model = TeamAppearanceClassifier(**kwargs)
                model.fit(red_blue())
                prediction = model.predict(crop)
                distances = np.sort(np.linalg.norm(model.centres - appearance_feature(crop), axis=1))
                self.assertEqual(prediction.reason, reason)
                self.assertEqual(prediction.distance, float(distances[0]))
                self.assertEqual(prediction.margin, float(distances[1]) - float(distances[0]))

    def test_margin_is_unavailable_on_early_rejections_and_positional_api_survives(self):
        self.assertIsNone(TeamPrediction(TEAM_A, 0.5, 0.1, "synthetic", None).margin)
        model = TeamAppearanceClassifier()
        self.assertIsNone(model.predict(red_blue()[0]).margin)
        model.fit(red_blue())
        self.assertIsNone(model.predict(None).margin)
        legacy = TeamAppearanceClassifier(detect_candidate_outliers=True)
        legacy.fit(red_blue())
        self.assertIsNone(legacy.predict(solid((20, 220, 240))).margin)
        masked = TeamAppearanceClassifier(use_grass_mask=True)
        masked.fit(red_blue())
        self.assertIsNone(masked.predict(solid((35, 140, 50))).margin)


if __name__ == "__main__":
    unittest.main()
