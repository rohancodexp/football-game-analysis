"""Descriptive two-kit geometry and externally reviewed separation policies.

Geometry is not accuracy or a probability. No real-video policy is bundled here.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from numbers import Real
from pathlib import Path, PureWindowsPath

import numpy as np


FEATURE_VERSION = "hsv_lab_v1"

# Schema bounds, not acceptance thresholds.
_THRESHOLD_BOUNDS = {
    "min_centroid_distance": (0.0, 2.0),
    "max_within_radius": (0.0, 2.0),
    "min_cluster_fraction": (0.0, 0.5),
    "min_mean_margin": (0.0, 2.0),
    "min_seed_agreement": (0.5, 1.0),
    "max_outlier_fraction": (0.0, 1.0),
    "min_crop_score": (0.0, 1.0),
    "min_crop_margin": (0.0, 2.0),
}
_SEPARATION_CHECKS = {
    "min_centroid_distance": ("centroid_distance", ">="),
    "max_within_radius": ("max_within_radius", "<="),
    "min_cluster_fraction": ("min_cluster_fraction", ">="),
    "min_mean_margin": ("mean_margin", ">="),
    "min_seed_agreement": ("min_seed_agreement", ">="),
    "max_outlier_fraction": ("outlier_fraction", "<="),
}
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def _finite_number(value) -> bool:
    if not isinstance(value, Real) or isinstance(value, (bool, np.bool_)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _validate_thresholds(thresholds) -> None:
    if not isinstance(thresholds, dict) or set(thresholds) != set(_THRESHOLD_BOUNDS):
        raise ValueError("thresholds must contain exactly the eight team policy keys")
    for key, (lower, upper) in _THRESHOLD_BOUNDS.items():
        value = thresholds[key]
        if not _finite_number(value) or not lower <= value <= upper:
            raise ValueError(f"{key} must be a finite number in [{lower}, {upper}]")


def _distance_summary(values: np.ndarray) -> dict:
    # Sorted reductions make reporting independent of sample order.
    values = np.sort(values)
    if not len(values):
        return {"rms": None, "quantiles": None}
    quantiles = np.quantile(values, [0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0])
    return {
        "rms": float(np.sqrt(np.mean(values * values))),
        "quantiles": dict(zip(
            ("q0", "q25", "q50", "q75", "q90", "q95", "q100"),
            map(float, quantiles),
        )),
    }


def measure_separation(x, labels, centres, radii, alternate_labels) -> dict:
    """Measure a fixed baseline partition, without changing its fit or labels.

    Labels index the two supplied centres. Outliers exceed their own fitted
    rejection radius; reported within radii instead use actual maximum distance.
    Initialization trials count the baseline plus alternate partitions. With no
    alternates, seed agreement is unavailable rather than presumed perfect.
    Memory is linear in sample count, never a sample-by-sample distance matrix.
    """
    x = np.asarray(x, dtype=np.float64)
    labels = np.asarray(labels)
    centres = np.asarray(centres, dtype=np.float64)
    radii = np.asarray(radii, dtype=np.float64)
    if x.ndim != 2 or not len(x) or not x.shape[1] or not np.isfinite(x).all():
        raise ValueError("x must be a nonempty finite feature matrix")
    if centres.shape != (2, x.shape[1]) or not np.isfinite(centres).all():
        raise ValueError("centres must contain two finite feature vectors")
    if radii.shape != (2,) or not np.isfinite(radii).all() or (radii < 0).any():
        raise ValueError("radii must contain two finite nonnegative values")
    if labels.shape != (len(x),) or not np.isin(labels, [0, 1]).all():
        raise ValueError("labels must index the two centres for every sample")
    labels = labels.astype(np.intp)

    distances = np.linalg.norm(x[:, None, :] - centres[None, :, :], axis=2)
    rows = np.arange(len(x))
    own = distances[rows, labels]
    other = distances[rows, 1 - labels]
    margins = other - own
    counts = np.bincount(labels, minlength=2)
    centroid_distance = float(np.linalg.norm(centres[0] - centres[1]))
    clusters = []
    max_radii = []
    for k in range(2):
        members = labels == k
        max_radii.append(float(own[members].max()) if counts[k] else None)
        clusters.append({
            "cluster": k,
            "count": int(counts[k]),
            "within_distances": _distance_summary(own[members]),
            "other_distances": _distance_summary(other[members]),
            "margins": _distance_summary(margins[members]),
        })

    agreements = []
    for alternate in alternate_labels:
        alternate = np.asarray(alternate)
        if alternate.shape != labels.shape or not np.isin(alternate, [0, 1]).all():
            raise ValueError("alternate labels must index both centres for every sample")
        matches = int(np.count_nonzero(alternate == labels))
        agreements.append(max(matches, len(x) - matches) / len(x))

    return {
        "feature_version": FEATURE_VERSION,
        "centroid_distance": centroid_distance,
        "min_cluster_fraction": float(counts.min() / len(x)),
        "max_within_radius": float(own.max()),
        "mean_margin": float(np.mean(np.sort(margins))),
        "min_seed_agreement": min(agreements) if agreements else None,
        "outlier_fraction": float(np.count_nonzero(own > radii[labels]) / len(x)),
        "clusters": clusters,
        "cluster_ball_gap": (
            centroid_distance - sum(max_radii) if all(counts) else None
        ),
        "initialization_trials": 1 + len(agreements),
    }


def assess_separation(metrics, calibration=None) -> dict:
    """Apply a loader-validated policy to geometry, not to club identity.

    Crop score/margin thresholds belong to the caller's prediction gate.
    """
    result = {
        "decision": "inconclusive",
        "reason": "uncalibrated_team_separation",
        "checks": {},
        "metrics": metrics,
        "calibration": dict(calibration) if isinstance(calibration, dict) else None,
    }
    if calibration is None:
        return result
    try:
        thresholds = calibration["thresholds"]
        _validate_thresholds(thresholds)
    except (KeyError, TypeError, ValueError):
        result["reason"] = "invalid_team_calibration"
        return result

    invalid = False
    for key, (metric, operator) in _SEPARATION_CHECKS.items():
        value = metrics.get(metric) if isinstance(metrics, dict) else None
        valid = _finite_number(value)
        invalid |= not valid
        passed = None
        if valid:
            passed = bool(value >= thresholds[key] if operator == ">=" else value <= thresholds[key])
        result["checks"][key] = {
            "metric": metric, "value": value, "operator": operator,
            "threshold": thresholds[key], "passed": passed,
        }
    if invalid:
        result["reason"] = "missing_or_invalid_separation_metrics"
        return result

    # Mathematical degeneracy guards, not empirically chosen policy thresholds.
    nondegenerate = metrics["centroid_distance"] > 0
    nonempty = metrics["min_cluster_fraction"] > 0
    result["checks"]["nondegenerate_centres"] = {"passed": bool(nondegenerate)}
    result["checks"]["nonempty_clusters"] = {"passed": bool(nonempty)}
    if not nondegenerate or not nonempty:
        result.update(decision="rejected", reason="degenerate_team_clusters")
    elif all(check["passed"] for check in result["checks"].values()):
        result.update(decision="accepted", reason="calibrated_team_separation")
    else:
        result.update(decision="rejected", reason="team_separation_threshold_failed")
    return result


def _unique_json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate calibration key: {key}")
        result[key] = value
    return result


def load_team_calibration(path, video_sha256) -> dict | None:
    """Load explicit review metadata and verify evidence integrity and holdout.

    Invalid profiles raise ValueError; None alone means no policy was supplied.
    The loader verifies declarations and file hashes, not the truth of a review.
    """
    if path is None:
        return None
    if not isinstance(video_sha256, str) or not _SHA256.fullmatch(video_sha256):
        raise ValueError("video_sha256 must be a lower-case SHA-256 digest")
    source = Path(path).resolve()
    try:
        raw = source.read_bytes()
        profile = json.loads(raw, object_pairs_hook=_unique_json_object)
    except (OSError, UnicodeError, ValueError) as exc:
        raise ValueError(f"Cannot read team calibration: {exc}") from exc
    if not isinstance(profile, dict):
        raise ValueError("team calibration must be a JSON object")
    if profile.get("kind") != "human_reviewed_validation":
        raise ValueError("calibration kind must be human_reviewed_validation")
    if profile.get("feature_version") != FEATURE_VERSION:
        raise ValueError(f"calibration feature_version must be {FEATURE_VERSION}")
    for key in ("reviewer", "rationale"):
        if not isinstance(profile.get(key), str) or not profile[key].strip():
            raise ValueError(f"calibration {key} must be nonempty")
    hashes = profile.get("validation_video_sha256")
    if not isinstance(hashes, list) or not hashes or any(
        not isinstance(digest, str) or not _SHA256.fullmatch(digest) for digest in hashes
    ):
        raise ValueError("validation_video_sha256 must be a nonempty array of lower-case SHA-256 digests")
    if video_sha256 in hashes:
        raise ValueError("target video must not be used for validation calibration")
    _validate_thresholds(profile.get("thresholds"))

    evidence_name = profile.get("evidence_report")
    if not isinstance(evidence_name, str) or not evidence_name.strip():
        raise ValueError("evidence_report must name a relative evidence file")
    evidence_path = Path(evidence_name)
    windows_path = PureWindowsPath(evidence_name)
    if evidence_path.is_absolute() or windows_path.drive or windows_path.root:
        raise ValueError("evidence_report must be relative to the calibration file")
    evidence_hash = profile.get("evidence_sha256")
    if not isinstance(evidence_hash, str) or not _SHA256.fullmatch(evidence_hash):
        raise ValueError("evidence_sha256 must be a lower-case SHA-256 digest")
    digest = hashlib.sha256()
    try:
        with (source.parent / evidence_path).open("rb") as evidence:
            for block in iter(lambda: evidence.read(1024 * 1024), b""):
                digest.update(block)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot read calibration evidence: {exc}") from exc
    if digest.hexdigest() != evidence_hash:
        raise ValueError("calibration evidence SHA-256 mismatch")
    profile["source_path"] = str(source)
    profile["profile_sha256"] = hashlib.sha256(raw).hexdigest()
    return profile
