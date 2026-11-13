"""
Unsupervised outfield-team appearance classifier.

This is an initial, interpretable baseline for broadcast football footage.
It assigns anonymous TEAM_A / TEAM_B labels; it does not identify real clubs.
Referees, goalkeepers, poor crops, and uncertain samples may remain UNKNOWN.

Dependencies: numpy, opencv-python
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

import cv2
import numpy as np


UNKNOWN = "UNKNOWN"
TEAM_A = "TEAM_A"
TEAM_B = "TEAM_B"

# Candidate detection modes: which signals are consulted.
CANDIDATE_MODE_OFF = "off"
CANDIDATE_MODE_LEGACY = "legacy"  # yellow HSV ratio only (v5 behaviour)
CANDIDATE_MODE_MULTI = "multi"    # yellow + off-palette + outfit contrast
CANDIDATE_MODES = (CANDIDATE_MODE_OFF, CANDIDATE_MODE_LEGACY, CANDIDATE_MODE_MULTI)

# Candidate consequence: what a candidate flag does to the team label.
CANDIDATE_ACTION_FLAG = "flag"    # annotate only; keep the team prediction
CANDIDATE_ACTION_REJECT = "reject"  # force UNKNOWN (legacy behaviour)
CANDIDATE_ACTIONS = (CANDIDATE_ACTION_FLAG, CANDIDATE_ACTION_REJECT)

# Roles. OUTFIT_MISMATCH is deliberately coarse: a kit unlike both teams can be
# a goalkeeper, a referee in a non-yellow strip, staff or a substitute. Separating
# them needs positional data or ground truth, which this baseline does not have.
ROLE_NONE = "NONE"
ROLE_REFEREE_CANDIDATE = "REFEREE_CANDIDATE"
ROLE_OUTFIT_MISMATCH_CANDIDATE = "OUTFIT_MISMATCH_CANDIDATE"

# Track-level exclusion needs persistent evidence, not one odd crop.
TRACK_CANDIDATE_SUPPORT_RATIO = 0.60
TRACK_CANDIDATE_MEAN_SCORE = 0.75

# Evidence ramps: raw signal value -> [0, 1] strength.
YELLOW_EVIDENCE_START = 0.15
YELLOW_EVIDENCE_FULL = 0.40
CONTRAST_EVIDENCE_START = 0.10
CONTRAST_EVIDENCE_FULL = 0.25
REFEREE_EVIDENCE_FLOOR = 0.60


def evidence_ramp(value: float, start: float, full: float) -> float:
    """Map a raw signal to evidence strength, 0 below ``start``, 1 above ``full``."""
    if full <= start:
        return 0.0
    return float(np.clip((value - start) / (full - start), 0.0, 1.0))


def extract_torso_crop(
    frame: np.ndarray,
    bbox_xyxy: Iterable[float],
    *,
    top: float = 0.16,
    bottom: float = 0.62,
    left: float = 0.18,
    right: float = 0.82,
    min_height: int = 18,
    min_width: int = 8,
) -> Optional[np.ndarray]:
    """Extract a conservative central upper-body crop from an XYXY box.

    The crop is intentionally narrower than the full detection to reduce
    grass, shorts, socks, and adjacent-player contamination. It is still
    only an approximation: bad person boxes produce bad torso crops.
    """
    if frame is None or frame.size == 0:
        return None

    height, width = frame.shape[:2]
    values = list(bbox_xyxy)
    if len(values) != 4 or not np.all(np.isfinite(values)):
        return None

    x1, y1, x2, y2 = map(float, values)
    x1 = int(np.clip(x1, 0, width))
    x2 = int(np.clip(x2, 0, width))
    y1 = int(np.clip(y1, 0, height))
    y2 = int(np.clip(y2, 0, height))

    box_w, box_h = x2 - x1, y2 - y1
    if box_w <= 0 or box_h <= 0:
        return None

    tx1 = x1 + int(box_w * left)
    tx2 = x1 + int(box_w * right)
    ty1 = y1 + int(box_h * top)
    ty2 = y1 + int(box_h * bottom)

    crop = frame[ty1:ty2, tx1:tx2]
    if crop.size == 0 or crop.shape[0] < min_height or crop.shape[1] < min_width:
        return None
    return crop


def extract_grass_mask(
    hsv_crop: np.ndarray,
    *,
    h_min: int = 35,
    h_max: int = 85,
    s_min: int = 35,
    v_min: int = 35,
) -> np.ndarray:
    """Return a binary mask where 255 indicates pitch grass pixels."""
    return cv2.inRange(hsv_crop, (h_min, s_min, v_min), (h_max, 255, 255))


def get_foreground_stats(
    bgr_crop: np.ndarray,
    *,
    min_foreground_ratio: float = 0.25,
    grass_params: Optional[dict] = None,
) -> dict:
    """Calculate grass removal percentage and foreground pixel ratio from a torso crop."""
    if bgr_crop is None or bgr_crop.size == 0 or bgr_crop.ndim != 3:
        return {
            "foreground_ratio": 0.0,
            "grass_ratio": 0.0,
            "removed_percentage": 0.0,
            "is_valid_foreground": False,
        }

    gp = grass_params or {}
    hsv = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2HSV)
    grass_mask = extract_grass_mask(
        hsv,
        h_min=gp.get("h_min", 35),
        h_max=gp.get("h_max", 85),
        s_min=gp.get("s_min", 35),
        v_min=gp.get("v_min", 35),
    )
    total_px = float(bgr_crop.shape[0] * bgr_crop.shape[1])
    grass_px = float(np.count_nonzero(grass_mask))
    fg_px = total_px - grass_px

    fg_ratio = fg_px / total_px if total_px > 0 else 0.0
    grass_ratio = grass_px / total_px if total_px > 0 else 0.0

    return {
        "foreground_ratio": round(fg_ratio, 4),
        "grass_ratio": round(grass_ratio, 4),
        "removed_percentage": round(grass_ratio * 100.0, 2),
        "is_valid_foreground": fg_ratio >= min_foreground_ratio,
    }


def detect_candidate_outlier(
    bgr_crop: np.ndarray,
    *,
    enabled: bool = False,
    yellow_ratio_threshold: float = 0.25,
) -> tuple[bool, str]:
    """Conservative check for candidate non-outfield appearance.

    Treats unusual kit colours as candidate outliers rather than confirmed roles.
    """
    if not enabled or bgr_crop is None or bgr_crop.size == 0 or bgr_crop.ndim != 3:
        return False, "outlier_check_disabled_or_invalid"

    hsv = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2HSV)
    # Fluorescent / bright yellow kit prior: Hue in [18, 38], Sat > 80, Val > 120
    mask_yellow = cv2.inRange(hsv, (18, 80, 120), (38, 255, 255))
    total_px = float(bgr_crop.shape[0] * bgr_crop.shape[1])
    yellow_ratio = float(np.count_nonzero(mask_yellow)) / total_px if total_px > 0 else 0.0

    if yellow_ratio >= yellow_ratio_threshold:
        return True, "candidate_non_outfield_yellow"

    return False, "not_candidate_outlier"


@dataclass
class RoleCandidate:
    """Weak, non-exclusive evidence that a crop is not an outfield team kit.

    A candidate is an annotation, not a verdict: it never assigns a team and,
    unless the caller chose the ``reject`` action, never removes one either.
    """

    is_candidate: bool = False
    role: str = ROLE_NONE
    score: float = 0.0
    signals: dict = field(default_factory=dict)
    reasons: list = field(default_factory=list)


def kit_colour_signals(crop: Optional[np.ndarray]) -> dict:
    """Context-free colour evidence for a torso crop.

    Returns the fluorescent-yellow ratio and an upper/lower outfit contrast
    value (Lab mean distance between the shirt band and the shorts band). Both
    are cheap appearance priors, not classifiers: neither proves a role.
    """
    empty = {"yellow_ratio": 0.0, "outfit_contrast": 0.0, "valid": False}

    if crop is None or getattr(crop, "size", 0) == 0 or crop.ndim != 3:
        return empty

    work = cv2.resize(crop, (48, 64), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(work, cv2.COLOR_BGR2HSV)
    total = float(work.shape[0] * work.shape[1])
    if total <= 0:
        return empty

    mask_yellow = cv2.inRange(hsv, (18, 80, 120), (38, 255, 255))
    yellow_ratio = float(np.count_nonzero(mask_yellow)) / total

    lab = cv2.cvtColor(work, cv2.COLOR_BGR2LAB)
    shirt = lab[: int(64 * 0.45), :, :].mean(axis=(0, 1))
    shorts = lab[int(64 * 0.60) :, :, :].mean(axis=(0, 1))
    # Normalised to roughly [0, 1] against the 0-255 Lab channel range.
    outfit_contrast = float(np.linalg.norm(shirt - shorts) / 255.0)

    return {
        "yellow_ratio": round(yellow_ratio, 4),
        "outfit_contrast": round(outfit_contrast, 4),
        "valid": True,
    }


def evaluate_role_candidate(
    crop: Optional[np.ndarray],
    *,
    mode: str = CANDIDATE_MODE_MULTI,
    nearest_distance: Optional[float] = None,
    nearest_radius: Optional[float] = None,
    yellow_evidence_start: float = YELLOW_EVIDENCE_START,
    yellow_evidence_full: float = YELLOW_EVIDENCE_FULL,
    contrast_evidence_start: float = CONTRAST_EVIDENCE_START,
    contrast_evidence_full: float = CONTRAST_EVIDENCE_FULL,
    candidate_threshold: float = 0.50,
    referee_evidence_floor: float = REFEREE_EVIDENCE_FLOOR,
) -> RoleCandidate:
    """Combine weak kit signals into a scored candidate.

    ``off_palette_evidence`` needs cluster context and is only available when
    ``nearest_distance``/``nearest_radius`` are supplied; without it the
    remaining signals are too weak to call anything on their own.
    """
    if mode == CANDIDATE_MODE_OFF:
        return RoleCandidate(reasons=["candidate_detection_disabled"])

    signals = kit_colour_signals(crop)
    if not signals["valid"]:
        return RoleCandidate(reasons=["candidate_signals_unavailable"])

    referee = evidence_ramp(
        signals["yellow_ratio"], yellow_evidence_start, yellow_evidence_full
    )
    contrast = evidence_ramp(
        signals["outfit_contrast"], contrast_evidence_start, contrast_evidence_full
    )

    off_palette = 0.0
    if (
        mode == CANDIDATE_MODE_MULTI
        and nearest_distance is not None
        and nearest_radius is not None
        and nearest_radius > 0
    ):
        off_palette = float(
            np.clip((nearest_distance - nearest_radius) / max(nearest_radius, 1e-8), 0.0, 1.0)
        )

    if mode == CANDIDATE_MODE_LEGACY:
        # v5 parity: the yellow prior alone, at the same decision boundary.
        score = referee
    else:
        mismatch = 0.5 * off_palette + 0.5 * contrast
        score = max(referee, mismatch)

    signals.update(
        {
            "referee_evidence": round(referee, 4),
            "contrast_evidence": round(contrast, 4),
            "off_palette_evidence": round(off_palette, 4),
            "has_cluster_context": bool(
                mode == CANDIDATE_MODE_MULTI and nearest_distance is not None
            ),
        }
    )

    reasons: list[str] = []
    role = ROLE_NONE
    if score >= candidate_threshold:
        if referee >= referee_evidence_floor:
            role = ROLE_REFEREE_CANDIDATE
            reasons.append(f"fluorescent_yellow_ratio={signals['yellow_ratio']}")
        elif mode == CANDIDATE_MODE_LEGACY:
            role = ROLE_REFEREE_CANDIDATE
            reasons.append(f"legacy_yellow_ratio={signals['yellow_ratio']}")
        else:
            role = ROLE_OUTFIT_MISMATCH_CANDIDATE
            if off_palette >= 0.5:
                reasons.append(
                    f"kit_outside_both_team_clusters(distance={nearest_distance:.3f}>"
                    f"radius={nearest_radius:.3f})"
                )
            if contrast >= 0.5:
                reasons.append(f"shirt_shorts_contrast={signals['outfit_contrast']}")
        if not reasons:
            reasons.append(f"candidate_score={round(score, 4)}")

    return RoleCandidate(
        is_candidate=score >= candidate_threshold,
        role=role,
        score=round(float(score), 4),
        signals=signals,
        reasons=reasons,
    )


def appearance_feature(
    bgr_crop: np.ndarray,
    *,
    use_grass_mask: bool = False,
    min_foreground_ratio: float = 0.25,
    grass_params: Optional[dict] = None,
) -> Optional[np.ndarray]:
    """Create a compact colour feature from a BGR torso crop.

    Uses HSV histograms and Lab statistics.
    When use_grass_mask=True, pitch grass pixels are masked out so histograms
    capture kit fabric rather than pitch green.
    """
    if bgr_crop is None or bgr_crop.size == 0:
        return None
    if bgr_crop.ndim != 3 or bgr_crop.shape[2] != 3:
        return None
    if bgr_crop.shape[0] < 8 or bgr_crop.shape[1] < 6:
        return None

    # Resize consistently, reducing sensitivity to original crop dimensions.
    crop = cv2.resize(bgr_crop, (48, 64), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)

    mask: Optional[np.ndarray] = None
    if use_grass_mask:
        gp = grass_params or {}
        grass_mask = extract_grass_mask(
            hsv,
            h_min=gp.get("h_min", 35),
            h_max=gp.get("h_max", 85),
            s_min=gp.get("s_min", 35),
            v_min=gp.get("v_min", 35),
        )
        fg_mask = cv2.bitwise_not(grass_mask)
        total_px = 48 * 64
        fg_px = np.count_nonzero(fg_mask)
        if (fg_px / float(total_px)) < min_foreground_ratio:
            return None
        mask = fg_mask

    h_hist = cv2.calcHist([hsv], [0], mask, [18], [0, 180]).flatten()
    s_hist = cv2.calcHist([hsv], [1], mask, [8], [0, 256]).flatten()
    v_hist = cv2.calcHist([hsv], [2], mask, [4], [0, 256]).flatten()

    def normalise_hist(hist: np.ndarray) -> np.ndarray:
        total = float(hist.sum())
        return hist / total if total > 0 else hist

    if mask is not None:
        valid_px = lab[mask > 0]
        if len(valid_px) > 0:
            lab_mean = valid_px.mean(axis=0) / 255.0
            lab_std = valid_px.std(axis=0) / 255.0
        else:
            lab_mean = np.zeros(3, dtype=np.float32)
            lab_std = np.zeros(3, dtype=np.float32)
    else:
        lab_mean = lab.mean(axis=(0, 1)) / 255.0
        lab_std = lab.std(axis=(0, 1)) / 255.0

    colour_stats = np.concatenate([lab_mean, lab_std])
    feature = np.concatenate(
        [
            1.5 * normalise_hist(h_hist),
            1.5 * normalise_hist(s_hist),
            0.75 * normalise_hist(v_hist),
            0.8 * colour_stats,
        ]
    ).astype(np.float32)

    norm = float(np.linalg.norm(feature))
    if not np.isfinite(norm) or norm < 1e-8:
        return None
    return feature / norm


@dataclass
class TeamPrediction:
    team_id: str = UNKNOWN
    confidence: float = 0.0
    distance: Optional[float] = None
    reason: str = "not_fitted"
    candidate: Optional[RoleCandidate] = None


class TeamAppearanceClassifier:
    """Two-cluster appearance classifier with conservative outlier rejection.

    Fit with a representative collection of torso crops from wide-view
    footage. Multiple samples from the same player are allowed, but avoid
    letting one long track dominate the training set.

    Anonymous labels are reproducible for a given multiset of crops: the
    features are canonically ordered before clustering and ``TEAM_A`` is the
    cluster containing a documented anchor crop rather than a coordinate
    accident. The anchor exists only inside one fit, so ``TEAM_A`` in one video
    has no relationship to ``TEAM_A`` in another and never names a club.

    ``candidate_mode`` selects the evidence used to call a crop a non-outfield
    candidate: ``off`` disables it, ``legacy`` is the v5 yellow-HSV-only rule,
    ``multi`` adds off-palette and shirt/shorts contrast evidence.
    ``candidate_action`` is orthogonal and decides the consequence: ``flag``
    annotates and keeps the team prediction, ``reject`` forces UNKNOWN.
    ``detect_candidate_outliers=True`` is the v5 shortcut for the pair
    (``legacy``, ``reject``) and is kept working unchanged.

    A candidate can never *grant* a team label; it can only annotate one, or,
    under ``reject``, remove one. Both knobs act on the single crop passed to
    ``predict``: this class has no notion of a track, so any escalation from one
    odd crop to a track-level exclusion belongs to the caller.
    """

    def __init__(
        self,
        *,
        min_samples: int = 12,
        min_cluster_samples: int = 4,
        max_iterations: int = 100,
        random_state: int = 17,
        outlier_multiplier: float = 1.35,
        minimum_radius: float = 0.12,
        use_grass_mask: bool = False,
        min_foreground_ratio: float = 0.25,
        detect_candidate_outliers: bool = False,
        candidate_mode: Optional[str] = None,
        candidate_action: Optional[str] = None,
        candidate_threshold: float = 0.50,
    ):
        if candidate_mode is None:
            candidate_mode = (
                CANDIDATE_MODE_LEGACY if detect_candidate_outliers else CANDIDATE_MODE_OFF
            )
        if candidate_action is None:
            # v5 parity: the legacy flag always removed the crop. New modes
            # default to annotating, because a candidate is evidence, not a verdict.
            candidate_action = (
                CANDIDATE_ACTION_REJECT
                if detect_candidate_outliers
                else CANDIDATE_ACTION_FLAG
            )
        if candidate_mode not in CANDIDATE_MODES:
            raise ValueError(f"candidate_mode must be one of {CANDIDATE_MODES}")
        if candidate_action not in CANDIDATE_ACTIONS:
            raise ValueError(f"candidate_action must be one of {CANDIDATE_ACTIONS}")

        self.min_samples = min_samples
        self.min_cluster_samples = min_cluster_samples
        self.max_iterations = max_iterations
        self.random_state = random_state
        self.outlier_multiplier = outlier_multiplier
        self.minimum_radius = minimum_radius
        self.use_grass_mask = use_grass_mask
        self.min_foreground_ratio = min_foreground_ratio
        self.candidate_mode = candidate_mode
        self.candidate_action = candidate_action
        self.candidate_threshold = candidate_threshold
        # Kept in sync so existing readers of this attribute still work.
        self.detect_candidate_outliers = detect_candidate_outliers
        self.centres: Optional[np.ndarray] = None
        self.radii: Optional[np.ndarray] = None
        self.label_order: Optional[tuple[str, str]] = None
        self.fit_report: dict = {}

    @staticmethod
    def _kmeans(x: np.ndarray, iterations: int, seed: int):
        rng = np.random.default_rng(seed)
        # Farthest-point second initialisation avoids choosing near-duplicates.
        first = int(rng.integers(0, len(x)))
        d2 = np.sum((x - x[first]) ** 2, axis=1)
        second = int(np.argmax(d2))
        centres = np.stack([x[first], x[second]]).astype(np.float32)

        labels = np.zeros(len(x), dtype=np.int32)
        for _ in range(iterations):
            distances = np.linalg.norm(x[:, None, :] - centres[None, :, :], axis=2)
            new_labels = distances.argmin(axis=1)
            new_centres = centres.copy()
            for k in range(2):
                members = x[new_labels == k]
                if len(members):
                    new_centres[k] = members.mean(axis=0)
                else:
                    new_centres[k] = x[int(rng.integers(0, len(x)))]
            if np.array_equal(new_labels, labels) and np.allclose(new_centres, centres):
                labels, centres = new_labels, new_centres
                break
            labels, centres = new_labels, new_centres
        return labels, centres

    def _fit_gate_rejects(self, crop: np.ndarray) -> bool:
        """Keep strong-evidence non-team kits out of the two-cluster fit.

        This is model protection, not a judgement about the person: the crop is
        still predicted, counted and reported. It exists because a k=2 split
        with referees in the training set converges on {everyone, referees}
        instead of {team A, team B} -- measured at 392/15 on the v4 crops,
        against 244/148 with the gate. Off-palette and shirt/shorts contrast
        are deliberately NOT used here: contrast alone removed 24 fitting crops
        on the v4 set, mostly legitimate kits (white shirt, black shorts), so
        deleting training data on that signal would be exclusion without
        sufficient evidence.
        """
        if self.candidate_mode == CANDIDATE_MODE_OFF:
            return False
        if self.candidate_mode == CANDIDATE_MODE_LEGACY:
            return detect_candidate_outlier(crop, enabled=True)[0]
        signals = kit_colour_signals(crop)
        if not signals["valid"]:
            return False
        return (
            evidence_ramp(signals["yellow_ratio"], YELLOW_EVIDENCE_START, YELLOW_EVIDENCE_FULL)
            >= REFEREE_EVIDENCE_FLOOR
        )

    def fit(self, torso_crops: Iterable[np.ndarray], crop_ids: Optional[Sequence] = None) -> dict:
        crops = list(torso_crops)
        ids = list(crop_ids) if crop_ids is not None else None
        if ids is not None and len(ids) != len(crops):
            raise ValueError("crop_ids must have the same length as torso_crops")

        features: list[np.ndarray] = []
        source_index: list[int] = []
        gated_out = 0
        for index, crop in enumerate(crops):
            if self._fit_gate_rejects(crop):
                gated_out += 1
                continue
            feature = appearance_feature(
                crop,
                use_grass_mask=self.use_grass_mask,
                min_foreground_ratio=self.min_foreground_ratio,
            )
            if feature is not None:
                features.append(feature)
                source_index.append(index)

        if len(features) < self.min_samples:
            raise ValueError(
                f"Need at least {self.min_samples} valid torso crops; got {len(features)}."
            )

        # Canonical input order: cluster results depend on the multiset of crops,
        # not the order they were collected in. Without this, the RNG picks a
        # different initial sample (and an empty-cluster reseed can pick a
        # different one) on every reshuffle, and TEAM_A/TEAM_B can flip.
        keys = [tuple(np.round(feature, 9).tolist()) for feature in features]
        positions = sorted(range(len(features)), key=keys.__getitem__)
        x = np.stack([features[j] for j in positions]).astype(np.float32)

        labels, centres = self._kmeans(x, self.max_iterations, self.random_state)
        counts = np.bincount(labels, minlength=2)

        if int(counts.min()) < self.min_cluster_samples:
            raise ValueError(
                "The two-team split is not sufficiently supported: "
                f"cluster sizes are {counts.tolist()}."
            )

        # Deterministic anonymous labels. TEAM_A is the cluster containing the
        # anchor crop -- the lexicographically smallest feature vector of the
        # canonical ordering. It is a member of the data, not a float-formatting
        # accident of the fitted centre, and it carries no club meaning: the
        # same rule applied to a different video produces an unrelated TEAM_A.
        anchor_row = 0
        anchor_cluster = int(labels[anchor_row])
        other_cluster = 1 - anchor_cluster
        mapping = {anchor_cluster: TEAM_A, other_cluster: TEAM_B}
        anchor_sample = source_index[positions[anchor_row]]

        radii = np.zeros(2, dtype=np.float32)
        within = np.zeros(2, dtype=np.float32)
        for k in range(2):
            distances = np.linalg.norm(x[labels == k] - centres[k], axis=1)
            q90 = float(np.quantile(distances, 0.90))
            radii[k] = max(self.minimum_radius, q90 * self.outlier_multiplier)
            within[k] = float(distances.mean())

        centre_keys = sorted(tuple(np.round(centres[k], 8).tolist()) for k in range(2))
        fingerprint = hashlib.sha256(repr(centre_keys).encode("utf-8")).hexdigest()[:16]

        self.centres = centres
        self.radii = radii
        self.label_order = (TEAM_A, TEAM_B)
        self._cluster_to_team = mapping
        self.fit_report = {
            "valid_samples": int(len(x)),
            "fitting_set_candidate_gated": int(gated_out),
            "cluster_sizes": {
                mapping[k]: int(counts[k]) for k in range(2)
            },
            "mean_within_cluster_distance": {
                mapping[k]: float(within[k]) for k in range(2)
            },
            "outlier_radii": {
                mapping[k]: float(radii[k]) for k in range(2)
            },
            "labeling_rule": "anchor_crop",
            "anchor_cluster": anchor_cluster,
            "anchor_sample_index": int(anchor_sample),
            "anchor_crop_id": ids[anchor_sample] if ids is not None else None,
            "cluster_fingerprint": fingerprint,
            "outlier_multiplier": self.outlier_multiplier,
            "use_grass_mask": self.use_grass_mask,
            "min_foreground_ratio": self.min_foreground_ratio,
            "detect_candidate_outliers": self.detect_candidate_outliers,
            "candidate_mode": self.candidate_mode,
            "candidate_action": self.candidate_action,
            "candidate_threshold": self.candidate_threshold,
            "warning": (
                "Unsupervised visual clusters only. TEAM_A/TEAM_B are anonymous "
                "labels, not verified club identities, and are only comparable "
                "within this fit. Referee/goalkeeper exclusion is not guaranteed "
                "by this baseline."
            ),
        }
        return self.fit_report

    def _context_free_candidate(self, crop: np.ndarray) -> Optional[RoleCandidate]:
        """Candidate evidence for a crop that never reached the cluster step.

        Off-palette needs a fitted cluster, so such crops are scored on the
        context-free signals alone and are marked as lacking cluster context.
        """
        if self.candidate_mode != CANDIDATE_MODE_MULTI:
            return None
        return evaluate_role_candidate(
            crop,
            mode=CANDIDATE_MODE_MULTI,
            candidate_threshold=self.candidate_threshold,
        )

    def predict(self, torso_crop: np.ndarray) -> TeamPrediction:
        if self.centres is None or self.radii is None:
            return TeamPrediction(reason="not_fitted")

        # Legacy v5 path: the yellow prior runs before every other check and,
        # under the reject action, replaces the team label with UNKNOWN.
        legacy_candidate: Optional[RoleCandidate] = None
        if self.candidate_mode == CANDIDATE_MODE_LEGACY:
            is_cand, cand_reason = detect_candidate_outlier(torso_crop, enabled=True)
            signals = kit_colour_signals(torso_crop)
            legacy_candidate = RoleCandidate(
                is_candidate=is_cand,
                role=ROLE_REFEREE_CANDIDATE if is_cand else ROLE_NONE,
                score=round(float(is_cand), 4),
                signals=signals,
                reasons=[cand_reason],
            )
            if is_cand and self.candidate_action == CANDIDATE_ACTION_REJECT:
                return TeamPrediction(
                    team_id=UNKNOWN,
                    confidence=0.0,
                    reason=cand_reason,
                    candidate=legacy_candidate,
                )

        if self.use_grass_mask:
            fg_stats = get_foreground_stats(
                torso_crop,
                min_foreground_ratio=self.min_foreground_ratio,
            )
            if not fg_stats["is_valid_foreground"]:
                return TeamPrediction(
                    team_id=UNKNOWN,
                    confidence=0.0,
                    reason="insufficient_foreground",
                    candidate=legacy_candidate
                    or self._context_free_candidate(torso_crop),
                )

        feature = appearance_feature(
            torso_crop,
            use_grass_mask=self.use_grass_mask,
            min_foreground_ratio=self.min_foreground_ratio,
        )
        if feature is None:
            return TeamPrediction(
                reason="invalid_or_too_small_crop",
                candidate=legacy_candidate or self._context_free_candidate(torso_crop),
            )

        distances = np.linalg.norm(self.centres - feature[None, :], axis=1)
        k = int(np.argmin(distances))
        distance = float(distances[k])
        radius = float(self.radii[k])

        candidate = legacy_candidate
        if self.candidate_mode == CANDIDATE_MODE_MULTI:
            candidate = evaluate_role_candidate(
                torso_crop,
                mode=CANDIDATE_MODE_MULTI,
                nearest_distance=distance,
                nearest_radius=radius,
                candidate_threshold=self.candidate_threshold,
            )
            if candidate.is_candidate and self.candidate_action == CANDIDATE_ACTION_REJECT:
                return TeamPrediction(
                    team_id=UNKNOWN,
                    confidence=0.0,
                    distance=distance,
                    reason="role_candidate_rejected",
                    candidate=candidate,
                )

        if distance > radius:
            return TeamPrediction(
                team_id=UNKNOWN,
                confidence=0.0,
                distance=distance,
                reason="appearance_outlier",
                candidate=candidate,
            )

        confidence = float(np.clip(1.0 - distance / max(radius, 1e-8), 0.0, 1.0))
        return TeamPrediction(
            team_id=self._cluster_to_team[k],
            confidence=confidence,
            distance=distance,
            reason="nearest_team_appearance_cluster",
            candidate=candidate,
        )
