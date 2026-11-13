"""
Shared, configurable quality-filtering policy for football player detections and torso crops.

This module unifies the filtering criteria across both review_team_crops_v2.py
and run_team_clustering.py so that crop generation and clustering evaluation
apply identical quality policies with transparent, non-overlapping rejection accounting.
"""

from __future__ import annotations

import argparse
import math
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Optional, Sequence, Union

import numpy as np


# Standardized rejection reason identifiers
REASON_INVALID_METADATA = "invalid_detection_record"
REASON_INVALID_BOX = "invalid_box"
REASON_LOW_CONFIDENCE = "low_detector_confidence"
REASON_BOX_TOO_NARROW = "box_too_narrow"
REASON_BOX_TOO_WIDE = "box_too_wide"
REASON_BOX_TOO_SHORT = "box_too_short"
REASON_BOX_TOO_TALL = "box_too_tall"
REASON_AREA_TOO_LARGE = "box_area_too_large"
REASON_ASPECT_RATIO_TOO_HIGH = "aspect_ratio_too_high"
REASON_ASPECT_RATIO_TOO_LOW = "aspect_ratio_too_low"

REASON_CROP_TOO_SHORT = "torso_crop_too_short"
REASON_CROP_TOO_NARROW = "torso_crop_too_narrow"
REASON_CROP_INVALID = "torso_crop_invalid"


@dataclass(frozen=True)
class CropFilterPolicy:
    """Configurable detection and crop quality policy.

    Defaults are calibrated for 720p/1080p broadcast football footage:
    - min_confidence (0.45): rejects detector clutter and background noise.
    - min_box_height (50 px): ensures torso region has adequate pixels for histograms.
    - max_box_height (160 px): rejects extreme foreground close-ups (sideline, ref, replay).
    - min_box_width (14 px): rejects heavily occluded vertical slivers.
    - max_box_area_fraction (0.025 = 2.5%): keeps tactical outfield players while filtering close-ups.
    - max_aspect_ratio (0.90 W/H): rejects merged multi-player boxes, fallen players, and clutter.
    - min_crop_height (14 px) & min_crop_width (8 px): minimum torso crop dimensions.
    """

    min_confidence: float = 0.45
    min_box_height: int = 50
    max_box_height: Optional[int] = 160
    min_box_width: int = 14
    max_box_width: Optional[int] = None
    max_box_area_fraction: float = 0.025
    max_aspect_ratio: float = 0.90
    min_aspect_ratio: Optional[float] = None
    min_crop_height: int = 14
    min_crop_width: int = 8

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence must be between 0.0 and 1.0")
        if self.min_box_height <= 0:
            raise ValueError("min_box_height must be positive")
        if self.max_box_height is not None and self.max_box_height < self.min_box_height:
            raise ValueError("max_box_height cannot be less than min_box_height")
        if self.min_box_width <= 0:
            raise ValueError("min_box_width must be positive")
        if self.max_box_width is not None and self.max_box_width < self.min_box_width:
            raise ValueError("max_box_width cannot be less than min_box_width")
        if not 0.0 < self.max_box_area_fraction <= 1.0:
            raise ValueError("max_box_area_fraction must be in (0.0, 1.0]")
        if self.max_aspect_ratio <= 0.0:
            raise ValueError("max_aspect_ratio must be positive")
        if self.min_aspect_ratio is not None and self.min_aspect_ratio <= 0.0:
            raise ValueError("min_aspect_ratio must be positive")
        if self.min_crop_height <= 0 or self.min_crop_width <= 0:
            raise ValueError("min_crop_height and min_crop_width must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CropFilterPolicy:
        valid_keys = {
            "min_confidence",
            "min_box_height",
            "max_box_height",
            "min_box_width",
            "max_box_width",
            "max_box_area_fraction",
            "max_aspect_ratio",
            "min_aspect_ratio",
            "min_crop_height",
            "min_crop_width",
        }
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)

    def evaluate_box(
        self,
        bbox_xyxy: Any,
        detector_confidence: Any,
        frame_width: int,
        frame_height: int,
    ) -> FilterResult:
        """Evaluate a raw detection bounding box against this quality policy."""
        reasons: list[str] = []
        metrics: dict[str, float] = {}

        if frame_width <= 0 or frame_height <= 0:
            return FilterResult(
                is_valid=False,
                reasons=[REASON_INVALID_METADATA],
                metrics={"frame_width": float(frame_width), "frame_height": float(frame_height)},
            )

        frame_area = float(frame_width * frame_height)

        try:
            coords = list(bbox_xyxy)
            if len(coords) != 4:
                return FilterResult(is_valid=False, reasons=[REASON_INVALID_BOX], metrics={})
            x1, y1, x2, y2 = map(float, coords)
            if not all(math.isfinite(c) for c in (x1, y1, x2, y2)):
                return FilterResult(is_valid=False, reasons=[REASON_INVALID_BOX], metrics={})
        except (TypeError, ValueError):
            return FilterResult(is_valid=False, reasons=[REASON_INVALID_BOX], metrics={})

        try:
            conf = float(detector_confidence)
            if not math.isfinite(conf):
                return FilterResult(is_valid=False, reasons=[REASON_INVALID_METADATA], metrics={})
        except (TypeError, ValueError):
            return FilterResult(is_valid=False, reasons=[REASON_INVALID_METADATA], metrics={})

        bw = x2 - x1
        bh = y2 - y1

        if bw <= 0.0 or bh <= 0.0:
            return FilterResult(
                is_valid=False,
                reasons=[REASON_INVALID_BOX],
                metrics={"box_width": bw, "box_height": bh, "detector_confidence": conf},
            )

        area_fraction = (bw * bh) / frame_area
        aspect_ratio = bw / bh

        metrics = {
            "box_width": round(bw, 2),
            "box_height": round(bh, 2),
            "area_fraction": round(area_fraction, 6),
            "aspect_ratio": round(aspect_ratio, 4),
            "detector_confidence": round(conf, 4),
        }

        if conf < self.min_confidence:
            reasons.append(REASON_LOW_CONFIDENCE)

        if bw < self.min_box_width:
            reasons.append(REASON_BOX_TOO_NARROW)
        elif self.max_box_width is not None and bw > self.max_box_width:
            reasons.append(REASON_BOX_TOO_WIDE)

        if bh < self.min_box_height:
            reasons.append(REASON_BOX_TOO_SHORT)
        elif self.max_box_height is not None and bh > self.max_box_height:
            reasons.append(REASON_BOX_TOO_TALL)

        if area_fraction > self.max_box_area_fraction:
            reasons.append(REASON_AREA_TOO_LARGE)

        if aspect_ratio > self.max_aspect_ratio:
            reasons.append(REASON_ASPECT_RATIO_TOO_HIGH)
        elif self.min_aspect_ratio is not None and aspect_ratio < self.min_aspect_ratio:
            reasons.append(REASON_ASPECT_RATIO_TOO_LOW)

        return FilterResult(
            is_valid=len(reasons) == 0,
            reasons=reasons,
            metrics=metrics,
        )

    def evaluate_torso_crop(
        self,
        crop: Optional[np.ndarray],
    ) -> FilterResult:
        """Evaluate an extracted torso crop array."""
        if crop is None or crop.size == 0 or crop.ndim != 3:
            return FilterResult(
                is_valid=False,
                reasons=[REASON_CROP_INVALID],
                metrics={},
            )

        h, w = crop.shape[:2]
        reasons: list[str] = []
        metrics = {"crop_width": w, "crop_height": h}

        if h < self.min_crop_height:
            reasons.append(REASON_CROP_TOO_SHORT)
        if w < self.min_crop_width:
            reasons.append(REASON_CROP_TOO_NARROW)

        return FilterResult(
            is_valid=len(reasons) == 0,
            reasons=reasons,
            metrics=metrics,
        )

    def evaluate_sample_record(
        self,
        sample: dict[str, Any],
        frame_width: int,
        frame_height: int,
    ) -> FilterResult:
        """Evaluate a crop index sample record (containing box_width, box_height, etc.)."""
        try:
            bw = float(sample["box_width"])
            bh = float(sample["box_height"])
            conf = float(sample["detector_confidence"])
        except (KeyError, TypeError, ValueError):
            return FilterResult(is_valid=False, reasons=[REASON_INVALID_METADATA], metrics={})

        frame_area = float(frame_width * frame_height) if frame_width > 0 and frame_height > 0 else 0.0
        if frame_area <= 0.0 or bw <= 0.0 or bh <= 0.0:
            return FilterResult(is_valid=False, reasons=[REASON_INVALID_BOX], metrics={})

        area_fraction = (bw * bh) / frame_area
        aspect_ratio = bw / bh

        reasons: list[str] = []
        metrics = {
            "box_width": round(bw, 2),
            "box_height": round(bh, 2),
            "area_fraction": round(area_fraction, 6),
            "aspect_ratio": round(aspect_ratio, 4),
            "detector_confidence": round(conf, 4),
        }

        if conf < self.min_confidence:
            reasons.append(REASON_LOW_CONFIDENCE)

        if bw < self.min_box_width:
            reasons.append(REASON_BOX_TOO_NARROW)
        elif self.max_box_width is not None and bw > self.max_box_width:
            reasons.append(REASON_BOX_TOO_WIDE)

        if bh < self.min_box_height:
            reasons.append(REASON_BOX_TOO_SHORT)
        elif self.max_box_height is not None and bh > self.max_box_height:
            reasons.append(REASON_BOX_TOO_TALL)

        if area_fraction > self.max_box_area_fraction:
            reasons.append(REASON_AREA_TOO_LARGE)

        if aspect_ratio > self.max_aspect_ratio:
            reasons.append(REASON_ASPECT_RATIO_TOO_HIGH)
        elif self.min_aspect_ratio is not None and aspect_ratio < self.min_aspect_ratio:
            reasons.append(REASON_ASPECT_RATIO_TOO_LOW)

        # Also verify crop dimensions if present in sample
        if "crop_width" in sample and "crop_height" in sample:
            cw = int(sample["crop_width"])
            ch = int(sample["crop_height"])
            metrics["crop_width"] = cw
            metrics["crop_height"] = ch
            if ch < self.min_crop_height:
                reasons.append(REASON_CROP_TOO_SHORT)
            if cw < self.min_crop_width:
                reasons.append(REASON_CROP_TOO_NARROW)

        return FilterResult(
            is_valid=len(reasons) == 0,
            reasons=reasons,
            metrics=metrics,
        )


@dataclass
class FilterResult:
    """Detailed result of evaluating a detection or crop against a filter policy."""

    is_valid: bool
    reasons: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class FilterStats:
    """Detailed rejection and acceptance accounting distinguishing unique vs. overlapping counts."""

    total_evaluated: int = 0
    unique_accepted: int = 0
    unique_rejected: int = 0
    rejection_reasons_overlapping: Counter = field(default_factory=Counter)
    primary_rejection_reasons: Counter = field(default_factory=Counter)
    rejection_combinations: Counter = field(default_factory=Counter)

    def record(self, result: FilterResult) -> None:
        self.total_evaluated += 1
        if result.is_valid:
            self.unique_accepted += 1
        else:
            self.unique_rejected += 1
            if result.reasons:
                for r in result.reasons:
                    self.rejection_reasons_overlapping[r] += 1
                self.primary_rejection_reasons[result.reasons[0]] += 1
                combo_key = "+".join(sorted(result.reasons))
                self.rejection_combinations[combo_key] += 1
            else:
                self.rejection_reasons_overlapping["unknown_rejection"] += 1
                self.primary_rejection_reasons["unknown_rejection"] += 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_evaluated": self.total_evaluated,
            "unique_accepted": self.unique_accepted,
            "unique_rejected": self.unique_rejected,
            "rejection_reasons_overlapping": dict(self.rejection_reasons_overlapping),
            "primary_rejection_reasons": dict(self.primary_rejection_reasons),
            "rejection_combinations": dict(self.rejection_combinations),
        }


def add_filter_arguments(
    parser: argparse.ArgumentParser,
    *,
    prefix: str = "",
    default_policy: Optional[CropFilterPolicy] = None,
) -> None:
    """Add unified filter arguments to an argparse parser."""
    p = default_policy or CropFilterPolicy()
    pref = f"--{prefix}-" if prefix else "--"

    parser.add_argument(
        f"{pref}min-confidence",
        type=float,
        default=p.min_confidence,
        help=f"Minimum detector confidence threshold (default: {p.min_confidence})",
    )
    parser.add_argument(
        f"{pref}min-box-height",
        type=int,
        default=p.min_box_height,
        help=f"Minimum detection box height in pixels (default: {p.min_box_height})",
    )
    parser.add_argument(
        f"{pref}max-box-height",
        type=int,
        default=p.max_box_height or 0,
        help=f"Maximum detection box height in pixels (0 for unlimited; default: {p.max_box_height})",
    )
    parser.add_argument(
        f"{pref}min-box-width",
        type=int,
        default=p.min_box_width,
        help=f"Minimum detection box width in pixels (default: {p.min_box_width})",
    )
    parser.add_argument(
        f"{pref}max-box-area-fraction",
        type=float,
        default=p.max_box_area_fraction,
        help=f"Maximum box area fraction relative to frame area (default: {p.max_box_area_fraction})",
    )
    parser.add_argument(
        f"{pref}max-aspect-ratio",
        type=float,
        default=p.max_aspect_ratio,
        help=f"Maximum aspect ratio width/height (default: {p.max_aspect_ratio})",
    )
    parser.add_argument(
        f"{pref}min-crop-height",
        type=int,
        default=p.min_crop_height,
        help=f"Minimum extracted torso crop height in pixels (default: {p.min_crop_height})",
    )
    parser.add_argument(
        f"{pref}min-crop-width",
        type=int,
        default=p.min_crop_width,
        help=f"Minimum extracted torso crop width in pixels (default: {p.min_crop_width})",
    )


def policy_from_args(
    args: argparse.Namespace,
    *,
    prefix: str = "",
) -> CropFilterPolicy:
    """Extract a CropFilterPolicy from parsed argparse namespace."""
    p = f"{prefix}_" if prefix else ""
    max_h = getattr(args, f"{p}max_box_height", 0)
    max_box_height = int(max_h) if max_h and int(max_h) > 0 else None

    return CropFilterPolicy(
        min_confidence=float(getattr(args, f"{p}min_confidence")),
        min_box_height=int(getattr(args, f"{p}min_box_height")),
        max_box_height=max_box_height,
        min_box_width=int(getattr(args, f"{p}min_box_width")),
        max_box_area_fraction=float(getattr(args, f"{p}max_box_area_fraction")),
        max_aspect_ratio=float(getattr(args, f"{p}max_aspect_ratio")),
        min_crop_height=int(getattr(args, f"{p}min_crop_height")),
        min_crop_width=int(getattr(args, f"{p}min_crop_width")),
    )

