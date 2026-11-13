
"""
Standalone team-colour clustering experiment.

Reads the crop_index.json produced by review_team_crops_v2.py,
filters low-quality samples, balances fitting samples by track,
fits TeamAppearanceClassifier, and produces track-level predictions.

This is an unsupervised experiment. TEAM_A / TEAM_B are anonymous
visual clusters, not verified real-world team identities.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from team_classifier import (
    CANDIDATE_MODE_MULTI,
    CANDIDATE_MODE_OFF,
    ROLE_NONE,
    TRACK_CANDIDATE_MEAN_SCORE,
    TRACK_CANDIDATE_SUPPORT_RATIO,
    TeamAppearanceClassifier,
    UNKNOWN,
    get_foreground_stats,
)
from crop_filter import (
    CropFilterPolicy,
    FilterResult,
    FilterStats,
    add_filter_arguments,
    policy_from_args,
)
from output_schema import (
    SCHEMA_VERSION,
    assign_detection_ids,
    detection_id_accounting,
    schema_descriptor,
    validate_crop_index_samples,
)


def load_and_filter_samples(
    index_path: Path,
    policy: CropFilterPolicy | None = None,
) -> tuple[dict, list[dict]]:
    with index_path.open("r", encoding="utf-8") as f:
        report = json.load(f)

    frame_w = int(report["video_width"])
    frame_h = int(report["video_height"])

    # If policy was not explicitly passed, inspect the index report
    index_policy_dict = report.get("filter_policy")
    if policy is None:
        if index_policy_dict is not None:
            policy = CropFilterPolicy.from_dict(index_policy_dict)
            policy_matched = True
        else:
            policy = CropFilterPolicy()
            policy_matched = False
    else:
        policy_matched = (index_policy_dict == policy.to_dict())

    stats = FilterStats()
    retained = []

    for sample in report["samples"]:
        result = policy.evaluate_sample_record(sample, frame_w, frame_h)
        stats.record(result)

        if not result.is_valid:
            continue

        item = dict(sample)
        item["track_id"] = int(item["track_id"])
        item["frame"] = int(item["frame"])
        item["crop_path"] = Path(item["file"])
        retained.append(item)

    # Legacy crop indexes predate the output contract. Each retained sample is
    # keyed by a unique (frame, track_id) pair, so re-applying the canonical
    # rule reproduces the same ID that run.py would have written.
    with_native_id = sum(
        1 for item in retained if isinstance(item.get("detection_id"), str)
    )
    retained = assign_detection_ids(retained)

    return {
        "video_width": frame_w,
        "video_height": frame_h,
        "original_samples": len(report["samples"]),
        "retained_samples": stats.unique_accepted,
        "rejected_samples": stats.unique_rejected,
        "filter_policy": policy.to_dict(),
        "index_policy_matches_runtime": policy_matched,
        "index_schema_version": report.get("schema_version"),
        "detection_id_source": (
            "empty"
            if not retained
            else "crop_index"
            if with_native_id == len(retained)
            else "reconstructed_from_samples"
            if with_native_id == 0
            else "mixed"
        ),
        "filter_stats": stats.to_dict(),
        "rejection_reasons_overlapping": dict(stats.rejection_reasons_overlapping),
        "primary_rejection_reasons": dict(stats.primary_rejection_reasons),
        "rejection_combinations": dict(stats.rejection_combinations),
    }, retained


def balance_by_track(
    samples: list[dict],
    max_per_track: int,
) -> list[dict]:
    """Select at most N evenly spaced samples from each track."""
    grouped = defaultdict(list)

    for sample in samples:
        grouped[sample["track_id"]].append(sample)

    selected = []

    for track_id in sorted(grouped):
        items = sorted(grouped[track_id], key=lambda x: x["frame"])
        count = min(max_per_track, len(items))

        if count == len(items):
            chosen = items
        elif count == 1:
            chosen = [items[len(items) // 2]]
        else:
            indices = np.linspace(
                0, len(items) - 1, count
            ).round().astype(int)
            chosen = [items[int(i)] for i in sorted(set(indices))]

        selected.extend(chosen)

    return selected


def read_crop(sample: dict):
    path = sample["crop_path"]

    if not path.is_absolute():
        path = Path.cwd() / path

    image = cv2.imread(str(path))
    return image


def confidence_summary(values: list[float]) -> dict:
    """Compact distribution description used to compare runs against v5."""
    if not values:
        return {"count": 0}
    arr = np.asarray(values, dtype=np.float64)
    hist, edges = np.histogram(arr, bins=10, range=(0.0, 1.0))
    return {
        "count": int(arr.size),
        "mean": round(float(arr.mean()), 4),
        "median": round(float(np.median(arr)), 4),
        "p10": round(float(np.percentile(arr, 10)), 4),
        "p90": round(float(np.percentile(arr, 90)), 4),
        "min": round(float(arr.min()), 4),
        "max": round(float(arr.max()), 4),
        "histogram_10_bins": [int(c) for c in hist],
        "bin_edges": [round(float(e), 2) for e in edges],
    }


def aggregate_track_candidates(
    samples: list[dict],
    *,
    support_ratio: float = TRACK_CANDIDATE_SUPPORT_RATIO,
    mean_score_floor: float = TRACK_CANDIDATE_MEAN_SCORE,
) -> dict[int, dict]:
    """Roll crop-level candidate evidence up to the track.

    Exclusion needs persistence: a single odd crop, or a high-scoring crop
    among many clean ones, is not enough to discard a track's team label.
    """
    grouped = defaultdict(list)
    for sample in samples:
        grouped[sample["track_id"]].append(sample)

    out = {}
    for track_id, items in grouped.items():
        roles = Counter(
            s["candidate_role"]
            for s in items
            if s["candidate_role"] not in (None, ROLE_NONE)
        )
        scores = [s["candidate_score"] for s in items if s["candidate_role"] not in (None, ROLE_NONE)]
        crops = len([s for s in items if s.get("candidate_role") is not None])
        support = len(scores) / crops if crops else 0.0
        mean_score = float(np.mean(scores)) if scores else 0.0
        persistent = (
            bool(scores)
            and support >= support_ratio
            and mean_score >= mean_score_floor
        )
        out[track_id] = {
            "candidate_crops": len(scores),
            "scored_crops": crops,
            "candidate_support": round(support, 4),
            "mean_candidate_score": round(mean_score, 4),
            "roles": dict(roles),
            "dominant_role": roles.most_common(1)[0][0] if roles else ROLE_NONE,
            "persistent_evidence": persistent,
        }
    return out


def apply_track_candidate_exclusions(
    track_results: dict[int, dict],
    track_candidate_stats: dict[int, dict],
    *,
    candidate_mode: str,
) -> list[int]:
    """Override a track's majority vote only on persistent candidate evidence.

    Scope note: ``candidate_action`` is a crop-level knob inside the classifier
    and deliberately does NOT gate this step. ``flag`` means "a single crop that
    looks like a non-team kit keeps its team label"; it does not mean a track
    whose every crop looks like that should still be assigned to a team. The
    escalation from weak single-crop evidence to sufficient track-level evidence
    is what the support/score thresholds express, and it applies identically
    under ``flag`` and ``reject``. Candidate detection switched off entirely is
    the only way to disable it.
    """
    excluded = []
    if candidate_mode == CANDIDATE_MODE_OFF:
        return excluded

    for track_id, result in track_results.items():
        evidence = track_candidate_stats[track_id]
        if result["team_id"] == UNKNOWN or not evidence["persistent_evidence"]:
            continue
        excluded.append(track_id)
        result["team_id_before_candidate_exclusion"] = result["team_id"]
        result["team_id"] = UNKNOWN
        result["reason"] = "persistent_role_candidate"

    return excluded


def make_contact_sheets_by_group(
    samples: list[dict],
    output_dir: Path,
    group_key: str = "track_team",
    view_mode: str = "track",
    batch_size: int = 40,
) -> dict[str, list[str]]:
    """Create contact sheets grouped either by track-level or crop-level prediction.

    In track view, minority-vote crops that disagree with the track majority
    are highlighted with a red border and warning label.
    """
    categories = {
        "TEAM_A": [],
        "TEAM_B": [],
        UNKNOWN: [],
    }

    for sample in samples:
        val = sample.get(group_key, UNKNOWN)
        categories.setdefault(val, []).append(sample)

    result = {}

    thumb_w = 220
    thumb_h = 150
    cols = 4
    caption_h = 44 if view_mode == "track" else 36
    rows_per_sheet = math.ceil(batch_size / cols)
    sheet_capacity = rows_per_sheet * cols

    for team_id, items in categories.items():
        team_dir = output_dir / team_id.lower()
        team_dir.mkdir(parents=True, exist_ok=True)

        paths = []

        for batch_no, start in enumerate(
            range(0, len(items), sheet_capacity), start=1
        ):
            batch = items[start:start + sheet_capacity]
            row_count = math.ceil(len(batch) / cols)

            sheet = np.full(
                (row_count * (thumb_h + caption_h), cols * thumb_w, 3),
                245,
                dtype=np.uint8,
            )

            for i, sample in enumerate(batch):
                image = read_crop(sample)
                if image is None:
                    continue

                h, w = image.shape[:2]
                scale = min(thumb_w / w, thumb_h / h)

                resized = cv2.resize(
                    image,
                    (
                        max(1, int(w * scale)),
                        max(1, int(h * scale)),
                    ),
                    interpolation=cv2.INTER_AREA,
                )

                x = (i % cols) * thumb_w
                y = (i // cols) * (thumb_h + caption_h)

                px = x + (thumb_w - resized.shape[1]) // 2
                py = y + (thumb_h - resized.shape[0]) // 2

                # Border and label styling
                is_disagree = (
                    view_mode == "track"
                    and sample.get("crop_prediction") != sample.get("track_team")
                )

                sheet[
                    py:py + resized.shape[0],
                    px:px + resized.shape[1]
                ] = resized

                if is_disagree:
                    # Highlight minority-vote disagreement with red border
                    cv2.rectangle(
                        sheet,
                        (px - 2, py - 2),
                        (px + resized.shape[1] + 1, py + resized.shape[0] + 1),
                        (0, 0, 220),
                        2,
                    )
                    label1 = (
                        f"T{sample['track_id']} F{sample['frame']} "
                        f"Trk:{sample['track_team']} ({sample['track_team_confidence']:.2f})"
                    )
                    label2 = (
                        f"! CROP: {sample['crop_prediction']} ({sample['crop_confidence']:.2f})"
                    )
                    cv2.putText(
                        sheet,
                        label1,
                        (x + 4, y + thumb_h + 18),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.40,
                        (15, 15, 15),
                        1,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        sheet,
                        label2,
                        (x + 4, y + thumb_h + 34),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.38,
                        (0, 0, 200),
                        1,
                        cv2.LINE_AA,
                    )
                else:
                    if view_mode == "track":
                        cv2.rectangle(
                            sheet,
                            (px - 1, py - 1),
                            (px + resized.shape[1], py + resized.shape[0]),
                            (50, 160, 50),
                            1,
                        )
                        label = (
                            f"T{sample['track_id']} F{sample['frame']} "
                            f"{sample['track_team']} ({sample['track_team_confidence']:.2f})"
                        )
                    else:
                        label = (
                            f"T{sample['track_id']} F{sample['frame']} "
                            f"Crop:{sample['crop_prediction']} ({sample['crop_confidence']:.2f})"
                        )
                    cv2.putText(
                        sheet,
                        label,
                        (x + 4, y + thumb_h + 22),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.42,
                        (15, 15, 15),
                        1,
                        cv2.LINE_AA,
                    )

            sheet_path = team_dir / f"contact_{batch_no:03d}.jpg"

            if not cv2.imwrite(str(sheet_path), sheet):
                raise RuntimeError(f"Could not write contact sheet: {sheet_path}")

            paths.append(str(sheet_path))

        result[team_id] = paths

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run standalone football team-colour clustering."
    )

    parser.add_argument(
        "--index",
        default=r"results\1_team_crop_review_v2\crop_index.json",
    )
    parser.add_argument(
        "--out",
        default=r"results\1_team_clustering",
    )
    parser.add_argument("--max-per-track-fit", type=int, default=2)
    parser.add_argument("--min-track-votes", type=int, default=2)
    parser.add_argument("--minimum-majority", type=float, default=0.60)
    parser.add_argument("--outlier-multiplier", type=float, default=1.35)
    parser.add_argument(
        "--use-grass-mask",
        action="store_true",
        help="Filter out pitch grass pixels from torso crops",
    )
    parser.add_argument(
        "--min-foreground-ratio",
        type=float,
        default=0.25,
        help="Minimum non-grass pixel ratio required",
    )
    parser.add_argument(
        "--detect-candidate-outliers",
        action="store_true",
        help="Deprecated v5 switch: yellow-only candidate detection that forces UNKNOWN",
    )
    parser.add_argument(
        "--candidate-mode",
        choices=["off", "legacy", "multi"],
        default=None,
        help="Signals used for referee/goalkeeper/odd-kit candidates. Also the "
             "switch for track-level candidate exclusion, which --candidate-action "
             "does not control. "
             "Default: 'legacy' with --detect-candidate-outliers, else 'off'.",
    )
    parser.add_argument(
        "--candidate-action",
        choices=["flag", "reject"],
        default=None,
        help="CROP-LEVEL consequence only. 'flag' annotates the crop and keeps its "
             "team prediction; 'reject' forces that crop to UNKNOWN. A track whose "
             "candidate evidence is persistent is excluded under either action.",
    )
    parser.add_argument(
        "--candidate-threshold",
        type=float,
        default=0.50,
        help="Multi-signal candidate score threshold (multi mode only)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=40,
        help="Number of crops per contact sheet",
    )

    add_filter_arguments(parser)

    args = parser.parse_args()

    if args.max_per_track_fit < 1:
        parser.error("--max-per-track-fit must be at least 1")

    if args.min_track_votes < 1:
        parser.error("--min-track-votes must be at least 1")

    if not 0.50 <= args.minimum_majority <= 1.0:
        parser.error("--minimum-majority must be between 0.50 and 1.0")

    if args.detect_candidate_outliers and args.candidate_mode not in (None, "legacy"):
        parser.error(
            "--detect-candidate-outliers is the legacy yellow-only path and cannot "
            f"be combined with --candidate-mode {args.candidate_mode}; use "
            "--candidate-mode and --candidate-action on their own instead."
        )

    if not 0.0 <= args.candidate_threshold <= 1.0:
        parser.error("--candidate-threshold must be between 0.0 and 1.0")

    index_path = Path(args.index)
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not index_path.exists():
        raise FileNotFoundError(f"Crop index not found: {index_path}")

    policy = policy_from_args(args)
    filter_report, samples = load_and_filter_samples(index_path, policy)

    print("\n========== TEAM CLUSTERING EXPERIMENT ==========")
    print("Original samples:", filter_report["original_samples"])
    print("Retained samples:", len(samples))
    print("Rejected samples:", filter_report["rejected_samples"])
    print("Unique retained tracks:", len({s["track_id"] for s in samples}))
    print("Policy status:", "Matches crop index policy perfectly" if filter_report["index_policy_matches_runtime"] else "Re-filtered with runtime policy")
    if filter_report["rejected_samples"] > 0:
        print("Overlapping rejection reasons:", filter_report["rejection_reasons_overlapping"])
        print("Primary rejection reasons:", filter_report["primary_rejection_reasons"])

    if len(samples) < 12:
        raise RuntimeError("Too few retained samples for clustering.")

    # Balance the fitting set so one long track cannot dominate.
    fit_samples = balance_by_track(
        samples,
        args.max_per_track_fit,
    )

    print("Balanced fitting samples:", len(fit_samples))
    print(
        "Unique fitting tracks:",
        len({s["track_id"] for s in fit_samples}),
    )

    fit_crops = []
    fit_ids = []

    for sample in fit_samples:
        crop = read_crop(sample)

        if crop is None:
            continue

        fit_crops.append(crop)
        fit_ids.append(sample["detection_id"])

    if len(fit_crops) < 12:
        raise RuntimeError(
            f"Only {len(fit_crops)} fitting crops could be read."
        )

    classifier = TeamAppearanceClassifier(
        min_samples=12,
        min_cluster_samples=4,
        random_state=17,
        outlier_multiplier=args.outlier_multiplier,
        use_grass_mask=args.use_grass_mask,
        min_foreground_ratio=args.min_foreground_ratio,
        detect_candidate_outliers=args.detect_candidate_outliers,
        candidate_mode=args.candidate_mode,
        candidate_action=args.candidate_action,
        candidate_threshold=args.candidate_threshold,
    )

    fit_report = classifier.fit(fit_crops, crop_ids=fit_ids)

    print("\nCluster fitting report:")
    print(json.dumps(fit_report, indent=2))

    # Predict all retained samples, not just fitting samples.
    track_votes = defaultdict(Counter)
    track_confidences = defaultdict(list)
    prediction_counts = Counter()
    crop_rejection_reasons = Counter()
    candidate_role_counts = Counter()
    candidate_signal_sums = Counter()
    candidate_signal_counts = Counter()
    crops_without_cluster_context = 0
    fg_ratios = []
    grass_pcts = []

    for sample in samples:
        crop = read_crop(sample)

        if crop is None:
            prediction = None
            team_id = UNKNOWN
            confidence = 0.0
            reason = "crop_read_failed"
            sample["foreground_ratio"] = 0.0
            sample["grass_removed_pct"] = 0.0
            sample["candidate_role"] = None
            sample["candidate_score"] = 0.0
            sample["candidate_signals"] = {}
            sample["candidate_reasons"] = []
        else:
            if args.use_grass_mask:
                fg_stats = get_foreground_stats(
                    crop,
                    min_foreground_ratio=args.min_foreground_ratio,
                )
                sample["foreground_ratio"] = fg_stats["foreground_ratio"]
                sample["grass_removed_pct"] = fg_stats["removed_percentage"]
                fg_ratios.append(fg_stats["foreground_ratio"])
                grass_pcts.append(fg_stats["removed_percentage"])
            else:
                sample["foreground_ratio"] = 1.0
                sample["grass_removed_pct"] = 0.0

            prediction = classifier.predict(crop)

            team_id = prediction.team_id
            confidence = float(prediction.confidence)
            reason = prediction.reason

            candidate = prediction.candidate
            sample["candidate_role"] = (
                candidate.role if candidate is not None else ROLE_NONE
            )
            sample["candidate_score"] = (
                round(float(candidate.score), 4) if candidate is not None else 0.0
            )
            sample["candidate_signals"] = (
                dict(candidate.signals) if candidate is not None else {}
            )
            sample["candidate_reasons"] = (
                list(candidate.reasons) if candidate is not None else []
            )

        sample["crop_prediction"] = team_id
        sample["crop_confidence"] = confidence
        sample["crop_reason"] = reason

        prediction_counts[team_id] += 1
        if team_id == UNKNOWN:
            crop_rejection_reasons[reason] += 1

        if sample.get("candidate_role") is not None:
            candidate_role_counts[sample["candidate_role"]] += 1
            for key, value in sample["candidate_signals"].items():
                # bool is an int subclass; context flags are counted separately.
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    continue
                candidate_signal_sums[key] += float(value)
                candidate_signal_counts[key] += 1
            if not sample["candidate_signals"].get("has_cluster_context", True):
                crops_without_cluster_context += 1

        if team_id != UNKNOWN:
            track_votes[sample["track_id"]][team_id] += 1
            track_confidences[sample["track_id"]].append(confidence)

    candidate_signal_means = {
        key: round(candidate_signal_sums[key] / candidate_signal_counts[key], 4)
        for key in sorted(candidate_signal_sums)
    }
    candidate_crop_stats = {
        "mode": classifier.candidate_mode,
        "action": classifier.candidate_action,
        "threshold": classifier.candidate_threshold,
        "fitting_set_gated_crops": int(fit_report["fitting_set_candidate_gated"]),
        "evaluated_crops": int(sum(
            1 for s in samples if s.get("candidate_role") is not None
        )),
        "crops_without_cluster_context": crops_without_cluster_context,
        "by_role": dict(candidate_role_counts),
        "flagged_crops": int(sum(
            count for role, count in candidate_role_counts.items()
            if role != ROLE_NONE
        )),
        "mean_signals": candidate_signal_means,
    }

    # Assign a team to a track only when enough non-UNKNOWN votes agree.
    track_results = {}
    track_rejection_reasons = Counter()

    for track_id in sorted({s["track_id"] for s in samples}):
        votes = track_votes[track_id]
        known_votes = sum(votes.values())

        if known_votes == 0:
            track_results[track_id] = {
                "team_id": UNKNOWN,
                "confidence": 0.0,
                "votes": dict(votes),
                "reason": "no_accepted_crop_predictions",
            }
            track_rejection_reasons["no_accepted_crop_predictions"] += 1
            continue

        team_id, winning_votes = votes.most_common(1)[0]
        majority = winning_votes / known_votes

        if known_votes < args.min_track_votes:
            final_team = UNKNOWN
            reason = "insufficient_track_votes"
        elif majority < args.minimum_majority:
            final_team = UNKNOWN
            reason = "conflicting_track_votes"
        else:
            final_team = team_id
            reason = "track_majority_vote"

        if final_team == UNKNOWN:
            track_rejection_reasons[reason] += 1

        track_results[track_id] = {
            "team_id": final_team,
            "confidence": round(
                float(np.mean(track_confidences[track_id])), 4
            ),
            "majority_ratio": round(float(majority), 4),
            "votes": dict(votes),
            "reason": reason,
        }

    # Persistent crop-level candidate evidence is the only route from a
    # majority vote to an excluded track. One strange crop never is.
    track_candidate_stats = aggregate_track_candidates(samples)
    for track_id, result in track_results.items():
        result["candidate_evidence"] = track_candidate_stats[track_id]

    tracks_excluded_as_candidates = apply_track_candidate_exclusions(
        track_results,
        track_candidate_stats,
        candidate_mode=classifier.candidate_mode,
    )
    track_rejection_reasons["persistent_role_candidate"] += len(
        tracks_excluded_as_candidates
    )

    # Add the final track label to each crop for contact-sheet grouping.
    for sample in samples:
        result = track_results[sample["track_id"]]
        sample["track_team"] = result["team_id"]
        sample["track_team_confidence"] = result["confidence"]

    track_counts = Counter(
        item["team_id"] for item in track_results.values()
    )

    # Track diagnostics
    track_crop_counts = Counter(s["track_id"] for s in samples)
    single_crop_tracks = sum(1 for cnt in track_crop_counts.values() if cnt == 1)
    multi_crop_tracks = sum(1 for cnt in track_crop_counts.values() if cnt > 1)

    disagree_crops = [
        s for s in samples
        if s["track_team"] != UNKNOWN and s["crop_prediction"] != s["track_team"]
    ]

    track_diagnostics = {
        "total_tracks": len(track_results),
        "single_crop_tracks": single_crop_tracks,
        "multi_crop_tracks": multi_crop_tracks,
        "minority_vote_disagreements": len(disagree_crops),
        "minority_disagreement_ratio": (
            round(len(disagree_crops) / len(samples), 4) if samples else 0.0
        ),
    }

    # Generate both contact sheet views
    sheets_by_track = make_contact_sheets_by_group(
        samples,
        output_dir / "contact_sheets" / "by_track_prediction",
        group_key="track_team",
        view_mode="track",
        batch_size=args.batch_size,
    )
    sheets_by_crop = make_contact_sheets_by_group(
        samples,
        output_dir / "contact_sheets" / "by_crop_prediction",
        group_key="crop_prediction",
        view_mode="crop",
        batch_size=args.batch_size,
    )
    candidate_items = [
        s for s in samples if s.get("candidate_role") not in (None, ROLE_NONE)
    ]
    sheets_by_candidate = (
        {
            role: paths
            for role, paths in make_contact_sheets_by_group(
                candidate_items,
                output_dir / "contact_sheets" / "by_role_candidate",
                group_key="candidate_role",
                view_mode="crop",
                batch_size=args.batch_size,
            ).items()
            if paths
        }
        if candidate_items
        else {}
    )

    # Store relative crop paths where possible.
    sample_records = []
    for sample in samples:
        sample_records.append({
            "file": str(sample["crop_path"]),
            "detection_id": sample["detection_id"],
            "track_id": sample["track_id"],
            "frame": sample["frame"],
            "detector_confidence": sample["detector_confidence"],
            "foreground_ratio": sample.get("foreground_ratio", 1.0),
            "grass_removed_pct": sample.get("grass_removed_pct", 0.0),
            "crop_prediction": sample["crop_prediction"],
            "crop_confidence": sample["crop_confidence"],
            "crop_reason": sample["crop_reason"],
            "candidate_role": sample.get("candidate_role"),
            "candidate_score": sample.get("candidate_score", 0.0),
            "candidate_reasons": sample.get("candidate_reasons", []),
            "track_team": sample["track_team"],
            "track_team_confidence": sample["track_team_confidence"],
        })

    grass_mask_stats = {}
    if args.use_grass_mask and fg_ratios:
        grass_mask_stats = {
            "mean_foreground_ratio": round(float(np.mean(fg_ratios)), 4),
            "mean_grass_removed_pct": round(float(np.mean(grass_pcts)), 2),
            "min_foreground_ratio": round(float(np.min(fg_ratios)), 4),
            "max_grass_removed_pct": round(float(np.max(grass_pcts)), 2),
        }

    confidences_by_team = defaultdict(list)
    for sample in samples:
        confidences_by_team[sample["crop_prediction"]].append(
            float(sample["crop_confidence"])
        )
    crop_confidence_distributions = {
        team: confidence_summary(values)
        for team, values in sorted(confidences_by_team.items(), key=lambda kv: str(kv[0]))
    }
    track_confidence_distributions = {
        team: confidence_summary([
            r["confidence"] for r in track_results.values() if r["team_id"] == team
        ])
        for team in sorted({r["team_id"] for r in track_results.values()}, key=str)
    }

    strongest_candidates = sorted(
        candidate_items,
        key=lambda s: (-s["candidate_score"], s["detection_id"]),
    )[:20]

    role_candidates = {
        "crop_stats": candidate_crop_stats,
        "track_stats": {
            "exclusion_rule": {
                "min_candidate_support": TRACK_CANDIDATE_SUPPORT_RATIO,
                "min_mean_candidate_score": TRACK_CANDIDATE_MEAN_SCORE,
                "applies_only_to": "tracks that would otherwise receive a team label",
                "controlled_by": "candidate_mode",
                "candidate_action_scope": (
                    "crop level only - persistent track evidence excludes under "
                    "both 'flag' and 'reject'"
                ),
            },
            "tracks_with_any_candidate_crop": int(sum(
                1 for e in track_candidate_stats.values() if e["candidate_crops"]
            )),
            "tracks_by_dominant_role": dict(Counter(
                e["dominant_role"] for e in track_candidate_stats.values()
                if e["dominant_role"] != ROLE_NONE
            )),
            "tracks_with_persistent_evidence": int(sum(
                1 for e in track_candidate_stats.values() if e["persistent_evidence"]
            )),
            "tracks_excluded_from_team": sorted(tracks_excluded_as_candidates),
        },
        "strongest_crops": [
            {
                "detection_id": s["detection_id"],
                "track_id": s["track_id"],
                "frame": s["frame"],
                "candidate_role": s["candidate_role"],
                "candidate_score": s["candidate_score"],
                "signals": s["candidate_signals"],
                "crop_prediction": s["crop_prediction"],
                "track_team": s["track_team"],
                "file": str(s["crop_path"]),
            }
            for s in strongest_candidates
        ],
    }

    final_report = {
        "experiment": "unsupervised_team_appearance_clustering",
        "schema_version": SCHEMA_VERSION,
        "input_index": str(index_path),
        "filtering": filter_report,
        "fit_settings": {
            "max_per_track_fit": args.max_per_track_fit,
            "min_track_votes": args.min_track_votes,
            "minimum_majority": args.minimum_majority,
            "outlier_multiplier": args.outlier_multiplier,
            "use_grass_mask": args.use_grass_mask,
            "min_foreground_ratio": args.min_foreground_ratio,
            "detect_candidate_outliers": args.detect_candidate_outliers,
            "candidate_mode": classifier.candidate_mode,
            "candidate_action": classifier.candidate_action,
            "candidate_threshold": classifier.candidate_threshold,
        },
        "fit_samples": len(fit_crops),
        "fit_report": fit_report,
        "grass_mask_stats": grass_mask_stats,
        "crop_prediction_counts": dict(prediction_counts),
        "crop_rejection_reasons": dict(crop_rejection_reasons),
        "crop_confidence_distributions": crop_confidence_distributions,
        "track_prediction_counts": dict(track_counts),
        "track_rejection_reasons": dict(track_rejection_reasons),
        "track_confidence_distributions": track_confidence_distributions,
        "track_diagnostics": track_diagnostics,
        "role_candidates": role_candidates,
        "tracks": {
            str(track_id): result
            for track_id, result in track_results.items()
        },
        "contact_sheets": {
            "by_track_prediction": sheets_by_track,
            "by_crop_prediction": sheets_by_crop,
            "by_role_candidate": sheets_by_candidate,
        },
        "samples": sample_records,
        "output_contract": schema_descriptor(),
        "detection_id": detection_id_accounting(sample_records),
        "sample_schema_validation": validate_crop_index_samples(
            sample_records, strict=True
        ),
        "label_stability": {
            "fit_is_permutation_invariant": True,
            "labeling_rule": fit_report["labeling_rule"],
            "cluster_fingerprint": fit_report["cluster_fingerprint"],
            "anchor_crop_id": fit_report["anchor_crop_id"],
            "note": (
                "TEAM_A/TEAM_B are reproducible for the same crop collection and "
                "are meaningless when compared across different videos; only the "
                "cluster_fingerprint is comparable, and only within one video."
            ),
        },
        "limitations": [
            "TEAM_A and TEAM_B are anonymous visual clusters.",
            "TEAM_A in one video is not TEAM_A in another; the anchor is per-fit.",
            "No real-world club identity is inferred.",
            "No ground-truth labels are available, so accuracy is not measured.",
            "Colour similarity does not prove player identity.",
            "Goalkeepers, referees, staff, and poor detections may be misclassified.",
            "Role candidates are weak evidence; only persistent track evidence excludes.",
            "Track labels are based on majority votes within each tracker ID.",
            "This experiment does not implement persistent player re-identification.",
        ],
    }

    report_path = output_dir / "team_clustering_report.json"
    report_path.write_text(
        json.dumps(final_report, indent=2),
        encoding="utf-8",
    )

    print("\nCrop predictions:")
    print(dict(prediction_counts))
    if crop_rejection_reasons:
        print("Crop UNKNOWN breakdown:", dict(crop_rejection_reasons))

    print("\nTrack-level predictions:")
    print(dict(track_counts))
    if track_rejection_reasons:
        print("Track UNKNOWN breakdown:", dict(track_rejection_reasons))

    print("\nTrack diagnostics:")
    print(json.dumps(track_diagnostics, indent=2))

    if classifier.candidate_mode != CANDIDATE_MODE_OFF:
        print("\nRole candidates (crop level):")
        print(json.dumps(candidate_crop_stats, indent=2))
        print("\nRole candidates (track level):")
        print(json.dumps(role_candidates["track_stats"], indent=2))

    if grass_mask_stats:
        print("\nGrass masking statistics:")
        print(json.dumps(grass_mask_stats, indent=2))

    print("\nConfidence distributions (crop level):")
    print(json.dumps(crop_confidence_distributions, indent=2))

    print("\nContact sheets (by track prediction):")
    for team_id, paths in sheets_by_track.items():
        print(f"  {team_id}: {len(paths)} sheet(s)")
        for path in paths:
            print("   ", path)

    print("\nContact sheets (by crop prediction):")
    for team_id, paths in sheets_by_crop.items():
        print(f"  {team_id}: {len(paths)} sheet(s)")
        for path in paths:
            print("   ", path)

    if sheets_by_candidate:
        print("\nContact sheets (by role candidate):")
        for role, paths in sheets_by_candidate.items():
            print(f"  {role}: {len(paths)} sheet(s)")
            for path in paths:
                print("   ", path)

    print("\nReport:", report_path)
    print("Schema version:", SCHEMA_VERSION, "| detection_id source:", filter_report["detection_id_source"])
    print(
        "Detection IDs: unique",
        final_report["detection_id"]["unique_detection_ids"],
        "of",
        final_report["detection_id"]["total_records"],
        "| validation errors:",
        final_report["sample_schema_validation"]["error_count"],
    )
    print("================================================\n")


if __name__ == "__main__":
    main()
