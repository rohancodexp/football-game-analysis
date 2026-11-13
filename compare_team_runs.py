
"""
Compare two unsupervised team-clustering runs.

TEAM_A / TEAM_B are anonymous, per-fit labels, so a raw label-by-label diff
would report a 100 % change whenever the anchor rule flips. This tool therefore
aligns the two label vocabularies first (identity or swap, whichever agrees
more), then reports the residual differences.

Anonymous labels are only comparable within one video. Comparing two different
input datasets is reported as not meaningful.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


LABEL_ALIASES = {
    "identity": {"TEAM_A": "TEAM_A", "TEAM_B": "TEAM_B", "UNKNOWN": "UNKNOWN"},
    "swap": {"TEAM_A": "TEAM_B", "TEAM_B": "TEAM_A", "UNKNOWN": "UNKNOWN"},
}


def load_report(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def track_labels(report: dict) -> dict[str, str]:
    return {
        str(track_id): item["team_id"]
        for track_id, item in report.get("tracks", {}).items()
    }


def crop_key(item: dict) -> str:
    """Join key for a crop across reports.

    (track_id, frame) is present in every report version; the canonical
    detection_id is derived from the same pair, so the join is equivalent and
    works against pre-contract baselines such as v5.
    """
    return f"track{item['track_id']}_frame{item['frame']}"


def crop_labels(report: dict) -> dict[str, str]:
    return {crop_key(item): item["crop_prediction"] for item in report.get("samples", [])}


def crop_confidences(report: dict) -> dict[str, list[float]]:
    grouped: dict[str, list[float]] = {}
    for item in report.get("samples", []):
        grouped.setdefault(str(item["crop_prediction"]), []).append(
            float(item.get("crop_confidence", 0.0))
        )
    return grouped


def histogram(values: list[float]) -> dict:
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
    }


def confidence_distributions(report: dict) -> dict:
    """Prefer the recorded section so the diff covers what was published."""
    recorded = report.get("crop_confidence_distributions")
    if recorded:
        return recorded
    return {key: histogram(v) for key, v in sorted(crop_confidences(report).items())}


def normalize(labels: dict[str, str], alias: dict[str, str]) -> dict[str, str]:
    return {key: alias.get(value, value) for key, value in labels.items()}


def agreement(a: dict[str, str], b: dict[str, str]) -> int:
    shared = set(a) & set(b)
    return sum(1 for key in shared if a[key] == b[key])


def choose_alignment(reference: dict, candidate: dict) -> tuple[str, dict]:
    """Pick the vocabulary mapping for `candidate` that best fits `reference`."""
    ref_tracks = track_labels(reference)
    cand_tracks = track_labels(candidate)
    scores = {
        name: agreement(ref_tracks, normalize(cand_tracks, alias))
        for name, alias in LABEL_ALIASES.items()
    }
    best = max(scores, key=lambda name: (scores[name], name == "identity"))
    return best, LABEL_ALIASES[best]


def rate(counts: Counter, total: int) -> dict:
    return {
        str(key): {"count": int(value), "rate": round(value / total, 4) if total else 0.0}
        for key, value in sorted(counts.items(), key=lambda kv: str(kv[0]))
    }


def distribution_diff(old: dict, new: dict) -> dict:
    out = {}
    for team in sorted(set(old) | set(new), key=str):
        a, b = old.get(team, {}), new.get(team, {})
        if not a or not b:
            out[str(team)] = {"only_in": "reference" if a else "candidate", "old": a, "new": b}
            continue
        out[str(team)] = {
            "count": {"old": a["count"], "new": b["count"]},
            "mean_delta": round(b["mean"] - a["mean"], 4),
            "median_delta": round(b["median"] - a["median"], 4),
            "p10_delta": round(b["p10"] - a["p10"], 4),
            "p90_delta": round(b["p90"] - a["p90"], 4),
            "histogram_shift": [
                n - o for o, n in zip(a["histogram_10_bins"], b["histogram_10_bins"])
            ],
        }
    return out


def cluster_diff(old: dict, new: dict) -> dict:
    keys = (
        "valid_samples",
        "cluster_sizes",
        "outlier_radii",
        "mean_within_cluster_distance",
        "labeling_rule",
        "cluster_fingerprint",
        "anchor_crop_id",
    )
    return {
        key: {"reference": old.get(key), "candidate": new.get(key)}
        for key in keys
    }


def compare(reference: dict, candidate: dict) -> dict:
    same_input = reference.get("input_index") == candidate.get("input_index")

    ref_tracks = track_labels(reference)
    cand_tracks = track_labels(candidate)
    alias_name, alias = choose_alignment(reference, candidate)
    cand_tracks_aligned = normalize(cand_tracks, alias)

    shared_tracks = sorted(set(ref_tracks) & set(cand_tracks))
    changed_tracks = [t for t in shared_tracks if ref_tracks[t] != cand_tracks_aligned[t]]

    ref_crops = crop_labels(reference)
    cand_crops = crop_labels(candidate)
    cand_crops_aligned = normalize(cand_crops, alias)
    shared_crops = sorted(set(ref_crops) & set(cand_crops))
    changed_crops = [c for c in shared_crops if ref_crops[c] != cand_crops_aligned[c]]

    ref_track_counts = Counter(r["team_id"] for r in reference["tracks"].values())
    cand_track_counts = Counter(
        alias.get(r["team_id"], r["team_id"]) for r in candidate["tracks"].values()
    )
    ref_crop_counts = Counter(r["crop_prediction"] for r in reference["samples"])
    cand_crop_counts = Counter(
        alias.get(r["crop_prediction"], r["crop_prediction"]) for r in candidate["samples"]
    )

    conf_diff = {
        "crop": distribution_diff(
            confidence_distributions(reference),
            {
                alias.get(key, key): value
                for key, value in confidence_distributions(candidate).items()
            },
        )
    }

    candidate_stats = candidate.get("role_candidates", {})
    reference_stats = reference.get("role_candidates", {})

    return {
        "inputs": {
            "reference_index": reference.get("input_index"),
            "candidate_index": candidate.get("input_index"),
            "same_input_dataset": same_input,
            "comparability_warning": (
                None
                if same_input
                else "Different crop indexes: TEAM_A/TEAM_B are per-fit anonymous "
                     "labels and are not comparable across videos."
            ),
            "reference_fit_settings": reference.get("fit_settings"),
            "candidate_fit_settings": candidate.get("fit_settings"),
        },
        "vocabulary_alignment": {
            "chosen": alias_name,
            "mapping_applied_to_candidate": alias,
            "scores": {
                name: agreement(ref_tracks, normalize(track_labels(candidate), a))
                for name, a in LABEL_ALIASES.items()
            },
            "shared_tracks": len(shared_tracks),
        },
        "clusters": cluster_diff(
            reference.get("fit_report", {}), candidate.get("fit_report", {})
        ),
        "crop_predictions": {
            "reference": rate(ref_crop_counts, len(reference["samples"])),
            "candidate": rate(cand_crop_counts, len(candidate["samples"])),
            "changed_count": len(changed_crops),
            "changed_ratio": round(len(changed_crops) / len(shared_crops), 4)
            if shared_crops
            else 0.0,
        },
        "track_predictions": {
            "reference": rate(ref_track_counts, len(reference["tracks"])),
            "candidate": rate(cand_track_counts, len(candidate["tracks"])),
            "changed_tracks": changed_tracks,
            "changed_count": len(changed_tracks),
            "changed_ratio": round(len(changed_tracks) / len(shared_tracks), 4)
            if shared_tracks
            else 0.0,
        },
        "unknown_reasons": {
            "reference_tracks": reference.get("track_rejection_reasons"),
            "candidate_tracks": candidate.get("track_rejection_reasons"),
            "reference_crops": reference.get("crop_rejection_reasons"),
            "candidate_crops": candidate.get("crop_rejection_reasons"),
        },
        "candidates": {
            "reference": reference_stats.get("crop_stats", {})
            or {"note": "reference report has no role_candidates section"},
            "candidate": candidate_stats,
        },
        "disagreements": {
            "reference": reference.get("track_diagnostics"),
            "candidate": candidate.get("track_diagnostics"),
        },
        "confidence_distributions": conf_diff,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, help="Baseline team_clustering_report.json")
    parser.add_argument("--candidate", required=True, help="Comparison team_clustering_report.json")
    parser.add_argument("--out", default=None, help="Where to write the diff JSON")
    args = parser.parse_args()

    summary = compare(load_report(Path(args.reference)), load_report(Path(args.candidate)))

    text = json.dumps(summary, indent=2)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")

    print("\n===== TEAM CLUSTERING RUN COMPARISON =====")
    print("Reference:", summary["inputs"]["reference_index"])
    print("Candidate:", summary["inputs"]["candidate_index"])
    print("Same input dataset:", summary["inputs"]["same_input_dataset"])
    if summary["inputs"]["comparability_warning"]:
        print("WARNING:", summary["inputs"]["comparability_warning"])
    print("Vocabulary alignment:", summary["vocabulary_alignment"]["chosen"],
          summary["vocabulary_alignment"]["mapping_applied_to_candidate"])
    print("Crop counts      ref:", summary["crop_predictions"]["reference"])
    print("                 new:", summary["crop_predictions"]["candidate"])
    print("Crop changes:", summary["crop_predictions"]["changed_count"],
          f"({summary['crop_predictions']['changed_ratio']:.1%})")
    print("Track counts     ref:", summary["track_predictions"]["reference"])
    print("                 new:", summary["track_predictions"]["candidate"])
    print("Track changes:", summary["track_predictions"]["changed_count"],
          f"({summary['track_predictions']['changed_ratio']:.1%})")
    print("Cluster fingerprint ref:", summary["clusters"]["cluster_fingerprint"]["reference"])
    print("Cluster fingerprint new:", summary["clusters"]["cluster_fingerprint"]["candidate"])
    if args.out:
        print("Diff written to:", args.out)
    print("==========================================\n")


if __name__ == "__main__":
    main()
