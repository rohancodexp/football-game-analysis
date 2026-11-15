"""
Read-only validation of a detection/tracking run.

Measures what the pipeline actually emitted - detection counts, per-frame
density, tracker-ID behaviour, fragment geometry, bounding-box and timestamp
validity, contract compliance and render output - without touching any
pipeline artefact.

Every statistic here is derived from the output files. Nothing in this module
judges identity correctness: without per-frame ground-truth annotations there is
no accuracy number to report, only measurable structural properties.

Usage:
    python validate_tracking_outputs.py --video PATH --run-dir results/videoval_x
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Optional

import cv2

from output_schema import (
    iter_jsonl, validate_accounting_rows, validate_identity_fields, validate_rows,
)
from identity_association import ACCEPTANCE_EXPLANATION, person_uid

ASSOCIATION_FIELDS = (
    'schema_version', 'person_uid', 'association_status',
    'association_reason', 'association_method',
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def quantiles(values: list[float], qs: Iterable[float] = (0.10, 0.25, 0.50, 0.75, 0.90)) -> dict:
    if not values:
        return {"count": 0}
    ordered = sorted(values)
    out = {
        "count": len(ordered),
        "min": round(ordered[0], 4),
        "max": round(ordered[-1], 4),
        "mean": round(sum(ordered) / len(ordered), 4),
    }
    for q in qs:
        pos = (len(ordered) - 1) * q
        low, high = math.floor(pos), math.ceil(pos)
        out[f"p{int(round(q * 100))}"] = round(ordered[low] + (ordered[high] - ordered[low]) * (pos - low), 4)
    return out


def histogram(values: list[float], edges: list[float]) -> dict:
    buckets = [0] * (len(edges) + 1)
    for v in values:
        index = len(edges)
        for position, edge in enumerate(edges):
            if v < edge:
                index = position
                break
        buckets[index] += 1
    return {
        "bin_edges": edges,
        "counts": buckets,
        "labels": (
            [f"<{edges[0]}"]
            + [f"{edges[i]}-{edges[i + 1]}" for i in range(len(edges) - 1)]
            + [f">={edges[-1]}"]
        ),
    }


def probe_video(path: Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {"opened": False, "path": str(path)}
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    claimed = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    decoded = 0
    while True:
        ok, _ = cap.read()
        if not ok:
            break
        decoded += 1
    cap.release()
    return {
        "opened": True,
        "path": str(path),
        "width": width,
        "height": height,
        "fps_metadata": round(fps, 6),
        "frame_count_metadata": claimed,
        "duration_s_metadata": round(claimed / fps, 4) if fps > 0 else None,
        "frames_decoded_by_sequential_read": decoded,
        "duration_s_decoded": round(decoded / fps, 4) if fps > 0 else None,
        "metadata_matches_decode": claimed == decoded,
    }


def iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def analyse_detections(rows: list[dict], width: int, height: int) -> dict:
    frames_seen = Counter(row["frame"] for row in rows)
    per_frame = list(frames_seen.values())
    track_ids = sorted({row["track_id"] for row in rows})

    areas = []
    invalid_order = 0
    non_finite = 0
    outside_left_top = 0
    outside_right_bottom = 0
    any_outside = 0
    outside_area_fraction = []
    for row in rows:
        box = [float(v) for v in row["bbox_xyxy"]]
        if any(not math.isfinite(v) for v in box):
            non_finite += 1
            continue
        x1, y1, x2, y2 = box
        if x2 <= x1 or y2 <= y1:
            invalid_order += 1
            continue
        areas.append((x2 - x1) * (y2 - y1))
        left_top = x1 < 0 or y1 < 0
        right_bottom = x2 > width or y2 > height
        if left_top:
            outside_left_top += 1
        if right_bottom:
            outside_right_bottom += 1
        if left_top or right_bottom:
            any_outside += 1
            cx1, cy1 = max(x1, 0.0), max(y1, 0.0)
            cx2, cy2 = min(x2, float(width)), min(y2, float(height))
            inside = max(0.0, cx2 - cx1) * max(0.0, cy2 - cy1)
            box_area = (x2 - x1) * (y2 - y1)
            outside_area_fraction.append(1.0 - inside / box_area if box_area > 0 else 0.0)

    max_frame_processed = max(frames_seen) if frames_seen else 0
    frames_with_zero = max_frame_processed + 1 - len(frames_seen)

    return {
        "total_detection_rows": len(rows),
        "distinct_frames_with_detections": len(frames_seen),
        "largest_frame_index": max_frame_processed,
        "frames_with_no_detection_in_processed_range": frames_with_zero,
        "detections_per_frame": {
            **quantiles([float(v) for v in per_frame], (0.10, 0.50, 0.90)),
            "mode": Counter(per_frame).most_common(3),
            "histogram": histogram([float(v) for v in per_frame], [1, 2, 3, 5, 8, 12, 16]),
        },
        "unique_track_ids": len(track_ids),
        "track_id_range": {"min": min(track_ids), "max": max(track_ids)} if track_ids else {},
        "detections_per_track": quantiles(
            [float(c) for c in Counter(row["track_id"] for row in rows).values()],
            (0.10, 0.50, 0.90),
        ),
        "detector_confidence": quantiles(
            [float(row["detector_confidence"]) for row in rows], (0.10, 0.50, 0.90)
        ),
        "bbox": {
            "non_finite_coordinates": non_finite,
            "degenerate_or_inverted": invalid_order,
            "well_formed": len(areas),
            "width_height_note": f"{width}x{height}",
            "area_px": quantiles(areas, (0.05, 0.50, 0.95)),
            "violates_left_or_top_boundary": outside_left_top,
            "violates_right_or_bottom_boundary": outside_right_bottom,
            "violates_any_boundary": any_outside,
            "fraction_of_box_outside_frame": quantiles(outside_area_fraction, (0.50, 0.90)),
        },
    }


def analyse_timestamps(rows: list[dict], fps: float) -> dict:
    missing = sum(1 for row in rows if row.get("timestamp_s") is None)
    non_numeric = sum(
        1
        for row in rows
        if row.get("timestamp_s") is not None
        and (isinstance(row["timestamp_s"], bool) or not isinstance(row["timestamp_s"], (int, float)))
    )
    checked = [row for row in rows if isinstance(row.get("timestamp_s"), (int, float))]
    mismatched = sum(
        1 for row in checked if abs(row["timestamp_s"] - row["frame"] / fps) > 1e-6
    )
    negative = sum(1 for row in checked if row["timestamp_s"] < 0)

    ordered = sorted(checked, key=lambda row: (row["frame"], row["detection_id"]))
    non_monotonic = sum(
        1
        for a, b in zip(ordered, ordered[1:])
        if a["frame"] == b["frame"] and b["timestamp_s"] < a["timestamp_s"]
    )

    frame_index_types = {type(row.get("frame")).__name__ for row in rows}
    return {
        "rows": len(rows),
        "missing_timestamp_s": missing,
        "non_numeric_timestamp_s": non_numeric,
        "negative_timestamp_s": negative,
        "non_finite_timestamp_s": sum(not math.isfinite(row['timestamp_s']) for row in checked),
        "timestamp_not_equal_to_frame_divided_by_fps": mismatched,
        "fps_used_by_writer": fps,
        "non_monotonic_within_same_frame": non_monotonic,
        "frame_index_types": sorted(frame_index_types),
        "span_s": {
            "first": round(ordered[0]["timestamp_s"], 4) if ordered else None,
            "last": round(ordered[-1]["timestamp_s"], 4) if ordered else None,
        },
    }


def analyse_track_continuity(rows: list[dict], gap_frames: int) -> dict:
    by_track = defaultdict(list)
    for row in rows:
        by_track[row["track_id"]].append(row["frame"])

    interruptions = []
    max_gap = []
    for track_id, frames in by_track.items():
        ordered = sorted(set(frames))
        gaps = [b - a for a, b in zip(ordered, ordered[1:])]
        breaks = [g for g in gaps if g > 1]
        interruptions.append(len(breaks))
        if gaps:
            max_gap.append(max(gaps))

    gap_counter = Counter()
    for track_id, frames in by_track.items():
        ordered = sorted(set(frames))
        for a, b in zip(ordered, ordered[1:]):
            gap_counter[min(b - a, 100)] += 1

    return {
        "note": (
            "Structural measure only: a track id absent for k frames is an "
            "interruption, not proof of an identity change."
        ),
        "tracks": len(by_track),
        "tracks_with_no_interruption": sum(1 for v in interruptions if v == 0),
        "interruptions_per_track": quantiles([float(v) for v in interruptions], (0.50, 0.90)),
        "total_interruptions": sum(interruptions),
        "longest_absence_per_track_frames": quantiles([float(v) for v in max_gap], (0.50, 0.90)),
        "absence_length_histogram": {str(k): v for k, v in sorted(gap_counter.items()) if k > 1},
        "interruptions_at_or_above_fragment_gap": sum(v for k, v in gap_counter.items() if k > gap_frames),
        "fragment_gap_setting_frames": gap_frames,
    }


def analyse_id_switch_heuristic(rows: list[dict], *, switch_iou: float = 0.35) -> dict:
    """Frame-adjacency heuristic, not ground truth.

    Two signals are measurable without annotations: (a) a track id that
    disappears and a different track id that appears inside the same box region
    in the next frame, and (b) two ids alive in the same frame whose boxes
    overlap heavily. Both are candidates for an ID switch; neither confirms one.
    """
    by_frame = defaultdict(list)
    for row in rows:
        by_frame[row["frame"]].append(row)

    consecutive_frames = sorted(by_frame)
    handoffs = []
    same_frame_overlaps = []

    for a, b in zip(consecutive_frames, consecutive_frames[1:]):
        if b - a != 1:
            continue
        prev = by_frame[a]
        nxt = by_frame[b]
        prev_ids = {row["track_id"] for row in prev}
        next_ids = {row["track_id"] for row in nxt}
        for new_row in nxt:
            if new_row["track_id"] in prev_ids:
                continue
            best = None
            for old_row in prev:
                if old_row["track_id"] not in next_ids:
                    continue
                value = iou(old_row["bbox_xyxy"], new_row["bbox_xyxy"])
                if best is None or value > best[0]:
                    best = (value, old_row["track_id"])
            if best and best[0] >= switch_iou:
                handoffs.append(
                    {
                        "frame": b,
                        "from_track_id": best[1],
                        "to_track_id": new_row["track_id"],
                        "iou": round(best[0], 4),
                    }
                )
        for i, row_a in enumerate(nxt):
            for row_b in nxt[i + 1 :]:
                value = iou(row_a["bbox_xyxy"], row_b["bbox_xyxy"])
                if value >= switch_iou:
                    same_frame_overlaps.append(
                        {
                            "frame": b,
                            "track_ids": [row_a["track_id"], row_b["track_id"]],
                            "iou": round(value, 4),
                        }
                    )

    new_ids = Counter(event["to_track_id"] for event in handoffs)
    return {
        "interpretation": (
            f"Counts of {switch_iou:.2f}+ IoU box continuity between a vanished id and a "
            "newly appearing id (candidate handoff), and of two ids overlapping in one frame. "
            "Unvalidated heuristic: no annotated IDs exist to confirm or refute any switch."
        ),
        "candidate_handoffs": len(handoffs),
        "candidate_handoffs_per_100_frames": round(
            100.0 * len(handoffs) / max(1, len(consecutive_frames)), 3
        ),
        "distinct_tracks_receiving_a_candidate_handoff": len(new_ids),
        "tracks_receiving_more_than_one": {str(k): v for k, v in new_ids.items() if v > 1},
        "handoff_iou": quantiles([e["iou"] for e in handoffs], (0.50, 0.90)),
        "strongest_examples": sorted(handoffs, key=lambda e: -e["iou"])[:10],
        "same_frame_high_overlap_pairs": len(same_frame_overlaps),
        "same_frame_overlap_examples": sorted(
            same_frame_overlaps, key=lambda e: -e["iou"]
        )[:10],
    }


def analyse_fragments(fragments: list[dict], fps: float, gap_frames: int, max_seconds: float) -> dict:
    durations = [f["end_s"] - f["start_s"] for f in fragments]
    spans = [f["end_frame"] - f["start_frame"] for f in fragments]
    per_track = Counter(f["track_id"] for f in fragments)

    at_cap = 0
    for f in fragments:
        cap_frames = int(max_seconds * fps)
        if (f["end_frame"] - f["start_frame"]) >= cap_frames - 1:
            at_cap += 1

    return {
        "fragments": len(fragments),
        "fragments_per_track": {
            "single_fragment": sum(1 for v in per_track.values() if v == 1),
            "multi_fragment": sum(1 for v in per_track.values() if v > 1),
            "max": max(per_track.values()) if per_track else 0,
            "distribution": quantiles([float(v) for v in per_track.values()], (0.50, 0.90)),
        },
        "duration_s": quantiles(durations, (0.10, 0.25, 0.50, 0.75, 0.90)),
        "duration_histogram": histogram(durations, [0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 14.9]),
        "span_frames": quantiles([float(s) for s in spans], (0.10, 0.50, 0.90)),
        "zero_length_fragments": sum(1 for s in spans if s < 0),
        "inverted_fragments": sum(1 for f in fragments if f["end_frame"] < f["start_frame"]),
        "fragments_at_max_duration_cap": at_cap,
        "max_duration_cap_s": max_seconds,
        "fragment_gap_setting_frames": gap_frames,
        "unique_fragment_ids": len({f["fragment_id"] for f in fragments}),
        "fragment_ids_matching_pattern": sum(
            1 for f in fragments if str(f.get("fragment_id", "")).startswith("F")
        ),
        "embeddings_present": sum(1 for f in fragments if f.get("embedding")),
        "embeddings_missing": sum(1 for f in fragments if not f.get("embedding")),
        "jersey_ocr_values": dict(Counter(str(f.get("jersey")) for f in fragments)),
        "player_id_values": dict(Counter(str(f.get("player_id")) for f in fragments)),
        "team_id_values": dict(Counter(str(f.get("team_id")) for f in fragments)),
    }


def analyse_output_consistency(
    boxes: list[dict], identified: list[dict], fragments: list[dict]
) -> dict:
    fragment_ids = {f["fragment_id"] for f in fragments}
    boxes_ids = {row["fragment_id"] for row in boxes}
    ident_ids = {row["fragment_id"] for row in identified}
    box_by_id = {row["detection_id"]: row for row in boxes}
    ident_by_id = {row["detection_id"]: row for row in identified}

    shared = set(box_by_id) & set(ident_by_id)
    field_drift = Counter()
    for detection_id in shared:
        a, b = box_by_id[detection_id], ident_by_id[detection_id]
        for key in ("frame", "timestamp_s", "track_id", "fragment_id", "bbox_xyxy",
                    "detector_confidence", "class_id"):
            if a.get(key) != b.get(key):
                field_drift[key] += 1

    return {
        "boxes_rows": len(boxes),
        "identified_rows": len(identified),
        "row_count_equal": len(boxes) == len(identified),
        "detection_ids_match": box_by_id.keys() == ident_by_id.keys(),
        "fragment_ids_unique": len(fragment_ids) == len(fragments),
        "detection_id_unique_in_boxes": len(box_by_id) == len(boxes),
        "detection_id_unique_in_identified": len(ident_by_id) == len(identified),
        "fragment_ids_referenced_by_boxes_not_in_fragments": sorted(boxes_ids - fragment_ids)[:10],
        "fragment_ids_in_fragments_never_written": sorted(fragment_ids - boxes_ids)[:10],
        "boxes_vs_identified_fragment_ids": {
            "only_in_boxes": len(boxes_ids - ident_ids),
            "only_in_identified": len(ident_ids - boxes_ids),
        },
        "spatiotemporal_field_drift_between_files": dict(field_drift),
    }


def analyse_render(run_dir: Path, expected_frames: int, video: dict) -> dict:
    path = run_dir / "identified.mp4"
    if not path.exists():
        return {"file": str(path), "exists": False}
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {"file": str(path), "exists": True, "opened": False}
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    claimed = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    decoded = 0
    while True:
        ok, _ = cap.read()
        if not ok:
            break
        decoded += 1
    cap.release()
    return {
        "file": str(path),
        "exists": True,
        "opened": True,
        "size_bytes": path.stat().st_size,
        "width": width,
        "height": height,
        "fps": round(fps, 4),
        "frame_count_metadata": claimed,
        "frames_decodable": decoded,
        "matches_frames_processed": decoded == expected_frames,
        "frames_processed": expected_frames,
        "geometry_matches_source": (width, height) == (video.get("width"), video.get("height")),
        "fourcc_note": "written with mp4v; container frame count is codec-dependent",
    }


def analyse_detection_accounting(
    run_dir: Path,
    *,
    boxes_rows: int,
    identified_rows: int,
    expected_frames: Optional[int],
) -> dict:
    """Verify detection_accounting.jsonl and its summary. Reports only; never rewrites files."""
    path = run_dir / "detection_accounting.jsonl"
    summary_path = run_dir / "detection_accounting_summary.json"
    if not path.exists():
        return {"exists": False, "valid": False,
                "errors": ["detection_accounting.jsonl is missing; this run predates Stage 2 accounting or failed to write it"]}

    records = list(iter_jsonl(path))
    result = validate_accounting_rows(
        records,
        expected_frames=expected_frames,
        boxes_rows=boxes_rows,
        identified_rows=identified_rows,
    )
    errors = list(result["errors"])

    # Per-frame flags, computed directly from the records rather than inferred from error
    # strings. validate_accounting_rows already reports the matching errors.
    flags = {"bands_sum_to_raw": True, "emitted_le_raw": True, "dropped_equals_arithmetic": True,
             "ids_match_emitted": True, "sequential_frames": True}
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            continue
        raw, emitted = record.get("raw_detections"), record.get("emitted_detections")
        dropped = record.get("dropped_detections")
        bands = [record.get(k) for k in ("raw_high_conf", "raw_low_conf", "raw_below_low")]
        if all(isinstance(b, int) and not isinstance(b, bool) for b in bands) \
                and isinstance(raw, int) and sum(bands) != raw:
            flags["bands_sum_to_raw"] = False
        if isinstance(raw, int) and isinstance(emitted, int) and emitted > raw:
            flags["emitted_le_raw"] = False
        if all(isinstance(v, int) for v in (raw, emitted, dropped)) and dropped != raw - emitted:
            flags["dropped_equals_arithmetic"] = False
        ids = record.get("emitted_track_ids")
        if isinstance(ids, list) and isinstance(emitted, int) and len(ids) != emitted:
            flags["ids_match_emitted"] = False
        if record.get("frame") != index:
            flags["sequential_frames"] = False

    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else None
    summary_check: dict[str, Any] = {"exists": summary is not None}
    if summary is not None:
        recomputed = result["totals"]
        reported = summary.get("totals", {})
        mismatched = {
            key: {"summary": reported.get(key), "recomputed": recomputed[key]}
            for key in recomputed
            if reported.get(key) != recomputed[key]
        }
        histogram_mismatch = reported.get("drop_histogram") != {str(k): v for k, v in sorted(
            Counter(record.get("dropped_detections", 0) for record in records if isinstance(record, dict)).items())}
        expected_max_drop = max((record.get("dropped_detections", 0) for record in records
                                 if isinstance(record, dict)), default=0)
        max_drop_mismatch = reported.get("max_dropped_detections") != expected_max_drop
        if mismatched:
            errors.append(f"summary totals disagree with per-frame records: {mismatched}")
        if histogram_mismatch:
            errors.append("summary drop_histogram disagrees with per-frame records")
        if max_drop_mismatch:
            errors.append(f"summary max_dropped_detections {reported.get('max_dropped_detections')!r} "
                          f"!= recomputed {expected_max_drop}")
        consistency = summary.get("consistency", {})
        if not consistency.get("all_hold", False):
            errors.append(f"summary reports consistency violations: {consistency.get('violations')}")
        summary_check.update({
            "accounting_version": summary.get("accounting_version"),
            "frames_processed": summary.get("frames_processed"),
            "totals_agree_with_records": not mismatched and not histogram_mismatch and not max_drop_mismatch,
            "consistency_all_hold": consistency.get("all_hold"),
            "consistency_violations": consistency.get("violations"),
            "rates": summary.get("rates"),
            "not_measured_present": "per_detection_drop_reason" in summary.get("not_measured", {}),
        })
    else:
        errors.append("detection_accounting_summary.json is missing")

    result["errors"] = errors[:50]
    result["error_count"] = len(errors)
    result["valid"] = not errors
    result["summary"] = summary_check
    result["checks"] = {
        "one_record_per_processed_frame": expected_frames is None or len(records) == expected_frames,
        "frame_indices_sequential": flags["sequential_frames"],
        "bands_are_mutually_exclusive_and_sum_to_raw": flags["bands_sum_to_raw"],
        "emitted_le_raw": flags["emitted_le_raw"],
        "dropped_equals_raw_minus_emitted": flags["dropped_equals_arithmetic"],
        "emitted_equals_boxes_rows": boxes_rows == result["totals"]["emitted_detections"],
        "emitted_equals_identified_rows": identified_rows == result["totals"]["emitted_detections"],
        "skipped_equals_zero_raw_frames": (
            result["totals"]["frames_tracker_update_skipped"] == result["totals"]["frames_raw_zero"]),
        "emitted_ids_match_emitted_count": flags["ids_match_emitted"],
        "summary_agrees_with_records": summary_check.get("totals_agree_with_records", False),
    }
    result["check_count_failed"] = sum(1 for ok in result["checks"].values() if ok is False)
    return result


def analyse_schema(run_dir: Path) -> dict:
    out = {}
    for name in ("boxes.jsonl", "identified_boxes.jsonl"):
        path = run_dir / name
        if not path.exists():
            out[name] = {"exists": False}
            continue
        rows = list(iter_jsonl(path))
        out[name] = {
            "exists": True,
            "required_level": validate_rows(rows, strict=False),
            "strict_level": validate_rows(rows, strict=True),
        }
    manifest_path = run_dir / "run_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        out["run_manifest.json"] = {
            "exists": True,
            "schema_version": manifest.get("schema_version"),
            "has_output_contract_descriptor": "output_contract" in manifest,
            "declares_trained_in_this_project": manifest.get("trained_in_this_project"),
            "identity_fields_note_present": "identity_fields_note" in manifest,
            "embedding_descriptor": manifest.get("embedding"),
        }
    else:
        out["run_manifest.json"] = {"exists": False}
    return out


def analyse_identity_associations(fragments, identified, report, manifest=None,
                                  provenance_digests=None):
    errors = []
    applicable = (report is not None or any(
        isinstance(row, dict) and ('person_uid' in row or 'schema_version' in row)
        for row in [*fragments, *identified]
    ) or (manifest or {}).get('schema_version') == '1.1')
    if not applicable:
        return {'applicable': False, 'valid': True, 'errors': [],
                'note': 'Legacy artifact; anonymous identity was not emitted.'}

    def index(rows, key, label):
        indexed = {}
        if not isinstance(rows, list):
            errors.append(f'{label}: expected a list')
            return indexed
        for row in rows:
            if not isinstance(row, dict):
                errors.append(f'{label}: expected an object')
                continue
            value = row.get(key)
            if not isinstance(value, str) or not value or value in indexed:
                errors.append(f'{label}: missing, invalid or duplicate {key}')
            else:
                indexed[value] = row
        return indexed

    by_fragment = index(fragments, 'fragment_id', 'fragments')
    for fid, fragment in by_fragment.items():
        errors.extend(validate_identity_fields(fragment, context=fid))
        team = fragment.get('team_classification')
        if not isinstance(team, dict) or team.get('team_id') != fragment.get('team_id'):
            errors.append(f'{fid}: team_classification must agree with the final team_id')
        else:
            score = team.get('score')
            if (type(score) not in (int, float) or not 0 <= score <= 1
                    or (fragment.get('team_id') == 'UNKNOWN' and score != 0)):
                errors.append(f'{fid}: invalid final team score')
        start, end = fragment.get('start_frame'), fragment.get('end_frame')
        if type(start) is not int or type(end) is not int or not 0 <= start <= end:
            errors.append(f'{fid}: invalid inclusive frame interval')
    for row in identified:
        if not isinstance(row, dict):
            errors.append('identified: expected an object')
            continue
        errors.extend(validate_identity_fields(row, context=str(row.get('detection_id'))))
        fragment = by_fragment.get(row.get('fragment_id')) if isinstance(row.get('fragment_id'), str) else None
        if fragment is None:
            errors.append('identified: unknown fragment reference')
        elif any(row.get(key) != fragment.get(key) for key in ASSOCIATION_FIELDS):
            errors.append(f"{row.get('detection_id')}: association differs from fragment")
    if not isinstance(report, dict):
        errors.append('missing or malformed identity_association_report.json')
        report = {}
    decisions = index(report.get('decisions'), 'fragment_id', 'decisions')
    if decisions.keys() != by_fragment.keys():
        errors.append('report decisions do not cover exactly the fragments')
    for fid in decisions.keys() & by_fragment.keys():
        if any(decisions[fid].get(key) != by_fragment[fid].get(key) for key in ASSOCIATION_FIELDS):
            errors.append(f'{fid}: report decision differs from fragment')
    pairs = report.get('pairs', [])
    accepted_pairs, seen_pairs = set(), set()
    if not isinstance(pairs, list):
        errors.append('report pairs must be a list')
        pairs = []
    for pair in pairs:
        if not isinstance(pair, dict):
            errors.append('report pair must be an object')
            continue
        left, right = pair.get('left'), pair.get('right')
        if (not isinstance(left, str) or not isinstance(right, str)
                or left not in by_fragment or right not in by_fragment or left == right):
            errors.append('report pair has invalid fragment references')
            continue
        key = tuple(sorted((left, right)))
        if key in seen_pairs:
            errors.append('report contains duplicate pair evidence')
        seen_pairs.add(key)
        if type(pair.get('accepted')) is not bool:
            errors.append('report pair accepted must be boolean')
        elif pair.get('accepted') is True:
            # pairs[].reasons carries an acceptance explanation for accepted
            # pairs; only entries other than that explanation count as vetoes.
            reasons = pair.get('reasons')
            vetoes = ([r for r in reasons if r != ACCEPTANCE_EXPLANATION]
                      if isinstance(reasons, list) else ['malformed_reasons'])
            if pair.get('decision') != 'same' or vetoes:
                errors.append('accepted pair must have a same judgment and no vetoes')
            else:
                accepted_pairs.add(key)
    groups = index(report.get('groups'), 'person_uid', 'groups')
    membership = {}
    for uid, group in groups.items():
        members = group.get('fragment_ids')
        if (not isinstance(members, list) or len(members) < 2
                or any(not isinstance(fid, str) for fid in members)):
            errors.append(f'{uid}: group must contain at least two fragment IDs')
            continue
        if len(members) != len(set(members)):
            errors.append(f'{uid}: duplicate group member')
        for i, fid in enumerate(members):
            if fid in membership or fid not in by_fragment:
                errors.append(f'{uid}: repeated or unknown member {fid}')
                continue
            membership[fid] = uid
            fragment = by_fragment[fid]
            if fragment.get('person_uid') != uid or fragment.get('association_status') != 'associated':
                errors.append(f'{fid}: group membership contradicts fragment')
            for other in members[:i]:
                if tuple(sorted((fid, other))) not in accepted_pairs:
                    errors.append(f'{uid}: missing accepted direct evidence for {fid}/{other}')
                previous = by_fragment.get(other, {})
                times = [fragment.get('start_frame'), fragment.get('end_frame'),
                         previous.get('start_frame'), previous.get('end_frame')]
                if all(type(value) is int for value in times):
                    if max(times[0], times[2]) <= min(times[1], times[3]):
                        errors.append(f'{uid}: inclusive temporal overlap between {fid}/{other}')
    for fid, fragment in by_fragment.items():
        if fragment.get('person_uid') != membership.get(fid):
            errors.append(f'{fid}: UID has no matching report group')
    provenance = report.get('provenance')
    if provenance is not None and not isinstance(provenance, dict):
        errors.append('report provenance must be an object when present')
        provenance = None
    if groups and provenance is None:
        errors.append('accepted identity groups require report provenance')
    digests_ok = False
    if isinstance(provenance, dict):
        video_hex, boxes_hex = provenance.get('video_sha256'), provenance.get('boxes_sha256')
        digests_ok = all(isinstance(value, str)
                         and re.fullmatch(r'[0-9a-f]{64}', value) is not None
                         for value in (video_hex, boxes_hex))
        if not digests_ok:
            errors.append('report provenance digests must be lower-case SHA256 hex digests')
        if provenance_digests is not None and digests_ok:
            for key, value in (('video_sha256', video_hex), ('boxes_sha256', boxes_hex)):
                expected = provenance_digests.get(key)
                if expected is not None and value != expected:
                    errors.append(f'stale provenance: report {key} does not match the current input')
        if digests_ok:
            for uid, group in groups.items():
                members = group.get('fragment_ids')
                if isinstance(members, list) and all(isinstance(fid, str) for fid in members):
                    if uid != person_uid(video_hex, boxes_hex, members):
                        errors.append(f'{uid}: person_uid is not the deterministic digest '
                                      'of the provenance digests and sorted member IDs')
    if manifest is not None:
        expected = {'boxes.jsonl': '1.0', 'identified_boxes.jsonl': '1.1',
                    'fragments.json': '1.1', 'detection_accounting.jsonl': '1.0'}
        if manifest.get('schema_version') != '1.1':
            errors.append('manifest must declare enriched schema 1.1')
        versions = manifest.get('artifact_schema_versions', {})
        if not isinstance(versions, dict) or any(versions.get(k) != v for k, v in expected.items()):
            errors.append('manifest artifact schema versions are missing or inconsistent')
    return {'applicable': True, 'valid': not errors, 'error_count': len(errors),
            'errors': errors, 'groups': len(groups), 'fragments': len(by_fragment),
            'associated_fragments': len(membership),
            'unresolved_fragments': len(by_fragment) - len(membership),
            'note': 'Cross-file consistency, not real-world identity accuracy.'}


def assess_acceptance(report: dict, *, require_render: bool = False) -> dict:
    """Aggregate structural checks into one pass/fail decision.

    Each check reads a field the analysis sections already computed; nothing is
    re-measured here. Passing means the outputs are internally consistent and
    schema-valid. It is not tracking or identity accuracy.
    """
    checks: dict[str, bool] = {}

    def check(name: str, passed: Any) -> None:
        checks[name] = bool(passed)

    schema = report.get("schema", {})
    for name in ("boxes.jsonl", "identified_boxes.jsonl"):
        entry = schema.get(name, {})
        check(f"schema.{name}.exists", entry.get("exists"))
        check(f"schema.{name}.strict_valid",
              entry.get("exists") and entry.get("strict_level", {}).get("valid"))
    check("schema.run_manifest.exists", schema.get("run_manifest.json", {}).get("exists"))

    consistency = report.get("output_consistency", {})
    for key in ("row_count_equal", "detection_ids_match", "fragment_ids_unique",
                "detection_id_unique_in_boxes", "detection_id_unique_in_identified"):
        check(f"consistency.{key}", consistency.get(key))
    check("consistency.no_unknown_fragment_references",
          consistency.get("fragment_ids_referenced_by_boxes_not_in_fragments") == [])
    check("consistency.no_field_drift",
          consistency.get("spatiotemporal_field_drift_between_files") == {})

    timestamps = report.get("timestamps", {})
    for key in ("missing_timestamp_s", "non_numeric_timestamp_s", "negative_timestamp_s",
                "non_finite_timestamp_s", "timestamp_not_equal_to_frame_divided_by_fps",
                "non_monotonic_within_same_frame"):
        check(f"timestamps.{key}_zero", timestamps.get(key) == 0)

    fragments = report.get("fragments", {})
    check("fragments.no_inverted", fragments.get("inverted_fragments") == 0)
    check("fragments.ids_unique", fragments.get("unique_fragment_ids") == fragments.get("fragments"))

    check("detection_accounting.valid", report.get("detection_accounting", {}).get("valid"))
    check("identity_association.valid", report.get("identity_association", {}).get("valid"))

    render = report.get("render", {})
    if render.get("exists"):
        check("render.opened", render.get("opened"))
        check("render.frames_match", render.get("matches_frames_processed"))
        check("render.geometry_matches_source", render.get("geometry_matches_source"))
    else:
        check("render.present_if_required", not require_render)

    failures = sorted(name for name, passed in checks.items() if not passed)
    return {"passed": not failures, "checks": checks, "failures": failures,
            "require_render": require_render,
            "note": "Structural acceptance only; not tracking or identity accuracy."}


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", help="New external report; defaults to a sibling <run-dir>_validation.json")
    ap.add_argument("--fps-fallback", type=float, default=0.0)
    ap.add_argument("--fragment-gap", type=int, default=30)
    ap.add_argument("--max-fragment-seconds", type=float, default=15.0)
    ap.add_argument("--switch-iou", type=float, default=0.35)
    ap.add_argument("--require-render", action="store_true",
                    help="fail acceptance when identified.mp4 is absent")
    args = ap.parse_args(argv)

    run_dir = Path(args.run_dir).resolve()
    out_path = Path(args.out).resolve() if args.out else run_dir.with_name(run_dir.name + '_validation.json')
    if out_path.exists() or out_path.is_symlink():
        ap.error(f'refusing to overwrite existing report or input: {out_path}')
    if out_path.is_relative_to(run_dir) or not out_path.parent.is_dir():
        ap.error('--out must have an existing parent outside the run directory')
    video = probe_video(Path(args.video))
    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    fps = float(manifest.get("fps") or video.get("fps_metadata") or args.fps_fallback or 0)
    if fps <= 0:
        raise SystemExit("No usable FPS; aborting rather than inventing timestamps.")

    boxes = list(iter_jsonl(run_dir / "boxes.jsonl")) if (run_dir / "boxes.jsonl").exists() else []
    identified = (
        list(iter_jsonl(run_dir / "identified_boxes.jsonl"))
        if (run_dir / "identified_boxes.jsonl").exists()
        else []
    )
    fragments = json.loads((run_dir / "fragments.json").read_text(encoding="utf-8")) if (
        run_dir / "fragments.json"
    ).exists() else []

    report = {
        "scope_note": (
            "Structural validation of detection/tracking output. No per-frame "
            "ground-truth annotations exist for these inputs, so no tracking "
            "accuracy, IDF1, MOTA or ID-switch rate is reported or implied."
        ),
        "inputs": {
            "video": video,
            "run_dir": str(run_dir),
            "manifest": {k: manifest.get(k) for k in ("video", "frames", "fps", "detector", "threshold", "margin", "ocr")},
        },
        "detections": analyse_detections(boxes, int(video.get("width") or 0), int(video.get("height") or 0)),
        "timestamps": analyse_timestamps(boxes, fps),
        "track_continuity": analyse_track_continuity(boxes, args.fragment_gap),
        "id_switch_heuristic": analyse_id_switch_heuristic(boxes, switch_iou=args.switch_iou),
        "fragments": analyse_fragments(fragments, fps, args.fragment_gap, args.max_fragment_seconds),
        "output_consistency": analyse_output_consistency(boxes, identified, fragments),
        "detection_accounting": analyse_detection_accounting(
            run_dir,
            boxes_rows=len(boxes),
            identified_rows=len(identified),
            expected_frames=int(manifest["frames"]) if manifest.get("frames") is not None else None,
        ),
        "render": analyse_render(run_dir, int(manifest.get("frames") or 0), video),
        "schema": analyse_schema(run_dir),
    }
    identity_path = run_dir / 'identity_association_report.json'
    identity_report = None
    if identity_path.exists():
        try:
            identity_report = json.loads(identity_path.read_text(encoding='utf-8'))
        except (ValueError, OSError):
            identity_report = {}
    report['identity_association'] = analyse_identity_associations(
        fragments, identified, identity_report, manifest,
        provenance_digests={
            'video_sha256': _file_sha256(Path(args.video)),
            'boxes_sha256': (_file_sha256(run_dir / 'boxes.jsonl')
                             if (run_dir / 'boxes.jsonl').exists() else None),
        })

    report['acceptance'] = assess_acceptance(report, require_render=args.require_render)

    with out_path.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({section: _brief(report[section]) for section in ("detections", "render")}, indent=2))
    acceptance = report['acceptance']
    print(f"Acceptance: {'PASS' if acceptance['passed'] else 'FAIL'}")
    for name in acceptance['failures']:
        print(f"  failed: {name}")
    print(f"Full report: {out_path}")
    return 0 if acceptance['passed'] else 1


def _brief(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _brief(v) for k, v in value.items() if k not in {"histogram", "strongest_examples", "same_frame_overlap_examples", "absence_length_histogram"}}
    if isinstance(value, list):
        return value[:3]
    return value


if __name__ == "__main__":
    raise SystemExit(main())
