
"""Quality-filtered, track-balanced torso-crop review tool.

Example:
    python review_team_crops_v2.py \
        --video datasets/infactory_soccer_events/data/1.mp4 \
        --boxes results/1_out_baseline/boxes.jsonl \
        --out results/1_team_crop_review_v2

This is a visual QA utility. It does not classify teams or identify players.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict, Counter
from pathlib import Path

import cv2
import numpy as np

from team_classifier import extract_torso_crop
from crop_filter import (
    CropFilterPolicy,
    FilterResult,
    FilterStats,
    REASON_INVALID_METADATA,
    add_filter_arguments,
    policy_from_args,
)
from output_schema import (
    SCHEMA_VERSION,
    assign_detection_ids,
    schema_descriptor,
    validate_rows,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate quality-filtered torso crop contact sheets."
    )

    parser.add_argument("--video", required=True)
    parser.add_argument("--boxes", required=True)
    parser.add_argument("--out", required=True)

    parser.add_argument("--max-per-track", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=40)

    add_filter_arguments(parser)

    args = parser.parse_args()

    if args.max_per_track < 1:
        parser.error("--max-per-track must be positive")

    if args.batch_size < 1:
        parser.error("--batch-size must be positive")

    return args


def load_detections(boxes_path: Path):
    """Load detection records from JSONL.

    Legacy files that predate the output contract have no ``detection_id``;
    they are backfilled here using the canonical rule, so every downstream
    record can be traced back to one detection.
    """

    detections = []

    with boxes_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                row = json.loads(line)
                if "bbox_xyxy" not in row:
                    raise ValueError("Missing bbox_xyxy")
                if "frame" not in row:
                    raise ValueError("Missing frame")
                if "track_id" not in row:
                    raise ValueError("Missing track_id")

                detections.append(row)

            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid detection record at line {line_number}: {exc}"
                ) from exc

    return assign_detection_ids(detections)


def filter_detections(detections, policy: CropFilterPolicy, frame_w: int, frame_h: int):
    """Filter detections using the unified policy and group accepted rows by tracker ID."""

    stats = FilterStats()
    by_track = defaultdict(list)

    for row in detections:
        if not isinstance(row, dict) or "bbox_xyxy" not in row or "frame" not in row or "track_id" not in row:
            stats.record(FilterResult(is_valid=False, reasons=[REASON_INVALID_METADATA]))
            continue

        result = policy.evaluate_box(
            row["bbox_xyxy"],
            row.get("detector_confidence", 0.0),
            frame_w,
            frame_h,
        )
        stats.record(result)

        if not result.is_valid:
            continue

        by_track[row["track_id"]].append(row)

    return by_track, stats


def select_track_samples(by_track, max_per_track):
    """Select evenly distributed detections from each track."""

    selected = []

    for track_id, track_rows in by_track.items():
        track_rows.sort(key=lambda row: int(row["frame"]))

        count = min(max_per_track, len(track_rows))

        if count == len(track_rows):
            chosen = track_rows

        elif count == 1:
            chosen = [track_rows[len(track_rows) // 2]]

        else:
            indices = np.linspace(
                0,
                len(track_rows) - 1,
                count
            ).round().astype(int)

            chosen = [
                track_rows[int(index)]
                for index in sorted(set(indices))
            ]

        selected.extend(chosen)

    selected.sort(
        key=lambda row: (
            int(row["frame"]),
            int(row["track_id"])
        )
    )

    return selected


def extract_and_save_crops(
    cap,
    selected,
    crops_dir,
    policy: CropFilterPolicy,
):
    """Extract torso crops while retaining only one video frame in memory."""

    saved = []
    crop_rejections = Counter()

    cached_frame_no = None
    cached_frame = None

    for row in selected:
        frame_no = int(row["frame"])

        # Decode a new frame only when the requested frame changes.
        if frame_no != cached_frame_no:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)

            ok, decoded_frame = cap.read()

            cached_frame = decoded_frame if ok else None
            cached_frame_no = frame_no

        frame = cached_frame

        if frame is None:
            crop_rejections["frame_read_failed"] += 1
            continue

        crop = extract_torso_crop(
            frame,
            row["bbox_xyxy"]
        )

        crop_res = policy.evaluate_torso_crop(crop)
        if not crop_res.is_valid:
            for r in crop_res.reasons:
                crop_rejections[r] += 1
            continue

        track_id = int(row["track_id"])

        filename = (
            f"track_{track_id:04d}_"
            f"frame_{frame_no:06d}.jpg"
        )

        crop_path = crops_dir / filename

        if not cv2.imwrite(str(crop_path), crop):
            crop_rejections["crop_write_failed"] += 1
            continue

        x1, y1, x2, y2 = map(float, row["bbox_xyxy"])

        saved.append({
            "file": str(crop_path),
            "detection_id": row.get("detection_id"),
            "track_id": track_id,
            "frame": frame_no,
            "timestamp_s": float(
                row.get("timestamp_s", 0.0)
            ),
            "detector_confidence": float(
                row.get("detector_confidence", 0.0)
            ),
            "box_width": round(x2 - x1, 2),
            "box_height": round(y2 - y1, 2),
            "crop_width": int(crop.shape[1]),
            "crop_height": int(crop.shape[0]),
        })

    return saved, crop_rejections


def generate_contact_sheets(saved, sheets_dir, batch_size):
    """Create contact sheets from saved crops."""

    thumb_w = 220
    thumb_h = 150
    cols = 4

    rows_per_sheet = math.ceil(batch_size / cols)
    sheet_capacity = rows_per_sheet * cols

    sheet_paths = []

    for batch_no, start in enumerate(
        range(0, len(saved), sheet_capacity),
        start=1
    ):
        batch = saved[start:start + sheet_capacity]

        rows_count = math.ceil(len(batch) / cols)

        sheet = np.full(
            (
                rows_count * (thumb_h + 34),
                cols * thumb_w,
                3
            ),
            245,
            dtype=np.uint8
        )

        for index, item in enumerate(batch):
            image = cv2.imread(item["file"])

            if image is None:
                continue

            height, width = image.shape[:2]

            scale = min(
                thumb_w / width,
                thumb_h / height
            )

            resized = cv2.resize(
                image,
                (
                    max(1, int(width * scale)),
                    max(1, int(height * scale))
                ),
                interpolation=cv2.INTER_AREA
            )

            x = (index % cols) * thumb_w
            y = (index // cols) * (thumb_h + 34)

            px = x + (thumb_w - resized.shape[1]) // 2
            py = y + (thumb_h - resized.shape[0]) // 2

            sheet[
                py:py + resized.shape[0],
                px:px + resized.shape[1]
            ] = resized

            label = (
                f"T{item['track_id']} "
                f"F{item['frame']} "
                f"C{item['detector_confidence']:.2f}"
            )

            cv2.putText(
                sheet,
                label,
                (x + 5, y + thumb_h + 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (15, 15, 15),
                1,
                cv2.LINE_AA
            )

        sheet_path = (
            sheets_dir / f"contact_{batch_no:03d}.jpg"
        )

        if not cv2.imwrite(str(sheet_path), sheet):
            raise RuntimeError(
                f"Could not write contact sheet: {sheet_path}"
            )

        sheet_paths.append(str(sheet_path))

    return sheet_paths


def main():
    args = parse_args()

    video_path = Path(args.video)
    boxes_path = Path(args.boxes)
    out = Path(args.out)

    if not video_path.is_file():
        raise FileNotFoundError(
            f"Video file not found: {video_path}"
        )

    if not boxes_path.is_file():
        raise FileNotFoundError(
            f"Detection file not found: {boxes_path}"
        )

    crops_dir = out / "crops"
    sheets_dir = out / "contact_sheets"

    crops_dir.mkdir(parents=True, exist_ok=True)
    sheets_dir.mkdir(parents=True, exist_ok=True)

    detections = load_detections(boxes_path)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open video: {video_path}"
        )

    try:
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        frame_area = frame_w * frame_h

        if frame_area <= 0:
            raise RuntimeError("Invalid video dimensions")

        policy = policy_from_args(args)

        by_track, filter_stats = filter_detections(
            detections,
            policy,
            frame_w,
            frame_h,
        )

        selected = select_track_samples(
            by_track,
            args.max_per_track
        )

        saved, crop_rejections = extract_and_save_crops(
            cap,
            selected,
            crops_dir,
            policy,
        )

    finally:
        cap.release()

    sheet_paths = generate_contact_sheets(
        saved,
        sheets_dir,
        args.batch_size
    )

    report = {
        "video": str(video_path),
        "boxes": str(boxes_path),
        "schema_version": SCHEMA_VERSION,
        "video_width": frame_w,
        "video_height": frame_h,
        "detections_read": len(detections),
        "tracks_with_quality_candidates": len(by_track),
        "tracks_sampled": len({
            item["track_id"] for item in selected
        }),
        "selected_before_crop_validation": len(selected),
        "saved_crops": len(saved),
        "filter_policy": policy.to_dict(),
        "filter_stats": filter_stats.to_dict(),
        "rejected_by_detection_quality": dict(filter_stats.rejection_reasons_overlapping),
        "unique_rejected_detections": filter_stats.unique_rejected,
        "unique_accepted_detections": filter_stats.unique_accepted,
        "rejected_during_crop_extraction": dict(crop_rejections),
        "settings": {
            "max_per_track": args.max_per_track,
            "batch_size": args.batch_size,
            **policy.to_dict(),
        },
        "output_contract": schema_descriptor(),
        "input_schema_validation": validate_rows(detections, strict=True),
        "contact_sheets": sheet_paths,
        "samples": saved,
        "warning": (
            "Review crops manually. Quality filters do not prove "
            "that a box contains an outfield player. Goalkeepers, "
            "referees, staff, and camera-cut artifacts may remain."
        ),
    }

    report_path = out / "crop_index.json"

    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8"
    )

    print("\n========== CROP REVIEW SUMMARY ==========")
    print("Detections read:", len(detections))
    print("Unique detections accepted:", filter_stats.unique_accepted)
    print("Unique detections rejected:", filter_stats.unique_rejected)
    print("Tracks with quality candidates:", len(by_track))
    print("Tracks sampled:", report["tracks_sampled"])
    print("Selected before crop validation:", len(selected))
    print("Crops saved:", len(saved))

    print(
        "Overlapping rejection reasons:",
        dict(filter_stats.rejection_reasons_overlapping)
    )
    print(
        "Primary rejection reasons:",
        dict(filter_stats.primary_rejection_reasons)
    )

    print(
        "Rejected during crop extraction:",
        dict(crop_rejections)
    )

    print("\nContact sheets:")

    for path in sheet_paths:
        print(" ", path)

    print("\nCrop index:", report_path)
    print("=========================================\n")


if __name__ == "__main__":
    main()
