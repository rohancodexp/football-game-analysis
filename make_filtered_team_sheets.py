
import json
import math
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np


INPUT_INDEX = Path(
    r"results\1_team_crop_review_v2\crop_index.json"
)

OUTPUT_DIR = Path(
    r"results\1_team_filtered_review"
)

MIN_CONFIDENCE = 0.45
MIN_BOX_HEIGHT = 50
MAX_BOX_HEIGHT = 160
MAX_AREA_FRACTION = 0.025
MAX_ASPECT_RATIO = 0.90

MAX_PER_TRACK = 2

THUMB_WIDTH = 180
THUMB_HEIGHT = 140
COLUMNS = 4
ROWS_PER_SHEET = 8


def main():
    if not INPUT_INDEX.exists():
        raise FileNotFoundError(
            f"Crop index not found: {INPUT_INDEX}"
        )

    with INPUT_INDEX.open(encoding="utf-8") as f:
        report = json.load(f)

    samples = report["samples"]

    frame_width = report["video_width"]
    frame_height = report["video_height"]
    frame_area = frame_width * frame_height

    rejected = defaultdict(int)
    candidates = []

    for sample in samples:
        width = float(sample["box_width"])
        height = float(sample["box_height"])
        confidence = float(
            sample["detector_confidence"]
        )

        area_fraction = (
            width * height / frame_area
            if frame_area > 0 else 1.0
        )

        aspect_ratio = (
            width / height if height > 0 else 999
        )

        reasons = []

        if confidence < MIN_CONFIDENCE:
            reasons.append("low_confidence")

        if height < MIN_BOX_HEIGHT:
            reasons.append("too_short")

        if height > MAX_BOX_HEIGHT:
            reasons.append("too_large")

        if area_fraction > MAX_AREA_FRACTION:
            reasons.append("large_area")

        if aspect_ratio > MAX_ASPECT_RATIO:
            reasons.append("wide_box")

        if reasons:
            for reason in reasons:
                rejected[reason] += 1
            continue

        candidates.append(sample)

    # Group accepted crops by track.
    by_track = defaultdict(list)

    for sample in candidates:
        by_track[int(sample["track_id"])].append(sample)

    # Select samples evenly across each track's duration.
    selected = []

    for track_id, track_samples in by_track.items():
        track_samples.sort(
            key=lambda x: int(x["frame"])
        )

        count = min(
            MAX_PER_TRACK,
            len(track_samples)
        )

        if count == len(track_samples):
            chosen = track_samples

        elif count == 1:
            chosen = [
                track_samples[len(track_samples) // 2]
            ]

        else:
            indices = np.linspace(
                0,
                len(track_samples) - 1,
                count
            ).round().astype(int)

            chosen = [
                track_samples[int(i)]
                for i in sorted(set(indices))
            ]

        selected.extend(chosen)

    selected.sort(
        key=lambda x: (
            int(x["frame"]),
            int(x["track_id"])
        )
    )

    sheets_dir = OUTPUT_DIR / "contact_sheets"
    sheets_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    missing = 0

    for sample in selected:
        crop_path = Path(sample["file"])

        image = cv2.imread(str(crop_path))

        if image is None:
            missing += 1
            continue

        saved.append({
            **sample,
            "review_crop": str(crop_path)
        })

    capacity = COLUMNS * ROWS_PER_SHEET
    sheet_paths = []

    for sheet_no, start in enumerate(
        range(0, len(saved), capacity),
        start=1
    ):
        batch = saved[start:start + capacity]

        sheet = np.full(
            (
                ROWS_PER_SHEET * (THUMB_HEIGHT + 35),
                COLUMNS * THUMB_WIDTH,
                3
            ),
            245,
            dtype=np.uint8
        )

        for index, item in enumerate(batch):
            image = cv2.imread(item["review_crop"])

            if image is None:
                continue

            h, w = image.shape[:2]

            scale = min(
                THUMB_WIDTH / w,
                THUMB_HEIGHT / h
            )

            resized = cv2.resize(
                image,
                (
                    max(1, int(w * scale)),
                    max(1, int(h * scale))
                ),
                interpolation=cv2.INTER_AREA
            )

            col = index % COLUMNS
            row = index // COLUMNS

            x = col * THUMB_WIDTH
            y = row * (THUMB_HEIGHT + 35)

            px = x + (
                THUMB_WIDTH - resized.shape[1]
            ) // 2

            py = y + (
                THUMB_HEIGHT - resized.shape[0]
            ) // 2

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
                (x + 4, y + THUMB_HEIGHT + 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.43,
                (15, 15, 15),
                1,
                cv2.LINE_AA
            )

        path = (
            sheets_dir /
            f"filtered_{sheet_no:03d}.jpg"
        )

        if not cv2.imwrite(str(path), sheet):
            raise RuntimeError(
                f"Could not save contact sheet: {path}"
            )

        sheet_paths.append(str(path))

    result = {
        "source_index": str(INPUT_INDEX),
        "original_samples": len(samples),
        "quality_candidates": len(candidates),
        "selected_after_track_balancing": len(selected),
        "saved_review_samples": len(saved),
        "missing_crop_files": missing,
        "unique_tracks": len(by_track),
        "rejected_by_filter": dict(rejected),
        "settings": {
            "min_confidence": MIN_CONFIDENCE,
            "min_box_height": MIN_BOX_HEIGHT,
            "max_box_height": MAX_BOX_HEIGHT,
            "max_area_fraction": MAX_AREA_FRACTION,
            "max_aspect_ratio": MAX_ASPECT_RATIO,
            "max_per_track": MAX_PER_TRACK
        },
        "contact_sheets": sheet_paths,
        "samples": saved
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    index_path = OUTPUT_DIR / "filtered_index.json"

    index_path.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8"
    )

    print("\nFILTERED TEAM REVIEW SUMMARY")
    print("----------------------------")
    print("Original samples:", len(samples))
    print("Quality candidates:", len(candidates))
    print("Selected after balancing:", len(selected))
    print("Saved review samples:", len(saved))
    print("Unique tracks:", len(by_track))
    print("Missing crop files:", missing)

    print("\nRejections:")
    for reason, count in sorted(rejected.items()):
        print(f"{reason}: {count}")

    print("\nContact sheets:")
    for path in sheet_paths:
        print(path)

    print("\nIndex:", index_path)


if __name__ == "__main__":
    main()
