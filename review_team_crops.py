"""Create a review set of torso crops from an existing pipeline run.

Example:
  python review_team_crops.py --video datasets/.../1.mp4 \
      --boxes results/1_out_baseline/boxes.jsonl \
      --out results/1_team_crop_review

This is a visual inspection tool, not a team classifier or ground-truth labeler.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from team_classifier import extract_torso_crop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--boxes", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-per-track", type=int, default=2)
    ap.add_argument("--max-box-area-fraction", type=float, default=0.08)
    args = ap.parse_args()

    out = Path(args.out)
    crops_dir = out / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_area = frame_w * frame_h
    if frame_area <= 0:
        raise RuntimeError("Invalid video dimensions")

    rows = []
    with open(args.boxes, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    # Keep detections distributed across each track's duration.
    by_track = defaultdict(list)
    rejected_large = 0
    for row in rows:
        x1, y1, x2, y2 = row["bbox_xyxy"]
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        if area / frame_area > args.max_box_area_fraction:
            rejected_large += 1
            continue
        by_track[row["track_id"]].append(row)

    selected = []
    for track_id, track_rows in by_track.items():
        track_rows.sort(key=lambda r: r["frame"])
        if len(track_rows) <= args.max_per_track:
            chosen = track_rows
        else:
            indices = [
                round(i * (len(track_rows) - 1) / (args.max_per_track - 1))
                for i in range(args.max_per_track)
            ] if args.max_per_track > 1 else [len(track_rows) // 2]
            chosen = [track_rows[i] for i in sorted(set(indices))]
        selected.extend(chosen)

    selected.sort(key=lambda r: (r["frame"], r["track_id"]))
    saved = []
    frame_cache = {}
    for row in selected:
        frame_no = int(row["frame"])
        if frame_no not in frame_cache:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            ok, frame = cap.read()
            frame_cache[frame_no] = frame if ok else None
        frame = frame_cache[frame_no]
        if frame is None:
            continue

        crop = extract_torso_crop(frame, row["bbox_xyxy"])
        if crop is None:
            continue

        filename = f"track_{int(row['track_id']):04d}_frame_{frame_no:06d}.jpg"
        path = crops_dir / filename
        cv2.imwrite(str(path), crop)
        saved.append({
            "file": str(path),
            "track_id": int(row["track_id"]),
            "frame": frame_no,
            "timestamp_s": float(row.get("timestamp_s", 0.0)),
            "detector_confidence": float(row.get("detector_confidence", 0.0)),
            "crop_width": int(crop.shape[1]),
            "crop_height": int(crop.shape[0]),
        })

    cap.release()
    (out / "crop_index.json").write_text(
        json.dumps({
            "video": args.video,
            "boxes": args.boxes,
            "video_width": frame_w,
            "video_height": frame_h,
            "detections_read": len(rows),
            "tracks_seen": len(by_track),
            "selected_detections": len(selected),
            "saved_crops": len(saved),
            "rejected_large_boxes": rejected_large,
            "max_box_area_fraction": args.max_box_area_fraction,
            "samples": saved,
            "warning": "Review crops manually. This tool does not assign teams or guarantee player-only crops."
        }, indent=2),
        encoding="utf-8",
    )

    # Contact sheet: thumbnails with track and frame identifiers.
    thumb_w, thumb_h = 150, 120
    cols = 6
    rows_count = max(1, (len(saved) + cols - 1) // cols)
    sheet = np.full((rows_count * (thumb_h + 28), cols * thumb_w, 3), 255, dtype=np.uint8)
    for i, item in enumerate(saved):
        image = cv2.imread(item["file"])
        if image is None:
            continue
        image = cv2.resize(image, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
        x = (i % cols) * thumb_w
        y = (i // cols) * (thumb_h + 28)
        sheet[y:y + thumb_h, x:x + thumb_w] = image
        label = f"T{item['track_id']} F{item['frame']}"
        cv2.putText(sheet, label, (x + 3, y + thumb_h + 19),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

    contact_path = out / "contact_sheet.jpg"
    cv2.imwrite(str(contact_path), sheet)
    print("Detections read:", len(rows))
    print("Tracks sampled:", len(by_track))
    print("Large boxes rejected:", rejected_large)
    print("Torso crops saved:", len(saved))
    print("Contact sheet:", contact_path)
    print("Crop index:", out / "crop_index.json")


if __name__ == "__main__":
    main()
