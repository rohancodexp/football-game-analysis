"""Automatic, UNVALIDATED persistent anonymous player IDs for one finished run.

Post-processing only. Reads an existing run directory (fragments.json, identified_boxes.jsonl)
and writes NEW files into a separate, new output directory. It never edits the run directory,
never changes detection, tracking, fragments, embeddings, player_id, team_id or person_uid,
and is not part of the human-reviewed association path in identity_association.py.

Each fragment gets exactly one anonymous persistent_id (ID001, ID002, ...). Fragments are
grouped by conservative greedy linking:

* continuity link: fragment B starts within --max-gap frames after fragment A ends and B's
  first box is close to A's last box (centre distance <= --max-center-dist x box height,
  height ratio within --max-height-ratio);
* appearance link: both fragments have ResNet18 embeddings with cosine >= --min-cosine.

A merge of two groups is refused if any two members overlap in time (one person cannot be
in two places), if their team labels disagree, or (appearance links only) if any embedded
cross-pair is below --min-cosine (no weak chaining).

Team per ID. A fragment's team label is its calibrated team_id when that is TEAM_A/TEAM_B,
otherwise its uncalibrated kit-cluster proposal (team_classification.team_id_before_reliability),
otherwise none. Because merges with disagreeing labels are refused, all labelled members of an
ID carry the same label, which becomes persistent_team_id (UNKNOWN when no member is labelled).
persistent_team_status records where that label came from:
  calibrated                        every labelled member is calibrated
  uncalibrated_proposal             every labelled member is an uncalibrated proposal
  mixed_calibrated_and_uncalibrated both kinds agree on the label
  none                              no member is labelled

Malformed inputs never produce a link: an embedding with the wrong length, non-finite values
or zero norm is ignored (that fragment can still link by continuity); a box that is not four
finite numbers with x2 > x1 and y2 > y1 is not used for continuity. Structural errors (bad
JSON, missing or duplicate fragment ids, inverted intervals, rows that reference an unknown
fragment, non-integer frames) stop the tool before anything is written.

No ground truth exists for these clips, so no accuracy is claimed. These are unvalidated
anonymous IDs, not verified real-world player identities. ImageNet ResNet18 is not a football
Re-ID model; expect wrong merges between similar kits and new IDs after camera cuts.
"""
import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

TOOL_VERSION = "persistent_ids_v2"
METHOD = "auto_continuity_appearance_unvalidated"
IDENTITY_TYPE = "anonymous_unvalidated"
TEAMS = ("TEAM_A", "TEAM_B")
DEFAULTS = {"max_gap": 45, "max_center_dist": 1.0, "max_height_ratio": 1.5, "min_cosine": 0.88}
TEAM_STATUSES = ("calibrated", "uncalibrated_proposal", "mixed_calibrated_and_uncalibrated", "none")


class InputError(ValueError):
    """Structural problem in the run directory; nothing is written."""


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_params(params):
    """Return a complete, checked parameter dict or raise ValueError."""
    p = {**DEFAULTS, **(params or {})}
    unknown = set(p) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"unknown parameters: {sorted(unknown)}")
    gap = p["max_gap"]
    if isinstance(gap, bool) or not isinstance(gap, int) or gap < 1:
        raise ValueError("max_gap must be an integer >= 1 frame")
    for key in ("max_center_dist", "max_height_ratio", "min_cosine"):
        value = p[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{key} must be a finite number")
        p[key] = float(value)
    if p["max_center_dist"] <= 0:
        raise ValueError("max_center_dist must be > 0 (in box heights)")
    if p["max_height_ratio"] < 1:
        raise ValueError("max_height_ratio must be >= 1")
    if not 0 < p["min_cosine"] <= 1:
        raise ValueError("min_cosine must be in (0, 1]")
    return p


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def load_fragments(run_dir):
    path = Path(run_dir) / "fragments.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InputError(f"cannot read {path}: {exc}") from exc
    fragments = data if isinstance(data, list) else (data.get("fragments") if isinstance(data, dict) else None)
    if not isinstance(fragments, list):
        raise InputError("fragments.json must be a list of fragments (or {'fragments': [...]})")
    return fragments


def check_fragments(fragments):
    seen = set()
    for i, f in enumerate(fragments):
        if not isinstance(f, dict):
            raise InputError(f"fragment #{i} is not an object")
        fid = f.get("fragment_id")
        if not isinstance(fid, str) or not fid:
            raise InputError(f"fragment #{i} has no fragment_id")
        if fid in seen:
            raise InputError(f"duplicate fragment_id {fid}")
        seen.add(fid)
        start, end = f.get("start_frame"), f.get("end_frame")
        if not _is_int(start) or not _is_int(end) or start < 0:
            raise InputError(f"{fid}: start_frame/end_frame must be non-negative integers")
        if end < start:
            raise InputError(f"inverted fragment interval: {fid}")
    return seen


def valid_box(box):
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return False
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in box):
        return False
    x1, y1, x2, y2 = box
    return x2 > x1 and y2 > y1


def read_rows(rows_path, fragment_ids):
    """Load rows, check every row references a known fragment and has an integer frame."""
    rows = []
    with open(rows_path, encoding="utf-8") as handle:
        for n, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise InputError(f"identified_boxes.jsonl line {n}: invalid JSON ({exc.msg})") from exc
            if not isinstance(row, dict):
                raise InputError(f"identified_boxes.jsonl line {n}: not an object")
            if row.get("fragment_id") not in fragment_ids:
                raise InputError(f"identified_boxes.jsonl line {n}: unknown fragment_id {row.get('fragment_id')!r}")
            if not _is_int(row.get("frame")) or row["frame"] < 0:
                raise InputError(f"identified_boxes.jsonl line {n}: frame must be a non-negative integer")
            rows.append(row)
    return rows


def fragment_endpoints(rows):
    """First and last VALID bbox (by frame) of every fragment, plus detection counts."""
    ends, invalid_boxes = {}, 0
    for row in rows:
        fid = row["fragment_id"]
        entry = ends.setdefault(fid, {"first": None, "last": None, "detections": 0})
        entry["detections"] += 1
        box = row.get("bbox_xyxy")
        if not valid_box(box):
            invalid_boxes += 1
            continue
        frame = row["frame"]
        if entry["first"] is None or frame < entry["first"][0]:
            entry["first"] = (frame, box)
        if entry["last"] is None or frame >= entry["last"][0]:
            entry["last"] = (frame, box)
    return ends, invalid_boxes


def team_proposal(fragment):
    """(label, source): calibrated team_id, else uncalibrated proposal, else (None, 'none')."""
    if fragment.get("team_id") in TEAMS:
        return fragment["team_id"], "calibrated"
    tc = fragment.get("team_classification")
    proposal = tc.get("team_id_before_reliability") if isinstance(tc, dict) else None
    if proposal in TEAMS:
        return proposal, "uncalibrated_proposal"
    return None, "none"


def team_status(sources):
    kinds = set(sources) - {"none"}
    if not kinds:
        return "none"
    if kinds == {"calibrated"}:
        return "calibrated"
    if kinds == {"uncalibrated_proposal"}:
        return "uncalibrated_proposal"
    return "mixed_calibrated_and_uncalibrated"


def clean_embeddings(fragments):
    """Unit vectors for well-formed embeddings; malformed ones are ignored and counted."""
    raw, problems = {}, Counter()
    for f in fragments:
        vec = f.get("embedding")
        if vec is None:
            continue
        if not isinstance(vec, (list, tuple)) or not vec or not all(
                isinstance(v, (int, float)) and not isinstance(v, bool) for v in vec):
            problems["not_a_numeric_list"] += 1
            continue
        arr = np.asarray(vec, dtype=np.float64)
        if not np.all(np.isfinite(arr)):
            problems["non_finite"] += 1
            continue
        norm = float(np.linalg.norm(arr))
        if norm == 0.0:
            problems["zero_norm"] += 1
            continue
        raw[f["fragment_id"]] = arr / norm
    dims = Counter(v.size for v in raw.values())
    expected = max(dims.items(), key=lambda kv: (kv[1], kv[0]))[0] if dims else None
    emb = {}
    for fid, v in raw.items():
        if v.size != expected:
            problems["wrong_dimension"] += 1
        else:
            emb[fid] = v
    return emb, expected, dict(problems)


def _centre_height(box):
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0, y2 - y1


def continuity(a, b, ends, params):
    """Continuity evidence that b continues a (a ends before b starts), or None."""
    gap = b["start_frame"] - a["end_frame"]
    if gap < 1 or gap > params["max_gap"]:
        return None
    ea, eb = ends.get(a["fragment_id"]), ends.get(b["fragment_id"])
    if not ea or not eb or ea["last"] is None or eb["first"] is None:
        return None
    ax, ay, ah = _centre_height(ea["last"][1])
    bx, by, bh = _centre_height(eb["first"][1])
    ratio = max(ah, bh) / min(ah, bh)
    dist = math.hypot(ax - bx, ay - by) / max(ah, bh)
    if ratio > params["max_height_ratio"] or dist > params["max_center_dist"]:
        return None
    return {"gap_frames": gap, "center_dist_over_height": round(dist, 4), "height_ratio": round(ratio, 4)}


def assign(fragments, ends, params=None):
    """Return (assignment, groups, links, refused, params, input_issues). Pure and deterministic."""
    params = validate_params(params)
    check_fragments(fragments)
    frags = sorted(fragments, key=lambda f: (f["start_frame"], f["fragment_id"]))
    ids = [f["fragment_id"] for f in frags]
    emb, dim, emb_problems = clean_embeddings(frags)
    team = {f["fragment_id"]: team_proposal(f) for f in frags}
    span = {f["fragment_id"]: (f["start_frame"], f["end_frame"]) for f in frags}

    candidates = []
    for i, a in enumerate(frags):
        for b in frags[i + 1:]:
            if b["start_frame"] <= a["end_frame"]:
                continue  # overlapping in time: never the same person
            ta, tb = team[a["fragment_id"]][0], team[b["fragment_id"]][0]
            if ta and tb and ta != tb:
                continue
            cont = continuity(a, b, ends, params)
            if cont:
                candidates.append((0, cont["gap_frames"] + cont["center_dist_over_height"], a["fragment_id"],
                                   b["fragment_id"], "continuity", cont))
                continue
            ua, ub = emb.get(a["fragment_id"]), emb.get(b["fragment_id"])
            if ua is not None and ub is not None:
                cos = float(ua @ ub)
                if cos >= params["min_cosine"]:
                    candidates.append((1, -cos, a["fragment_id"], b["fragment_id"], "appearance",
                                       {"cosine": round(cos, 4)}))
    candidates.sort(key=lambda c: (c[0], c[1], c[2], c[3]))

    parent = {fid: fid for fid in ids}
    members = {fid: [fid] for fid in ids}

    def root(fid):
        while parent[fid] != fid:
            parent[fid] = parent[parent[fid]]
            fid = parent[fid]
        return fid

    links, refused = [], Counter()
    for _, _, left, right, kind, evidence in candidates:
        ra, rb = root(left), root(right)
        if ra == rb:
            continue
        ma, mb = members[ra], members[rb]
        reason = None
        if any(not (span[x][1] < span[y][0] or span[y][1] < span[x][0]) for x in ma for y in mb):
            reason = "temporal_overlap"
        elif len({team[x][0] for x in ma + mb if team[x][0]}) > 1:
            reason = "team_conflict"
        elif kind == "appearance" and any(
                x in emb and y in emb and float(emb[x] @ emb[y]) < params["min_cosine"] for x in ma for y in mb):
            reason = "weak_appearance_pair"
        if reason:
            refused[reason] += 1
            continue
        keep, drop = (ra, rb) if ra < rb else (rb, ra)
        parent[drop] = keep
        members[keep] = sorted(members[keep] + members.pop(drop))
        links.append({"left": left, "right": right, "kind": kind, **evidence})

    groups = sorted((sorted(m) for r, m in members.items() if root(r) == r),
                    key=lambda m: (min(span[x][0] for x in m), m[0]))
    assignment, out_groups = {}, []
    for n, group in enumerate(groups, 1):
        pid = f"ID{n:03d}"
        labels = {team[fid][0] for fid in group if team[fid][0]}
        if len(labels) > 1:  # cannot happen: merges with disagreeing labels are refused
            raise AssertionError(f"{pid} mixes team labels {sorted(labels)}")
        group_team = labels.pop() if labels else "UNKNOWN"
        sources = [team[fid][1] for fid in group]
        status = team_status(sources)
        out_groups.append({"persistent_id": pid, "fragments": group, "team_id": group_team,
                           "team_status": status,
                           "calibrated_fragments": sources.count("calibrated"),
                           "uncalibrated_fragments": sources.count("uncalibrated_proposal"),
                           "start_frame": min(span[f][0] for f in group),
                           "end_frame": max(span[f][1] for f in group)})
        for fid in group:
            assignment[fid] = {"persistent_id": pid, "persistent_team_id": group_team,
                               "persistent_team_status": status}
    issues = {"embedding_dimension": dim, "embeddings_used": len(emb), "embeddings_ignored": emb_problems}
    return assignment, out_groups, links, dict(refused), params, issues


def render(video, rows, out_path, assignment):
    import cv2
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    width, height = int(cap.get(3)), int(cap.get(4))
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError("MP4 writer unavailable")
    colours = {"TEAM_A": (255, 120, 0), "TEAM_B": (0, 0, 230), "UNKNOWN": (0, 165, 255)}
    frames, i = 0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        while i < len(rows) and rows[i]["frame"] == frames:
            row = rows[i]
            i += 1
            if not valid_box(row.get("bbox_xyxy")):
                continue
            info = assignment[row["fragment_id"]]
            x1, y1, x2, y2 = map(int, row["bbox_xyxy"])
            colour = colours.get(info["persistent_team_id"], colours["UNKNOWN"])
            team_tag = {"TEAM_A": "A", "TEAM_B": "B"}.get(info["persistent_team_id"], "?")
            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
            cv2.putText(frame, f"{info['persistent_id']} {team_tag}", (x1, max(15, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, .5, colour, 2)
        writer.write(frame)
        frames += 1
    cap.release()
    writer.release()
    return frames, len(rows) - i


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run-dir", required=True, help="finished run.py output directory (read only)")
    ap.add_argument("--out", required=True, help="new or empty output directory")
    ap.add_argument("--video", help="source video; with --render draws identified_persistent.mp4")
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--max-gap", type=int, default=DEFAULTS["max_gap"])
    ap.add_argument("--max-center-dist", type=float, default=DEFAULTS["max_center_dist"])
    ap.add_argument("--max-height-ratio", type=float, default=DEFAULTS["max_height_ratio"])
    ap.add_argument("--min-cosine", type=float, default=DEFAULTS["min_cosine"])
    a = ap.parse_args(argv)
    params = {"max_gap": a.max_gap, "max_center_dist": a.max_center_dist,
              "max_height_ratio": a.max_height_ratio, "min_cosine": a.min_cosine}
    try:
        params = validate_params(params)
    except ValueError as exc:
        ap.error(str(exc))
    run_dir, out = Path(a.run_dir), Path(a.out)
    frag_path, rows_path = run_dir / "fragments.json", run_dir / "identified_boxes.jsonl"
    for needed in (frag_path, rows_path):
        if not needed.is_file():
            ap.error(f"missing {needed}")
    if a.render and not (a.video and Path(a.video).is_file()):
        ap.error("--render needs --video pointing at the source video")
    if out.resolve() == run_dir.resolve():
        ap.error("--out must differ from --run-dir; run outputs are never modified")
    if out.exists() and any(out.iterdir()):
        ap.error("Output directory must be new or empty; existing evidence will not be overwritten")

    inputs = {"fragments_json_sha256": file_sha256(frag_path), "identified_boxes_sha256": file_sha256(rows_path)}
    try:
        fragments = load_fragments(run_dir)
        fragment_ids = check_fragments(fragments)
        rows = read_rows(rows_path, fragment_ids)
        ends, invalid_boxes = fragment_endpoints(rows)
        assignment, groups, links, refused, params, issues = assign(fragments, ends, params)
    except (InputError, ValueError) as exc:
        raise SystemExit(f"Input rejected, nothing written: {exc}")

    input_frame_ordered = all(rows[i]["frame"] <= rows[i + 1]["frame"] for i in range(len(rows) - 1))
    order = sorted(range(len(rows)), key=lambda i: (rows[i]["frame"], i))  # stable: keeps per-frame order
    rows = [rows[i] for i in order]
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "persistent_boxes.jsonl", "w", encoding="utf-8") as dst:
        for row in rows:
            row.update(assignment[row["fragment_id"]])
            dst.write(json.dumps(row) + "\n")
    output_checks = {
        "every_row_has_assignment": all(r["fragment_id"] in assignment for r in rows),
        "every_fragment_has_one_id": len(assignment) == len(fragments),
        "output_rows_frame_ordered": all(rows[i]["frame"] <= rows[i + 1]["frame"] for i in range(len(rows) - 1)),
        "input_rows_frame_ordered": input_frame_ordered,
        "rows_in": len(rows), "rows_out": len(rows),
    }

    multi = [g for g in groups if len(g["fragments"]) > 1]
    report = {
        "tool_version": TOOL_VERSION, "method": METHOD, "identity_type": IDENTITY_TYPE,
        "parameters": params, "source_run_dir": str(run_dir), "inputs": inputs,
        "counts": {"fragments": len(fragments), "rows": len(rows), "persistent_ids": len(groups),
                   "multi_fragment_ids": len(multi),
                   "fragments_in_multi_fragment_ids": sum(len(g["fragments"]) for g in multi),
                   "links": len(links),
                   "links_by_kind": {k: sum(1 for l in links if l["kind"] == k) for k in ("continuity", "appearance")},
                   "refused_merges": refused,
                   "ids_by_team": dict(sorted(Counter(g["team_id"] for g in groups).items())),
                   "ids_by_team_status": {s: sum(1 for g in groups if g["team_status"] == s) for s in TEAM_STATUSES}},
        "input_issues": {**issues, "rows_with_invalid_bbox": invalid_boxes},
        "output_checks": output_checks,
        "accuracy_claim": None,
        "semantics": ("persistent_id is an automatic, unvalidated anonymous grouping of fragments within this one run. "
                      "It is not a verified real-world player identity, does not change player_id/team_id/person_uid, "
                      "and is not human-reviewed. persistent_team_status says whether persistent_team_id comes from "
                      "calibrated team_id, uncalibrated kit-cluster proposals, both (agreeing), or none. No ground truth "
                      "was used; expect wrong merges between similar kits and new IDs after camera cuts."),
        "groups": groups, "links": links,
    }
    if a.render:
        frames, unrendered = render(a.video, rows, out / "identified_persistent.mp4", assignment)
        report["render"] = {"file": "identified_persistent.mp4", "frames_written": frames,
                            "rows_beyond_last_frame": unrendered, "input_video_sha256": file_sha256(a.video)}
    (out / "persistent_ids.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    c = report["counts"]
    print(f"Persistent IDs: {c['persistent_ids']} IDs for {c['fragments']} fragments "
          f"({c['multi_fragment_ids']} IDs span >1 fragment; links={c['links_by_kind']}; "
          f"refused={c['refused_merges']}; teams={c['ids_by_team']}). "
          f"Unvalidated anonymous IDs, no accuracy claim.")
    print(f"Checks: {output_checks}; input issues: {report['input_issues']}")
    print(f"Outputs in {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
