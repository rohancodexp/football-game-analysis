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
in two places), if their team proposals disagree, or (appearance links only) if any embedded
cross-pair is below --min-cosine (no weak chaining). Team per ID is the detection-weighted
majority of the members' team proposals. Without a calibration these proposals are the
uncalibrated kit-cluster proposals (team_classification.team_id_before_reliability), so the
output marks them as unvalidated.

No ground truth exists for these clips, so no accuracy is claimed. ImageNet ResNet18 is not a
football Re-ID model; expect wrong merges between similar kits and over-splitting across cuts.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

TOOL_VERSION = "persistent_ids_v1"
METHOD = "auto_continuity_appearance_unvalidated"
TEAMS = ("TEAM_A", "TEAM_B")
DEFAULTS = {"max_gap": 45, "max_center_dist": 1.0, "max_height_ratio": 1.5, "min_cosine": 0.88}


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_fragments(run_dir):
    data = json.loads((Path(run_dir) / "fragments.json").read_text(encoding="utf-8"))
    fragments = data if isinstance(data, list) else data.get("fragments")
    if not isinstance(fragments, list):
        raise ValueError("fragments.json must be a list of fragments")
    return fragments


def team_proposal(fragment):
    """Calibrated team_id if present, else the uncalibrated proposal, else None."""
    if fragment.get("team_id") in TEAMS:
        return fragment["team_id"], "calibrated"
    tc = fragment.get("team_classification") or {}
    proposal = tc.get("team_id_before_reliability")
    if proposal in TEAMS:
        return proposal, "uncalibrated_proposal"
    return None, "none"


def fragment_endpoints(rows_path):
    """First and last bbox (by frame) of every fragment, streamed."""
    ends = {}
    with open(rows_path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            fid, frame, box = row["fragment_id"], row["frame"], row["bbox_xyxy"]
            entry = ends.setdefault(fid, {"first": (frame, box), "last": (frame, box), "detections": 0})
            entry["detections"] += 1
            if frame < entry["first"][0]:
                entry["first"] = (frame, box)
            if frame >= entry["last"][0]:
                entry["last"] = (frame, box)
    return ends


def _unit(vector):
    if vector is None:
        return None
    arr = np.asarray(vector, dtype=np.float64).reshape(-1)
    norm = float(np.linalg.norm(arr))
    if arr.size == 0 or not math.isfinite(norm) or norm == 0.0:
        return None
    return arr / norm


def _centre_height(box):
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0, max(1.0, y2 - y1)


def continuity(a, b, ends, params):
    """Continuity evidence that b continues a (a ends before b starts), or None."""
    gap = b["start_frame"] - a["end_frame"]
    if gap < 1 or gap > params["max_gap"]:
        return None
    ea, eb = ends.get(a["fragment_id"]), ends.get(b["fragment_id"])
    if not ea or not eb:
        return None
    ax, ay, ah = _centre_height(ea["last"][1])
    bx, by, bh = _centre_height(eb["first"][1])
    ratio = max(ah, bh) / min(ah, bh)
    dist = math.hypot(ax - bx, ay - by) / max(ah, bh)
    if ratio > params["max_height_ratio"] or dist > params["max_center_dist"]:
        return None
    return {"gap_frames": gap, "center_dist_over_height": round(dist, 4), "height_ratio": round(ratio, 4)}


def assign(fragments, ends, params=None):
    """Return (per-fragment assignment dict, groups list, links list). Pure and deterministic."""
    params = {**DEFAULTS, **(params or {})}
    frags = sorted(fragments, key=lambda f: (f["start_frame"], f["fragment_id"]))
    ids = [f["fragment_id"] for f in frags]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate fragment_id in fragments.json")
    for f in frags:
        if f["end_frame"] < f["start_frame"]:
            raise ValueError(f"inverted fragment interval: {f['fragment_id']}")
    emb = {f["fragment_id"]: _unit(f.get("embedding")) for f in frags}
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
            ua, ub = emb[a["fragment_id"]], emb[b["fragment_id"]]
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

    links, refused = [], {}
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
                emb[x] is not None and emb[y] is not None and float(emb[x] @ emb[y]) < params["min_cosine"]
                for x in ma for y in mb):
            reason = "weak_appearance_pair"
        if reason:
            refused[reason] = refused.get(reason, 0) + 1
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
        votes = {}
        for fid in group:
            label = team[fid][0]
            if label:
                weight = ends.get(fid, {}).get("detections", 1)
                votes[label] = votes.get(label, 0) + weight
        if votes and len(votes) == 1:
            group_team = next(iter(votes))
        else:
            group_team = "UNKNOWN"
        sources = sorted({team[fid][1] for fid in group if team[fid][0]})
        team_status = ("calibrated" if sources == ["calibrated"] else
                       "uncalibrated_proposal" if sources else "none")
        out_groups.append({"persistent_id": pid, "fragments": group, "team_id": group_team,
                           "team_status": team_status,
                           "start_frame": min(span[f][0] for f in group),
                           "end_frame": max(span[f][1] for f in group)})
        for fid in group:
            assignment[fid] = {"persistent_id": pid, "persistent_team_id": group_team,
                               "persistent_team_status": team_status}
    return assignment, out_groups, links, refused, params


def render(video, rows_path, out_path, assignment):
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
    frames = 0
    with open(rows_path, encoding="utf-8") as src:
        line = src.readline()
        row = json.loads(line) if line.strip() else None
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            while row is not None and row["frame"] == idx:
                info = assignment[row["fragment_id"]]
                x1, y1, x2, y2 = map(int, row["bbox_xyxy"])
                colour = colours.get(info["persistent_team_id"], colours["UNKNOWN"])
                team_tag = {"TEAM_A": "A", "TEAM_B": "B"}.get(info["persistent_team_id"], "?")
                cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
                cv2.putText(frame, f"{info['persistent_id']} {team_tag}", (x1, max(15, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, .5, colour, 2)
                line = src.readline()
                row = json.loads(line) if line.strip() else None
            writer.write(frame)
            frames += 1
            idx += 1
    cap.release()
    writer.release()
    return frames


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
    run_dir, out = Path(a.run_dir), Path(a.out)
    rows_path = run_dir / "identified_boxes.jsonl"
    for needed in (run_dir / "fragments.json", rows_path):
        if not needed.is_file():
            ap.error(f"missing {needed}")
    if a.render and not (a.video and Path(a.video).is_file()):
        ap.error("--render needs --video pointing at the source video")
    if out.resolve() == run_dir.resolve():
        ap.error("--out must differ from --run-dir; run outputs are never modified")
    if out.exists() and any(out.iterdir()):
        ap.error("Output directory must be new or empty; existing evidence will not be overwritten")
    out.mkdir(parents=True, exist_ok=True)

    params = {"max_gap": a.max_gap, "max_center_dist": a.max_center_dist,
              "max_height_ratio": a.max_height_ratio, "min_cosine": a.min_cosine}
    inputs = {"fragments_json_sha256": file_sha256(run_dir / "fragments.json"),
              "identified_boxes_sha256": file_sha256(rows_path)}
    fragments = load_fragments(run_dir)
    ends = fragment_endpoints(rows_path)
    assignment, groups, links, refused, params = assign(fragments, ends, params)

    rows = 0
    with open(rows_path, encoding="utf-8") as src, open(out / "persistent_boxes.jsonl", "w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            row = json.loads(line)
            row.update(assignment[row["fragment_id"]])
            dst.write(json.dumps(row) + "\n")
            rows += 1

    multi = [g for g in groups if len(g["fragments"]) > 1]
    team_counts = {}
    for g in groups:
        team_counts[g["team_id"]] = team_counts.get(g["team_id"], 0) + 1
    report = {
        "tool_version": TOOL_VERSION, "method": METHOD, "parameters": params,
        "source_run_dir": str(run_dir), "inputs": inputs,
        "counts": {"fragments": len(fragments), "rows": rows, "persistent_ids": len(groups),
                   "multi_fragment_ids": len(multi), "fragments_in_multi_fragment_ids":
                   sum(len(g["fragments"]) for g in multi), "links": len(links),
                   "links_by_kind": {k: sum(1 for l in links if l["kind"] == k) for k in ("continuity", "appearance")},
                   "refused_merges": refused, "ids_by_team": team_counts},
        "accuracy_claim": None,
        "semantics": ("persistent_id is an automatic, unvalidated anonymous grouping of fragments within this "
                      "one run. It is not a real-world identity, does not change player_id/team_id/person_uid, "
                      "and is not human-reviewed. persistent_team_id uses calibrated team_id when present, "
                      "otherwise the uncalibrated kit-cluster proposal (team_status says which). No ground "
                      "truth was used; expect wrong merges between similar kits and splits across camera cuts."),
        "groups": groups, "links": links,
    }
    if a.render:
        report["render"] = {"file": "identified_persistent.mp4",
                            "frames_written": render(a.video, out / "persistent_boxes.jsonl",
                                                     out / "identified_persistent.mp4", assignment),
                            "input_video_sha256": file_sha256(a.video)}
    (out / "persistent_ids.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    c = report["counts"]
    print(f"Persistent IDs: {c['persistent_ids']} IDs for {c['fragments']} fragments "
          f"({c['multi_fragment_ids']} IDs span >1 fragment; links={c['links_by_kind']}; "
          f"refused={c['refused_merges']}; teams={c['ids_by_team']}). Unvalidated, no accuracy claim.")
    print(f"Outputs in {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
