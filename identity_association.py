"""Conservative, review-assisted fragment association; no model inference.

Generic ImageNet ResNet18 is not trained/validated football Re-ID. Numerical
similarity, frame gaps and image-space displacement are diagnostics, never
acceptance thresholds. Synthetic tests establish policy behavior, not accuracy.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from datetime import date, datetime
from numbers import Integral, Real
from pathlib import Path


SCHEMA_VERSION = "1.1"
ASSOCIATION_METHOD = "human_reviewed_pair_clique"
# Kept in pairs[].reasons for accepted pairs so every pair carries an
# explanation; report validators must read it as evidence of acceptance, not a veto.
ACCEPTANCE_EXPLANATION = "explicit_same_review_and_all_safety_checks_passed"
EXTENSION_FIELDS = (
    "schema_version", "person_uid", "association_status",
    "association_reason", "association_method",
)
_REVIEW_KIND = "human_fragment_identity_review"
_UNKNOWN_IDS = {"", "UNK", "UNKNOWN", "NONE"}


def _text(value):
    return isinstance(value, str) and bool(value.strip())


def _hash(value, name):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-fA-F]{64}", value) is None:
        raise ValueError(f"{name} must be an exact 64-character SHA256 hex digest")


def _unique_object(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValueError(f"non-finite JSON constant: {value}")


def _fields(record, required, where):
    if not isinstance(record, dict) or set(record) != set(required):
        raise ValueError(f"{where} must contain exactly: {', '.join(required)}")


def _iso_review_time(value):
    # Accept ISO calendar dates or date-times, including Z on Python 3.10.
    if not _text(value) or re.fullmatch(
        r"\d{4}-\d{2}-\d{2}(?:[Tt ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?"
        r"(?:[Zz]|[+-]\d{2}:\d{2})?)?", value
    ) is None:
        raise ValueError("reviewed_at must be an ISO date or date-time")
    try:
        if len(value) == 10:
            date.fromisoformat(value)
        else:
            datetime.fromisoformat(value.replace("t", "T").replace("Z", "+00:00").replace("z", "+00:00"))
    except ValueError as exc:
        raise ValueError("reviewed_at must be a valid ISO date or date-time") from exc


def _load_review(path, ids, video_sha256, boxes_sha256):
    source = Path(path).resolve()
    raw = source.read_bytes()
    try:
        review = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object,
                            parse_constant=_invalid_constant)
    except (UnicodeError, ValueError) as exc:
        raise ValueError(f"invalid identity review JSON: {exc}") from exc
    _fields(review, ("kind", "video_sha256", "boxes_sha256", "reviewer", "reviewed_at",
                     "fragments", "pairs"), "review")
    if review["kind"] != _REVIEW_KIND:
        raise ValueError(f"review kind must be {_REVIEW_KIND}")
    for key, expected in (("video_sha256", video_sha256), ("boxes_sha256", boxes_sha256)):
        _hash(review[key], key)
        if review[key] != expected:
            raise ValueError(f"stale review: {key} does not match the exact input digest")
    if not _text(review["reviewer"]):
        raise ValueError("reviewer must be a nonempty string")
    _iso_review_time(review["reviewed_at"])
    if not isinstance(review["fragments"], list) or not isinstance(review["pairs"], list):
        raise ValueError("review fragments and pairs must be lists")
    annotations = {}
    for entry in review["fragments"]:
        _fields(entry, ("fragment_id", "single_person", "full_span_reviewed", "shot_id"),
                "fragment review")
        fid = entry["fragment_id"]
        if not _text(fid) or fid not in ids:
            raise ValueError("fragment review references an unknown fragment_id")
        if fid in annotations:
            raise ValueError(f"duplicate fragment review: {fid}")
        if type(entry["single_person"]) is not bool or type(entry["full_span_reviewed"]) is not bool:
            raise ValueError("single_person and full_span_reviewed must be booleans")
        if entry["shot_id"] is not None and not _text(entry["shot_id"]):
            raise ValueError("shot_id must be a nonempty string or null")
        annotations[fid] = entry
    pairs = {}
    for entry in review["pairs"]:
        _fields(entry, ("left", "right", "decision", "reason"), "pair review")
        left, right = entry["left"], entry["right"]
        if not _text(left) or not _text(right) or left not in ids or right not in ids:
            raise ValueError("pair references unknown fragment IDs")
        if left == right:
            raise ValueError("self-pairs are prohibited")
        if left not in annotations or right not in annotations:
            raise ValueError("both pair fragment entries must exist in the review")
        key = tuple(sorted((left, right)))
        if key in pairs:
            raise ValueError(f"duplicate unordered pair: {key}")
        if entry["decision"] not in ("same", "different", "uncertain"):
            raise ValueError("pair decision must be same, different or uncertain")
        if not _text(entry["reason"]):
            raise ValueError("pair reason must be a nonempty string")
        pairs[key] = entry
    provenance = {
        "kind": _REVIEW_KIND, "source_path": str(source),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "reviewer": review["reviewer"], "reviewed_at": review["reviewed_at"],
        "video_sha256": review["video_sha256"], "boxes_sha256": review["boxes_sha256"],
    }
    return annotations, pairs, provenance


def _vector(value):
    """Accept ordered real 1-D lists/arrays; reject strings, booleans, shapes and unordered inputs."""
    # Sets have no iteration order, so their element sequence is not reproducible.
    if isinstance(value, (str, bytes, Mapping, set, frozenset)):
        return None, "invalid_embedding_shape"
    try:
        values = list(value)
    except TypeError:
        return None, "invalid_embedding_shape"
    if not values or any(not isinstance(x, Real) or isinstance(x, bool) for x in values):
        return None, "invalid_embedding_shape"
    try:
        values = [float(x) for x in values]
    except (ValueError, OverflowError):
        return None, "nonfinite_embedding"
    if not all(math.isfinite(x) for x in values):
        return None, "nonfinite_embedding"
    return values, None


def _unit_embedding(value):
    if value is None:
        return None, "missing_embedding"
    values, error = _vector(value)
    if error:
        return None, error
    scale = max(abs(x) for x in values)
    if scale == 0:
        return None, "zero_embedding"
    # Scaling avoids norm overflow/underflow without imposing an epsilon gate.
    scaled = [x / scale for x in values]
    norm = math.sqrt(math.fsum(x * x for x in scaled))
    return [x / norm for x in scaled], None


def _known_id(value):
    return value if isinstance(value, str) and value.strip().upper() not in _UNKNOWN_IDS else None


def _record(value):
    return value if isinstance(value, Mapping) else {}


def _features(fragment):
    reasons = []
    evidence = _record(fragment.get("appearance_evidence"))
    if evidence.get("status") != "ok":
        reasons.append("appearance_evidence_not_ok")
    if evidence.get("embedding_generated") is not True:
        reasons.append("embedding_not_generated")
    unit, error = _unit_embedding(fragment.get("embedding"))
    if error:
        reasons.append(error)
    start, end = fragment.get("start_frame"), fragment.get("end_frame")
    interval = None
    if (isinstance(start, Integral) and not isinstance(start, bool)
            and isinstance(end, Integral) and not isinstance(end, bool)
            and 0 <= start <= end):
        interval = [int(start), int(end)]
    else:
        reasons.append("invalid_frame_interval")
    team = _record(fragment.get("team_classification"))
    proposal = _record(team.get("proposal"))
    records = (fragment, evidence, team, proposal)
    for marker in ("mixed_fragment_appearance", "persistent_role_candidate"):
        flagged = any(
            record.get(marker) is True or any(record.get(key) == marker for key in
                                             ("status", "evidence_status", "reason", "team_evidence_status"))
            for record in records
        )
        if marker == "mixed_fragment_appearance":
            # Even a reliability-abstained UNKNOWN team can conceal mixed kit votes.
            flagged = flagged or any(len({key for key in _record(record.get("crop_votes"))
                                          if _known_id(key) is not None}) > 1
                                     for record in (team, proposal))
        else:
            flagged = flagged or any(_record(record.get("candidate_evidence")).get(
                "persistent_evidence") is True for record in (team, proposal))
        if flagged:
            reasons.append(marker)
    teams = {_known_id(fragment.get("team_id")), _known_id(team.get("team_id"))} - {None}
    return {"reasons": reasons, "unit": unit, "interval": interval, "teams": teams,
            "player_id": _known_id(fragment.get("player_id"))}


def _bbox_center(box):
    coords, error = _vector(box)
    if error or len(coords) != 4:
        return None
    x1, y1, x2, y2 = coords
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1 / 2 + x2 / 2, y1 / 2 + y2 / 2)


def _displacement(endpoints, earlier, later):
    if endpoints is None:
        return None, "not_provided"
    a, b = _record(endpoints.get(earlier)), _record(endpoints.get(later))
    if a.get("end_bbox") is None or b.get("start_bbox") is None:
        return None, "missing_endpoints"
    ca, cb = _bbox_center(a["end_bbox"]), _bbox_center(b["start_bbox"])
    if ca is None or cb is None:
        return None, "invalid_endpoints"
    distance = math.hypot(cb[0] - ca[0], cb[1] - ca[1])
    if not math.isfinite(distance):
        return None, "nonfinite_displacement"
    return distance, "ok"


def _assess_pair(key, review, fragments, features, annotations, endpoints):
    left, right = key
    a, b = features[left], features[right]
    ra, rb = annotations[left], annotations[right]
    reasons = []
    if review["decision"] != "same":
        reasons.append("review_decision_" + review["decision"])
    for side, feature, annotation in (("left", a, ra), ("right", b, rb)):
        reasons.extend(f"{side}:{reason}" for reason in feature["reasons"])
        if not annotation["single_person"]:
            reasons.append(f"{side}:single_person_not_confirmed")
        if not annotation["full_span_reviewed"]:
            reasons.append(f"{side}:full_span_not_reviewed")
    same_shot = ra["shot_id"] is not None and ra["shot_id"] == rb["shot_id"]
    if ra["shot_id"] is None or rb["shot_id"] is None:
        reasons.append("unknown_shot_continuity")
    elif not same_shot:
        reasons.append("different_reviewed_shots")
    cosine = None
    if a["unit"] is not None and b["unit"] is not None:
        if len(a["unit"]) != len(b["unit"]):
            reasons.append("incompatible_embedding_shapes")
        else:
            cosine = max(-1.0, min(1.0, math.fsum(x * y for x, y in zip(a["unit"], b["unit"]))))
    overlap, gap, displacement, direction = None, None, None, None
    displacement_status = "overlap_or_invalid_intervals"
    if a["interval"] is not None and b["interval"] is not None:
        sa, ea = a["interval"]
        sb, eb = b["interval"]
        gap = max(sa, sb) - min(ea, eb)
        overlap = gap <= 0  # Inclusive intervals: touching endpoints overlap.
        if overlap:
            reasons.append("inclusive_frame_overlap")
        else:
            earlier, later = (left, right) if sa < sb else (right, left)
            direction = {"from": earlier, "to": later}
            displacement, displacement_status = _displacement(endpoints, earlier, later)
    if len(a["teams"] | b["teams"]) > 1:
        reasons.append("conflicting_known_team_ids")
    if a["player_id"] is not None and b["player_id"] is not None and a["player_id"] != b["player_id"]:
        reasons.append("conflicting_resolved_player_ids")
    ta, tb = fragments[left].get("track_id"), fragments[right].get("track_id")
    same_track = (bool(ta == tb) if all(isinstance(t, Integral) and not isinstance(t, bool)
                                     and t >= 0 for t in (ta, tb)) else None)
    return {
        "left": left, "right": right, "decision": review["decision"],
        "review_reason": review["reason"], "accepted": not reasons,
        "reasons": reasons or [ACCEPTANCE_EXPLANATION],
        "diagnostics": {
            "descriptive_only": True, "cosine_similarity": cosine,
            "left_interval": a["interval"], "right_interval": b["interval"],
            "temporal_gap_frames": gap, "intervals_overlap": overlap,
            "same_reviewed_shot": same_shot, "same_track_id": same_track,
            "endpoint_displacement_pixels": displacement,
            "endpoint_displacement_status": displacement_status, "endpoint_direction": direction,
        },
    }


def _clique_groups(ids, pairs):
    accepted = {(p["left"], p["right"]) for p in pairs if p["accepted"]}
    owner = {fid: fid for fid in ids}
    members = {fid: [fid] for fid in ids}
    for left, right in sorted(accepted):
        ga, gb = owner[left], owner[right]
        if ga == gb:
            continue
        if all(tuple(sorted((a, b))) in accepted for a in members[ga] for b in members[gb]):
            merged = sorted(members.pop(ga) + members.pop(gb))
            root = merged[0]
            members[root] = merged
            for fid in merged:
                owner[fid] = root
    lookup = {(p["left"], p["right"]): p for p in pairs}
    blocked = []
    for pair in pairs:
        left, right = pair["left"], pair["right"]
        pair["grouped"] = pair["accepted"] and owner[left] == owner[right]
        if not pair["accepted"] or pair["grouped"]:
            continue
        ma, mb = members[owner[left]], members[owner[right]]
        missing = sorted(tuple(sorted((a, b))) for a in ma for b in mb
                         if tuple(sorted((a, b))) not in accepted)
        blocked.append({
            "left": left, "right": right, "accepted_evidence": True,
            "reason": "clique_condition_not_met",
            "left_group_fragment_ids": list(ma), "right_group_fragment_ids": list(mb),
            "blocking_pairs": [
                {"left": a, "right": b,
                 "decision": lookup[(a, b)]["decision"] if (a, b) in lookup else None,
                 "reasons": list(lookup[(a, b)]["reasons"]) if (a, b) in lookup else ["unreviewed_pair"]}
                for a, b in missing
            ],
        })
    return sorted(group for group in members.values() if len(group) >= 2), blocked


def _person_uid(video_sha256, boxes_sha256, members):
    payload = json.dumps([video_sha256, boxes_sha256, sorted(members)],
                         ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    return "P" + hashlib.sha256(payload).hexdigest()[:32]


def associate_fragments(fragments, *, video_sha256, boxes_sha256, review_path=None, endpoints=None):
    """Return a JSON-safe report and write ONLY the five extension fields.

    ``fragments`` contains mutable dictionaries in run.py's fragment format;
    embeddings may be lists or NumPy vectors. ``review_path`` is optional strict
    UTF-8 JSON. Invalid/stale reviews raise ValueError before any mutation (file
    access errors propagate). Review fragments may be a subset. ``endpoints``
    optionally maps IDs to start_bbox/end_bbox xyxy boxes for diagnostics only.

    Only reviewed pairs are assessed; unreviewed pairs abstain. Accepted pair
    evidence is distinct from final grouping: deterministic sorted greedy merges
    require an accepted complete cross-pair clique, never transitive closure.
    UIDs are P + the first 32 lowercase SHA256 hex characters of UTF-8 compact
    JSON [exact video hash, exact boxes hash, sorted member IDs], ensure_ascii=True.
    No clock, review ordering, tracker ID or numeric similarity affects the UID.
    """
    _hash(video_sha256, "video_sha256")
    _hash(boxes_sha256, "boxes_sha256")
    by_id = {}
    for fragment in fragments:
        if not isinstance(fragment, dict) or not _text(fragment.get("fragment_id")):
            raise ValueError("each fragment must be a dict with a nonempty string fragment_id")
        fid = fragment["fragment_id"]
        if fid in by_id:
            raise ValueError(f"duplicate input fragment_id: {fid}")
        by_id[fid] = fragment
    if endpoints is not None and not isinstance(endpoints, Mapping):
        raise ValueError("endpoints must be a mapping or None")
    ids = sorted(by_id)
    annotations, reviews, review_provenance = {}, {}, None
    if review_path is not None:
        annotations, reviews, review_provenance = _load_review(
            review_path, by_id, video_sha256, boxes_sha256)
    # Do not build or assess an unreviewed n-squared similarity matrix.
    reviewed_ids = {fid for pair in reviews for fid in pair}
    features = {fid: _features(by_id[fid]) for fid in sorted(reviewed_ids)}
    pairs = [_assess_pair(key, reviews[key], by_id, features, annotations, endpoints)
             for key in sorted(reviews)]
    member_groups, blocked = _clique_groups(ids, pairs)
    groups = [{"person_uid": _person_uid(video_sha256, boxes_sha256, members),
               "fragment_ids": members} for members in member_groups]
    uids = {fid: group["person_uid"] for group in groups for fid in group["fragment_ids"]}
    incident = {fid: [] for fid in ids}
    for pair in pairs:
        incident[pair["left"]].append(pair)
        incident[pair["right"]].append(pair)
    decisions = []
    for fid in ids:
        uid = uids.get(fid)
        if uid is not None:
            reason = "human_reviewed_same_identity_complete_pair_clique"
        elif review_path is None:
            reason = "no_human_review"
        elif not incident[fid]:
            reason = "no_reviewed_pairs_for_fragment"
        else:
            reasons = {r for pair in incident[fid] if not pair["accepted"] for r in pair["reasons"]}
            if any(pair["accepted"] for pair in incident[fid]):
                reasons.add("accepted_pair_blocked_by_clique_condition")
            reason = "; ".join(sorted(reasons))
        decisions.append({
            "fragment_id": fid, "schema_version": SCHEMA_VERSION, "person_uid": uid,
            "association_status": "associated" if uid else "unresolved",
            "association_reason": reason,
            "association_method": ASSOCIATION_METHOD if uid else "abstain",
        })
    accepted_count = sum(pair["accepted"] for pair in pairs)
    report = {
        "schema_version": SCHEMA_VERSION, "kind": "identity_association_report",
        "assessment_scope": "only_reviewed_pairs_assessed; unreviewed_pairs_abstain",
        "pairs": pairs, "groups": groups, "blocked_links": blocked, "decisions": decisions,
        "counts": {
            "fragments_total": len(ids), "fragments_associated": len(uids),
            "fragments_unresolved": len(ids) - len(uids), "groups": len(groups),
            "reviewed_fragments": len(annotations), "reviewed_pairs": len(pairs),
            "accepted_pairs": accepted_count, "rejected_pairs": len(pairs) - accepted_count,
            "blocked_accepted_pairs": len(blocked),
            "unreviewed_pairs": len(ids) * (len(ids) - 1) // 2 - len(pairs),
        },
        "provenance": {"video_sha256": video_sha256, "boxes_sha256": boxes_sha256,
                       "review": review_provenance},
        "limitations": [
            "Generic ImageNet ResNet18 is not trained or validated football Re-ID; no automatic identity acceptance.",
            "Only reviewed pairs are assessed. Unreviewed pairs abstain; no unreviewed all-pairs matrix is generated.",
            "Human review is input evidence, not held-out ground truth; synthetic scenarios are not real identity accuracy.",
            "Cosine, temporal gap and endpoint displacement are descriptive only, with no numerical acceptance thresholds.",
            "Frame intervals are inclusive; temporal_gap_frames is max(starts) minus min(ends), so zero overlaps.",
            "Same track, uniform, team or roster ID alone never links fragments. Unknown shot continuity vetoes links.",
            "Sorted greedy complete-pair cliques may split one person; accepted evidence blocked by a clique is reported separately.",
            "person_uid is a run-scoped anonymous hypothesis, not a real-world identity; null UIDs do not identify one person.",
        ],
    }
    # All validation/assessment completes before the sole mutation boundary.
    for decision in decisions:
        by_id[decision["fragment_id"]].update({key: decision[key] for key in EXTENSION_FIELDS})
    return report
