"""Read-only Pillar 1 regression gates against an immutable Stage 2 baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path


EXACT_FILES = (
    "boxes.jsonl", "detection_accounting.jsonl",
    "detection_accounting_summary.json", "players.json",
)
RECORD_FILES = {"detections": ("identified_boxes.jsonl", "detection_id"),
                "fragments": ("fragments.json", "fragment_id")}
RENDER_FILE = "identified.mp4"
REQUIRED_FILES = (*EXACT_FILES, *(item[0] for item in RECORD_FILES.values()), RENDER_FILE)
IDENTITY_FIELDS = {"player_id", "jersey", "identification_confidence",
                   "identification_reason", "confidence", "reason"}


def _check(report, name, passed, message):
    report["checks"][name] = bool(passed)
    if not passed:
        report["errors"].append(f"{name}: {message}")


def _snapshot(root, report, name):
    hashes, errors = {}, []
    try:
        if not root.is_dir():
            raise FileNotFoundError(f"run directory does not exist: {root}")
        for path in sorted(root.rglob("*")):
            if path.is_file():
                try:
                    digest = hashlib.sha256()
                    with path.open("rb") as stream:
                        for block in iter(lambda: stream.read(1024 * 1024), b""):
                            digest.update(block)
                    hashes[path.relative_to(root).as_posix()] = digest.hexdigest()
                except OSError as exc:
                    errors.append(f"{path}: {exc}")
    except OSError as exc:
        errors.append(str(exc))
    _check(report, name, not errors, "; ".join(errors))
    return hashes


def _load_records(path, report, name):
    try:
        with path.open(encoding="utf-8") as stream:
            if path.suffix == ".jsonl":
                rows = [json.loads(line) for line in stream if line.strip()]
            else:
                rows = json.load(stream)
        if not isinstance(rows, list):
            raise ValueError("expected a list of records")
    except (OSError, ValueError, UnicodeError) as exc:
        _check(report, name, False, f"cannot load {path}: {exc}")
        return []
    _check(report, name, True, "")
    return rows


def _index(rows, id_key):
    indexed, errors = {}, []
    for position, row in enumerate(rows):
        if not isinstance(row, dict):
            errors.append(f"record {position}: expected an object")
            continue
        key = row.get(id_key)
        if not isinstance(key, str) or not key:
            errors.append(f"record {position}: missing or invalid {id_key}")
        elif key in indexed:
            errors.append(f"record {position}: duplicate {id_key} {key}")
        else:
            indexed[key] = row
    return indexed, errors


def _equal(old, new):
    if type(old) is not type(new):
        return False
    if isinstance(old, dict):
        return old.keys() == new.keys() and all(_equal(v, new[k]) for k, v in old.items())
    if isinstance(old, list):
        return len(old) == len(new) and all(_equal(v, new[i]) for i, v in enumerate(old))
    return old == new


def _team(row, kind):
    if kind == "fragments":
        evidence = row.get("team_classification", {})
        score = evidence.get("score") if isinstance(evidence, dict) else None
    else:
        score = row.get("team_score")
        evidence = {key: value for key, value in row.items()
                    if key.startswith("team_") and key not in {"team_id", "team_score"}}
    return {"team_id": row.get("team_id"), "score": score}, evidence


def _score_errors(rows, kind, id_key):
    errors = []
    for position, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        key = row.get(id_key, f"record {position}")
        team, evidence = _team(row, kind)
        label, score = team["team_id"], team["score"]
        if not isinstance(label, str) or not label:
            errors.append(f"{key}: missing or invalid team_id")
        valid = (type(score) in (int, float) and 0 <= score <= 1 and math.isfinite(score))
        if not valid:
            errors.append(f"{key}: final team score must be finite and in [0, 1], got {score!r}")
        elif label == "UNKNOWN" and score != 0:
            errors.append(f"{key}: UNKNOWN final team score must be exactly zero, got {score!r}")
        if kind == "fragments" and (
                not isinstance(evidence, dict) or evidence.get("team_id") != label):
            errors.append(f"{key}: team_classification.team_id must match team_id")
    return errors


def _distribution(rows):
    return dict(sorted(Counter(str(row.get("team_id", "<missing>"))
                               for row in rows if isinstance(row, dict)).items()))


def _compare_records(old_rows, new_rows, kind, id_key, report):
    old, old_errors = _index(old_rows, id_key)
    new, new_errors = _index(new_rows, id_key)
    _check(report, f"{kind}.baseline_ids", not old_errors, "; ".join(old_errors))
    _check(report, f"{kind}.candidate_ids", not new_errors, "; ".join(new_errors))
    _check(report, f"{kind}.counts", len(old_rows) == len(new_rows),
           f"baseline has {len(old_rows)} records; candidate has {len(new_rows)}")
    missing, added = sorted(old.keys() - new.keys()), sorted(new.keys() - old.keys())
    _check(report, f"{kind}.matching_ids", not missing and not added,
           f"missing IDs: {missing}; added IDs: {added}")
    field_changes, team_changes, new_fields = [], [], set()
    for key in sorted(old.keys() & new.keys()):
        before, after = old[key], new[key]
        new_fields.update(after.keys() - before.keys())
        for field, value in before.items():
            if field != "team_id" and (field not in after or not _equal(value, after[field])):
                field_changes.append({id_key: key, "field": field, "baseline": value,
                                      "candidate": after.get(field), "missing": field not in after})
        old_team, _ = _team(before, kind)
        new_team, evidence = _team(after, kind)
        label_changed = old_team["team_id"] != new_team["team_id"]
        score_changed = not _equal(old_team["score"], new_team["score"])
        if label_changed or score_changed:
            team_changes.append({id_key: key, "baseline": old_team, "candidate": new_team,
                                 "label_changed": label_changed, "score_changed": score_changed,
                                 "candidate_evidence": evidence})
    _check(report, f"{kind}.original_fields", not field_changes,
           f"{len(field_changes)} original non-team fields changed; see {kind}.field_changes")
    score_errors = _score_errors(new_rows, kind, id_key)
    _check(report, f"{kind}.final_scores", not score_errors, "; ".join(score_errors))
    return {"counts": {"baseline": len(old_rows), "candidate": len(new_rows)},
            "missing_ids": missing, "added_ids": added,
            "id_errors": {"baseline": old_errors, "candidate": new_errors},
            "field_changes": field_changes,
            "identity_changes": [change for change in field_changes
                                 if change["field"] in IDENTITY_FIELDS],
            "new_fields": sorted(new_fields), "score_errors": score_errors,
            "team_distributions": {"baseline": _distribution(old_rows),
                                   "candidate": _distribution(new_rows)},
            "team_changes": team_changes}


def _json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def compare_runs(baseline: Path, candidate: Path, allow_render_change=False) -> dict:
    """Preserve original fields except team_id; candidate final team scores are mandatory."""
    roots = {"baseline": Path(baseline).resolve(), "candidate": Path(candidate).resolve()}
    report = {**{key: str(path) for key, path in roots.items()}, "checks": {}, "errors": []}
    before = {key: _snapshot(path, report, f"{key}.hash_before") for key, path in roots.items()}
    for side in roots:
        missing = [name for name in REQUIRED_FILES if name not in before[side]]
        _check(report, f"{side}.required_files", not missing,
               f"missing mandatory files in {roots[side]}: {', '.join(missing)}")
    report["exact_files"] = {}
    for name in EXACT_FILES:
        old, new = before["baseline"].get(name), before["candidate"].get(name)
        equal = old is not None and old == new
        report["exact_files"][name] = {"baseline_sha256": old, "candidate_sha256": new,
                                       "equal": equal}
        _check(report, f"exact.{name}", equal, "SHA256 mismatch or missing file")
    for kind, (filename, id_key) in RECORD_FILES.items():
        rows = {side: _load_records(path / filename, report, f"{kind}.{side}.load")
                for side, path in roots.items()}
        report[kind] = _compare_records(rows["baseline"], rows["candidate"], kind, id_key, report)
    old = before["baseline"].get(RENDER_FILE)
    new = before["candidate"].get(RENDER_FILE)
    present = old is not None and new is not None
    changed = old != new if present else None
    status = ("missing" if not present else "identical" if not changed else
              "intentional_difference" if allow_render_change else "unexpected_difference")
    render_ok = present and (not changed or allow_render_change)
    report["render"] = {"baseline_sha256": old, "candidate_sha256": new, "changed": changed,
                        "allow_render_change": bool(allow_render_change), "status": status}
    _check(report, "render", render_ok, f"{status}: use --allow-render-change only for intentional overlays")
    report["read_only"] = {}
    for side, path in roots.items():
        after = _snapshot(path, report, f"{side}.hash_after")
        unchanged = (before[side] == after and report["checks"][f"{side}.hash_before"]
                     and report["checks"][f"{side}.hash_after"])
        report["read_only"][side] = {"before": before[side], "after": after, "unchanged": unchanged}
        _check(report, f"{side}.read_only", unchanged, "input SHA256 inventory changed or could not be read")
    report["passed"] = all(report["checks"].values())
    return _json_safe(report)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--allow-render-change", action="store_true")
    args = parser.parse_args(argv)
    if args.out.exists() or args.out.is_symlink():
        parser.error(f"refusing to overwrite existing --out: {args.out}")
    if not args.out.parent.is_dir():
        parser.error(f"--out parent directory does not exist: {args.out.parent}")
    if any(args.out.resolve().is_relative_to(root.resolve()) for root in (args.baseline, args.candidate)):
        parser.error('--out must be outside both run directories to preserve read-only inputs')
    report = compare_runs(args.baseline, args.candidate, args.allow_render_change)
    try:
        with args.out.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    except OSError as exc:
        parser.error(f"cannot create --out without overwriting: {exc}")
    print(f"Pillar 1 regression: {'PASS' if report['passed'] else 'FAIL'} ({args.out.resolve()})")
    for error in report["errors"]:
        print(error)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
