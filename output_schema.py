"""
Canonical output contract for Pillar 1 detection, tracking and identity fields.

Task 1.2 (output-contract normalization) only:
- one deterministic ``detection_id`` per detection record;
- one documented field registry shared by every writer and reader;
- one team vocabulary (TEAM_A / TEAM_B / UNKNOWN), independent of the
  ``player_id`` unresolved sentinel (UNK).

Nothing in this module classifies, clusters or associates anything. It only
defines, normalizes and validates the output contract.

``person_uid`` is declared RESERVED: it is not emitted by any current stage
and validators must not require it.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional, Sequence

SCHEMA_VERSION = "1.0"

# --- Team vocabulary -------------------------------------------------------

TEAM_A = "TEAM_A"
TEAM_B = "TEAM_B"
TEAM_UNKNOWN = "UNKNOWN"

CANONICAL_TEAM_LABELS: tuple[str, str, str] = (TEAM_A, TEAM_B, TEAM_UNKNOWN)

# --- Player identity sentinel (NOT a team label) ---------------------------

PLAYER_ID_UNRESOLVED = "UNK"

# detection_id
# D<frame>_<kind><key>_<seq> where kind T = tracker-derived key and
# N = ordinal-derived key (no usable track_id). The tag keeps the two key
# namespaces provably disjoint.
TRACK_KEY_FROM_TRACKER = "T"
TRACK_KEY_FROM_ORDINAL = "N"

DETECTION_ID_PATTERN = re.compile(r"^D\d{6}_[TN]\d{6,}_\d{3}$")

# Rows whose team field carries the old player-identity sentinel.
LEGACY_TEAM_SENTINELS = frozenset({"UNK", "unk", ""})

# Aliases that can be normalized losslessly into the canonical vocabulary.
TEAM_LABEL_ALIASES: dict[str, str] = {
    "": TEAM_UNKNOWN,
    "UNK": TEAM_UNKNOWN,
    "unk": TEAM_UNKNOWN,
    "unknown": TEAM_UNKNOWN,
    "none": TEAM_UNKNOWN,
    "None": TEAM_UNKNOWN,
    "team_a": TEAM_A,
    "Team_A": TEAM_A,
    "TEAMA": TEAM_A,
    "team_b": TEAM_B,
    "Team_B": TEAM_B,
    "TEAMB": TEAM_B,
}

# Field names that are legal in team_id but are not anonymous classification
# labels. They come from roster metadata and must never be silently coerced
# into TEAM_A / TEAM_B.
NON_CANONICAL_TEAM_LABELS_ACCEPTED = frozenset({"HOME", "AWAY"})

# --- Field registry --------------------------------------------------------

_STR = (str,)
_INT = (int,)
_NUM = (int, float)
_STR_LIST = (list,)

# field -> (types, required_in, notes)
# required_in is the set of record kinds that must carry the field.
FIELD_REGISTRY: dict[str, tuple[tuple[type, ...], frozenset[str], str]] = {
    "schema_version": (_STR, frozenset(), "contract version stamped into reports"),
    # spatiotemporal
    "frame": (_INT, frozenset({"detection"}), "0-based video frame index"),
    "timestamp_s": (_NUM, frozenset(), "frame / fps"),
    # detection
    "detection_id": (_STR, frozenset(), "unique per detection within a run"),
    "bbox_xyxy": (_STR_LIST, frozenset({"detection"}), "x1, y1, x2, y2 pixels"),
    "detector_confidence": (_NUM, frozenset(), "YOLO confidence"),
    "class_id": (_INT, frozenset(), "detector class index"),
    # tracking
    "track_id": (_INT, frozenset({"detection"}), "short-term ByteTrack trajectory"),
    "fragment_id": (_STR, frozenset(), "bounded segment of a trajectory"),
    # identity (player)
    "person_uid": (_STR, frozenset(), "RESERVED - not implemented in Task 1.2"),
    "player_id": (_STR, frozenset(), "verified roster identity or UNK"),
    "jersey": (_INT, frozenset(), "OCR-derived jersey number, nullable"),
    # team (independent of roster identity)
    "team_id": (_STR, frozenset(), "TEAM_A / TEAM_B / UNKNOWN"),
    # provenance
    "identification_confidence": (_NUM, frozenset(), "roster similarity score"),
    "identification_reason": (_STR, frozenset(), "why this identity decision"),
}

# Deprecated names still emitted for readers that predate the contract.
CANONICAL_FIELD_ALIASES: dict[str, str] = {
    "confidence": "identification_confidence",
    "reason": "identification_reason",
    "player": "player_id",
}

DETECTION_FIELDS: tuple[str, ...] = (
    "frame",
    "timestamp_s",
    "detection_id",
    "track_id",
    "fragment_id",
    "bbox_xyxy",
    "detector_confidence",
    "class_id",
)

DETECTION_IDENTITY_FIELDS: tuple[str, ...] = (
    "player_id",
    "team_id",
    "jersey",
    "identification_confidence",
    "identification_reason",
)

# Reserved identity fields: declared for the roadmap, never validated as
# required by any current stage.
RESERVED_FIELDS: frozenset[str] = frozenset({"person_uid"})


class SchemaError(ValueError):
    """Raised when a record cannot be normalized into the canonical contract."""


# --- detection_id ----------------------------------------------------------


def detection_key_of(track_id: Any, row_ordinal: int) -> tuple[str, int]:
    """Return the (kind, key) pair identifying a detection's lineage.

    A usable non-negative integer ``track_id`` yields ``("T", track_id)``;
    anything else yields ``("N", abs(row_ordinal))``. Because the kind tag is
    part of the ID, an ordinal key can never collide with a tracker key of the
    same numeric value.
    """
    try:
        value = int(track_id)
    except (TypeError, ValueError):
        return TRACK_KEY_FROM_ORDINAL, abs(int(row_ordinal))
    if isinstance(track_id, bool) or value < 0:
        return TRACK_KEY_FROM_ORDINAL, abs(int(row_ordinal))
    return TRACK_KEY_FROM_TRACKER, value


def build_detection_id(frame: int, track_id: Any, seq: int = 0, row_ordinal: int = 0) -> str:
    """Compose the canonical ``detection_id``.

    ``track_id`` may be any value; anything that is not a non-negative integer
    is mapped onto an ordinal-derived key tagged ``N`` so that legacy rows
    without a tracker ID still receive a unique identifier. Callers normally
    obtain ``seq`` from :func:`assign_detection_ids`.
    """
    frame_int = _require_int(frame, "frame")
    if frame_int < 0:
        raise SchemaError(f"frame must be non-negative; got {frame_int}")
    if seq < 0:
        raise SchemaError(f"seq must be non-negative; got {seq}")
    kind, key = detection_key_of(track_id, row_ordinal)
    return f"D{frame_int:06d}_{kind}{key:06d}_{seq % 1000:03d}"


def _require_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or value is None:
        raise SchemaError(f"{name} must be an integer; got {value!r}")
    try:
        as_int = int(value)
    except (TypeError, ValueError):
        raise SchemaError(f"{name} must be an integer; got {value!r}") from None
    if isinstance(value, float) and as_int != value:
        raise SchemaError(f"{name} must be a whole number; got {value!r}")
    return as_int


def assign_detection_ids(
    rows: Iterable[dict[str, Any]],
    *,
    overwrite: bool = False,
) -> list[dict[str, Any]]:
    """Return rows with a deterministic, collision-free ``detection_id``.

    Applied in file order, so the result is reproducible for the same input
    sequence. Rows that already carry a valid ID are left untouched unless
    ``overwrite=True``. Uniqueness is guaranteed by the (frame, key kind, key,
    occurrence) tuple: tracker-derived and ordinal-derived keys live in
    separate tagged namespaces, and two rows that would otherwise collide are
    separated by the occurrence counter. A mixed file whose pre-existing IDs
    clash with a synthesized one raises :class:`SchemaError`.
    """
    materialized = list(rows)
    seen: Counter[tuple[int, str, int]] = Counter()
    assigned: list[dict[str, Any]] = []

    for ordinal, row in enumerate(materialized):
        if not isinstance(row, dict):
            raise SchemaError(f"Row {ordinal} is not a mapping: {row!r}")
        existing = row.get("detection_id")
        if isinstance(existing, str) and existing and not overwrite:
            if not is_valid_detection_id(existing):
                raise SchemaError(
                    f"Row {ordinal} has a malformed detection_id: {existing!r}"
                )
            assigned.append(row)
            continue

        frame = _require_int(row.get("frame", ordinal), "frame")
        kind, key = detection_key_of(row.get("track_id"), ordinal)
        seq = seen[(frame, kind, key)]
        seen[(frame, kind, key)] += 1

        row["detection_id"] = build_detection_id(frame, row.get("track_id"), seq, ordinal)
        assigned.append(row)

    duplicates = find_duplicate_detection_ids(assigned)
    if duplicates:
        raise SchemaError(
            f"detection_id collision detected after assignment: {duplicates[:5]}"
        )
    return assigned


def find_duplicate_detection_ids(rows: Sequence[dict[str, Any]]) -> list[str]:
    counts: Counter[str] = Counter(
        row["detection_id"]
        for row in rows
        if isinstance(row, dict) and isinstance(row.get("detection_id"), str)
    )
    return sorted(id_ for id_, count in counts.items() if count > 1)


def is_valid_detection_id(value: Any) -> bool:
    return isinstance(value, str) and bool(DETECTION_ID_PATTERN.fullmatch(value))


def detection_id_accounting(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Summarize ID coverage; used by report writers, never to gate predictions."""
    total = len(rows)
    present = sum(
        1
        for row in rows
        if isinstance(row, dict) and isinstance(row.get("detection_id"), str)
    )
    valid = sum(1 for row in rows if isinstance(row, dict) and is_valid_detection_id(row.get("detection_id")))
    unique = len(
        {
            row.get("detection_id")
            for row in rows
            if isinstance(row, dict) and isinstance(row.get("detection_id"), str)
        }
    )
    return {
        "total_records": total,
        "records_with_detection_id": present,
        "records_with_canonical_format": valid,
        "unique_detection_ids": unique,
        "duplicate_detection_ids": find_duplicate_detection_ids(rows),
        "missing_detection_id": total - present,
    }


# --- JSONL I/O -------------------------------------------------------------


def iter_jsonl(path: Path | str) -> Iterator[dict[str, Any]]:
    """Yield JSON objects from a .jsonl file, skipping blank lines."""
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SchemaError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise SchemaError(f"{path}:{line_number} is not a JSON object")
            yield value


def load_detection_rows(
    path: Path | str,
    *,
    enforce_bbox: bool = True,
) -> list[dict[str, Any]]:
    """Read a detection JSONL (legacy or current) and normalize its IDs.

    Legacy files without ``detection_id`` are backfilled using the same rule
    that ``run.py`` now applies at write time, so a re-read and a fresh run
    agree.
    """
    rows = list(iter_jsonl(path))
    for index, row in enumerate(rows):
        missing = [key for key in ("frame", "track_id", "bbox_xyxy") if key not in row]
        if missing:
            raise SchemaError(f"{path}: row {index} missing required fields {missing}")
        if enforce_bbox:
            validate_bbox(row["bbox_xyxy"], context=f"{path}: row {index}")
    return assign_detection_ids(rows)


def write_jsonl(rows: Sequence[dict[str, Any]], path: Path | str) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


# --- field validation ------------------------------------------------------


def validate_bbox(value: Any, *, context: str = "bbox_xyxy") -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise SchemaError(f"{context}: bbox_xyxy must have 4 values, got {value!r}")
    try:
        coords = [float(v) for v in value]
    except (TypeError, ValueError):
        raise SchemaError(f"{context}: bbox_xyxy must be numeric, got {value!r}") from None
    if any(v != v or v in (float("inf"), float("-inf")) for v in coords):
        raise SchemaError(f"{context}: bbox_xyxy must be finite, got {coords}")
    if coords[2] <= coords[0] or coords[3] <= coords[1]:
        raise SchemaError(f"{context}: bbox_xyxy must be ordered x1<x2, y1<y2, got {coords}")
    return coords


def validate_fields(
    row: dict[str, Any],
    fields: Iterable[str],
    *,
    required: Iterable[str] = (),
    context: str = "record",
) -> tuple[list[str], list[str]]:
    """Type-check ``fields``; return (errors, warnings)."""
    errors: list[str] = []
    warnings: list[str] = []
    required_set = set(required)

    for name in fields:
        if name not in row:
            if name in required_set:
                errors.append(f"{context}: missing required field '{name}'")
            continue
        spec = FIELD_REGISTRY.get(name)
        if spec is None:
            warnings.append(f"{context}: unknown field '{name}'")
            continue
        types, _, _ = spec
        value = row[name]
        if value is None:
            continue
        if name == "bbox_xyxy":
            try:
                validate_bbox(value, context=context)
            except SchemaError as exc:
                errors.append(str(exc))
            continue
        if types == _NUM:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                errors.append(f"{context}: {name} must be numeric, got {value!r}")
            continue
        if types == _INT:
            if isinstance(value, bool) or not isinstance(value, int):
                errors.append(f"{context}: {name} must be an integer, got {value!r}")
            continue
        if not isinstance(value, types):
            errors.append(f"{context}: {name} must be {types}, got {type(value).__name__}")
    return errors, warnings


def validate_detection_record(
    row: dict[str, Any],
    *,
    strict: bool = False,
    context: str = "detection",
) -> tuple[list[str], list[str]]:
    """Validate one detection/tracking record against the contract.

    ``strict=False`` (required level) accepts legacy records and reports what
    normalization would change. ``strict=True`` additionally demands a canonical
    ``detection_id`` and canonical team vocabulary.
    """
    if not isinstance(row, dict):
        return [f"{context}: record is not a mapping"], []

    fields = [
        name
        for name, (_, required_in, _) in FIELD_REGISTRY.items()
        if "detection" in required_in
    ]
    errors, warnings = validate_fields(
        row,
        DETECTION_FIELDS + DETECTION_IDENTITY_FIELDS,
        required=fields,
        context=context,
    )

    if "detection_id" not in row or row.get("detection_id") in (None, ""):
        message = f"{context}: no detection_id"
        if strict:
            errors.append(message)
        else:
            warnings.append(message)
    elif not is_valid_detection_id(row["detection_id"]):
        errors.append(f"{context}: detection_id {row['detection_id']!r} violates {DETECTION_ID_PATTERN.pattern}")

    if "team_id" in row:
        team = row["team_id"]
        if team is None or team in LEGACY_TEAM_SENTINELS:
            message = f"{context}: team_id {team!r} is the legacy player sentinel; normalize to {TEAM_UNKNOWN!r}"
            if strict:
                errors.append(message)
            else:
                warnings.append(message)
        elif not is_canonical_team_label(team):
            # Roster club labels are preserved verbatim and only warn: coercing
            # them into TEAM_A/TEAM_B would fabricate a classification result.
            message = f"{context}: team_id {team!r} is outside the anonymous team vocabulary"
            if strict and team not in NON_CANONICAL_TEAM_LABELS_ACCEPTED:
                errors.append(message)
            else:
                warnings.append(message)

    if "player_id" in row and row["player_id"] in CANONICAL_TEAM_LABELS:
        errors.append(f"{context}: player_id {row['player_id']!r} is a team label, not a player identity")

    for alias, canonical in CANONICAL_FIELD_ALIASES.items():
        if alias in row and canonical not in row:
            warnings.append(f"{context}: deprecated field '{alias}' should be '{canonical}'")

    if "person_uid" in row:
        errors.append(f"{context}: person_uid is reserved and must not be emitted yet")

    return errors, warnings


def normalize_team_label(value: Any) -> str:
    """Map a team field onto the canonical anonymous vocabulary.

    Lossless aliases only. Roster club labels (HOME / AWAY) and any unrecognized
    value are returned unchanged, because inventing a HOME -> TEAM_A equivalence
    would fabricate a classification result.
    """
    if value is None:
        return TEAM_UNKNOWN
    if not isinstance(value, str):
        raise SchemaError(f"team_id must be a string or None, got {value!r}")
    if value in TEAM_LABEL_ALIASES:
        return TEAM_LABEL_ALIASES[value]
    if value in NON_CANONICAL_TEAM_LABELS_ACCEPTED:
        return value
    return value.strip()


def is_canonical_team_label(value: Any) -> bool:
    return isinstance(value, str) and value in CANONICAL_TEAM_LABELS


def normalize_detection_record(
    row: dict[str, Any],
    *,
    assign_id: bool = True,
    row_ordinal: int = 0,
) -> dict[str, Any]:
    """Return a copy of ``row`` carrying only canonical vocabulary.

    Adds ``detection_id`` when absent, normalizes ``team_id``, and promotes
    deprecated aliases to their canonical names while keeping the legacy keys
    so existing readers continue to work.

    This is a single-row helper: it always uses occurrence ``0``, so batch
    callers must use :func:`assign_detection_ids` to obtain the occurrence
    counter and the collision check. ``row_ordinal`` only matters when the row
    has no usable ``track_id``, in which case the ordinal becomes the tagged
    ``N`` key.
    """
    if not isinstance(row, dict):
        raise SchemaError("Cannot normalize a non-mapping record")
    normalized = dict(row)

    if assign_id and not is_valid_detection_id(normalized.get("detection_id")):
        frame = _require_int(normalized.get("frame", 0), "frame")
        normalized["detection_id"] = build_detection_id(
            frame, normalized.get("track_id"), 0, row_ordinal
        )

    if "team_id" in normalized:
        normalized["team_id"] = normalize_team_label(normalized["team_id"])

    for alias, canonical in CANONICAL_FIELD_ALIASES.items():
        if alias in normalized:
            normalized.setdefault(canonical, normalized[alias])
            if canonical != "player_id":
                normalized[alias] = normalized[canonical]

    if "player_id" in normalized:
        player = normalized["player_id"]
        if player in (None, ""):
            normalized["player_id"] = PLAYER_ID_UNRESOLVED

    return normalized


def validate_rows(
    rows: Sequence[dict[str, Any]],
    *,
    strict: bool = False,
) -> dict[str, Any]:
    """Validate a batch of detection records and summarize the result."""
    all_errors: list[str] = []
    all_warnings: list[str] = []
    team_counts: Counter[str] = Counter()
    player_counts: Counter[str] = Counter()

    for index, row in enumerate(rows):
        errors, warnings = validate_detection_record(
            row,
            strict=strict,
            context=f"row {index}",
        )
        all_errors.extend(errors)
        all_warnings.extend(warnings)
        if isinstance(row, dict):
            team_counts[str(row.get("team_id", "<absent>"))] += 1
            player_counts[str(row.get("player_id", "<absent>"))] += 1

    return {
        "schema_version": SCHEMA_VERSION,
        "level": "strict" if strict else "required",
        "records": len(rows),
        "valid": not all_errors,
        "error_count": len(all_errors),
        "warning_count": len(all_warnings),
        "errors": all_errors[:50],
        "warnings": all_warnings[:50],
        "team_id_values": dict(team_counts),
        "player_id_values": dict(player_counts),
        "detection_id": detection_id_accounting(rows),
    }


def validate_crop_index_samples(
    samples: Sequence[dict[str, Any]],
    *,
    strict: bool = True,
) -> dict[str, Any]:
    """Validate ``crop_index.json`` / clustering sample records."""
    errors: list[str] = []
    missing_ids = 0
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict):
            errors.append(f"sample {index}: not a mapping")
            continue
        for name in ("track_id", "frame", "detector_confidence"):
            if name not in sample:
                errors.append(f"sample {index}: missing {name}")
        if "detection_id" not in sample:
            missing_ids += 1
            if strict:
                errors.append(f"sample {index}: missing detection_id")
        elif not is_valid_detection_id(sample["detection_id"]):
            errors.append(
                f"sample {index}: malformed detection_id {sample['detection_id']!r}"
            )

    return {
        "schema_version": SCHEMA_VERSION,
        "samples": len(samples),
        "valid": not errors,
        "missing_detection_id": missing_ids,
        "errors": errors[:50],
        "error_count": len(errors),
        "detection_id": detection_id_accounting(
            [s for s in samples if isinstance(s, dict)]
        ),
    }


def schema_descriptor() -> dict[str, Any]:
    """Machine-readable description of the contract, for manifests and reports."""
    return {
        "schema_version": SCHEMA_VERSION,
        "detection_id_format": "D{frame:06d}_{kind}{key:06d}_{seq:03d}, kind in T|N",
        "detection_id_pattern": DETECTION_ID_PATTERN.pattern,
        "detection_id_key_kinds": {
            TRACK_KEY_FROM_TRACKER: "key is the ByteTrack track_id",
            TRACK_KEY_FROM_ORDINAL: "key is the row ordinal, used only when no usable track_id exists",
        },
        "detection_id_rule": (
            "The key is tagged T when derived from a track_id and N when derived "
            "from the row ordinal, so the two key sources cannot collide. seq is "
            "the 0-based occurrence count of the (frame, kind, key) triple within "
            "the run, which makes the full ID unique even if a tracker emits the "
            "same ID twice in one frame."
        ),
        "team_vocabulary": list(CANONICAL_TEAM_LABELS),
        "team_label_aliases": dict(TEAM_LABEL_ALIASES),
        "non_canonical_team_labels_accepted": sorted(NON_CANONICAL_TEAM_LABELS_ACCEPTED),
        "player_id_unresolved": PLAYER_ID_UNRESOLVED,
        "detection_fields": list(DETECTION_FIELDS),
        "detection_identity_fields": list(DETECTION_IDENTITY_FIELDS),
        "reserved_fields": sorted(RESERVED_FIELDS),
        "deprecated_aliases": {
            alias: canonical for alias, canonical in CANONICAL_FIELD_ALIASES.items()
        },
        "fields": {
            name: {"types": [t.__name__ for t in types], "description": note}
            for name, (types, _required_in, note) in FIELD_REGISTRY.items()
        },
    }


def schema_descriptor_json(*, indent: int = 2) -> str:
    return json.dumps(schema_descriptor(), indent=indent, sort_keys=True)
