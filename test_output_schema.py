"""
Unit and integration tests for the Task 1.2 output contract.

Covers:
- canonical field registry and team vocabulary;
- deterministic, collision-free detection_id construction;
- legacy input compatibility (files and crop indexes without detection_id);
- downstream compatibility (review_team_crops_v2, run_team_clustering, run.resolve);
- strict/required validation levels and the UNK vs UNKNOWN separation.
"""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import output_schema as os_
from output_schema import (
    CANONICAL_TEAM_LABELS,
    PLAYER_ID_UNRESOLVED,
    SCHEMA_VERSION,
    TEAM_A,
    TEAM_B,
    TEAM_UNKNOWN,
    SchemaError,
    assign_detection_ids,
    build_detection_id,
    detection_id_accounting,
    find_duplicate_detection_ids,
    is_canonical_team_label,
    is_valid_detection_id,
    load_detection_rows,
    normalize_detection_record,
    normalize_team_label,
    schema_descriptor,
    validate_crop_index_samples,
    validate_detection_record,
    validate_rows,
    write_jsonl,
)


def legacy_row(frame=10, track_id=3, conf=0.7, bbox=(100, 100, 130, 180)):
    """A row in the exact shape run.py wrote before Task 1.2."""
    return {
        "frame": frame,
        "timestamp_s": frame / 15.0,
        "fragment_id": "F000001",
        "track_id": track_id,
        "bbox_xyxy": [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
        "detector_confidence": conf,
        "class_id": 0,
    }


class DetectionIdTests(unittest.TestCase):
    def test_format_is_canonical_and_sortable(self):
        self.assertEqual(build_detection_id(123, 42, 0), "D000123_T000042_000")
        self.assertTrue(is_valid_detection_id(build_detection_id(0, 0, 0)))
        self.assertRegex(build_detection_id(7, 9, 2), r"^D\d{6}_T\d{6}_\d{3}$")
        # Lexicographic sort equals frame sort because frame is the leading field.
        ids = [build_detection_id(f, 1, 0) for f in (5, 1, 12, 120)]
        self.assertEqual(sorted(ids), [build_detection_id(f, 1, 0) for f in (1, 5, 12, 120)])

    def test_ordinal_key_is_tagged_and_revalidates(self):
        self.assertEqual(build_detection_id(123, None, 0, row_ordinal=42), "D000123_N000042_000")
        self.assertTrue(is_valid_detection_id(build_detection_id(1, None, 0, row_ordinal=1)))

    def test_wide_track_ids_still_validate(self):
        self.assertTrue(is_valid_detection_id(build_detection_id(1, 1_234_567, 0)))

    def test_construction_is_deterministic(self):
        first = [legacy_row(frame=f, track_id=t) for f, t in [(0, 1), (0, 2), (5, 1)]]
        second = [legacy_row(frame=f, track_id=t) for f, t in [(0, 1), (0, 2), (5, 1)]]
        self.assertEqual(
            [r["detection_id"] for r in assign_detection_ids(first)],
            [r["detection_id"] for r in assign_detection_ids(second)],
        )

    def test_duplicate_frame_track_pairs_get_distinct_occurrence(self):
        rows = [legacy_row(frame=4, track_id=9) for _ in range(3)]
        assigned = assign_detection_ids(rows)
        ids = [r["detection_id"] for r in assigned]
        self.assertEqual(ids, ["D000004_T000009_000", "D000004_T000009_001", "D000004_T000009_002"])
        self.assertEqual(find_duplicate_detection_ids(assigned), [])

    def test_uniqueness_over_many_rows(self):
        rows = []
        for frame in range(200):
            for track_id in (1, 2, 3):
                rows.append(legacy_row(frame=frame, track_id=track_id))
        # Force collisions that a (frame, track_id) key alone would not survive.
        rows.append(legacy_row(frame=100, track_id=2))
        rows.append(legacy_row(frame=100, track_id=2))
        assigned = assign_detection_ids(rows)
        self.assertEqual(len(assigned), 602)
        self.assertEqual(len({r["detection_id"] for r in assigned}), 602)
        self.assertEqual(find_duplicate_detection_ids(assigned), [])

    def test_existing_ids_are_preserved(self):
        rows = [legacy_row(), legacy_row(frame=11)]
        rows[0]["detection_id"] = "D000010_T000003_000"
        assigned = assign_detection_ids(rows)
        self.assertEqual(assigned[0]["detection_id"], "D000010_T000003_000")
        self.assertEqual(assigned[1]["detection_id"], "D000011_T000003_000")

    def test_assignment_is_idempotent(self):
        rows = [legacy_row(frame=f, track_id=f % 4) for f in range(20)]
        once = assign_detection_ids(rows)
        ids_first = [r["detection_id"] for r in once]
        twice = assign_detection_ids(once)
        self.assertEqual(ids_first, [r["detection_id"] for r in twice])

    def test_overwrite_reassigns_from_scratch(self):
        rows = [legacy_row()]
        rows[0]["detection_id"] = "D999999_T999999_999"
        assigned = assign_detection_ids(rows, overwrite=True)
        self.assertEqual(assigned[0]["detection_id"], "D000010_T000003_000")

    def test_malformed_existing_id_is_rejected(self):
        rows = [legacy_row()]
        rows[0]["detection_id"] = "not-an-id"
        with self.assertRaises(SchemaError):
            assign_detection_ids(rows)

    def test_missing_track_id_falls_back_to_tagged_row_ordinal(self):
        row = legacy_row(frame=7)
        del row["track_id"]
        rows = [row, legacy_row(frame=8)]
        rows[1].pop("track_id", None)
        assigned = assign_detection_ids(rows)
        self.assertEqual(assigned[0]["detection_id"], "D000007_N000000_000")
        self.assertEqual(assigned[1]["detection_id"], "D000008_N000001_000")
        self.assertEqual(find_duplicate_detection_ids(assigned), [])

    def test_untagged_legacy_format_is_rejected(self):
        # The pre-tag design wrote D{frame}_{key}_{seq}; those are no longer valid.
        self.assertFalse(is_valid_detection_id("D000010_000003_000"))
        with self.assertRaises(SchemaError):
            assign_detection_ids([dict(legacy_row(), detection_id="D000010_000003_000")])

    def test_invalid_frame_is_rejected(self):
        with self.assertRaises(SchemaError):
            build_detection_id(-1, 1, 0)
        with self.assertRaises(SchemaError):
            build_detection_id(1.5, 1, 0)
        with self.assertRaises(SchemaError):
            build_detection_id(1, 1, -1)

    def test_accounting_reports_coverage(self):
        rows = [legacy_row(), legacy_row(frame=1)]
        rows[1]["detection_id"] = None
        summary = detection_id_accounting(assign_detection_ids(rows[:1]) + rows[1:])
        self.assertEqual(summary["total_records"], 2)
        self.assertEqual(summary["records_with_detection_id"], 1)
        self.assertEqual(summary["missing_detection_id"], 1)


class DetectionIdCollisionTests(unittest.TestCase):
    """Regression tests for the ordinal-vs-tracker key namespace collision."""

    def _rows_with_ordinal_42_clash(self):
        # 42 filler rows first so that the untracked row genuinely lands on
        # ordinal 42, then both contenders share frame 7 and the number 42.
        rows = [legacy_row(frame=0, track_id=1000 + index) for index in range(42)]
        untracked = legacy_row(frame=7)
        del untracked["track_id"]
        rows.append(untracked)                      # ordinal 42, no track_id
        rows.append(legacy_row(frame=7, track_id=42))  # real track_id 42
        return rows, 42, 43

    def test_ordinal_42_and_track_42_in_same_frame_get_different_ids(self):
        rows, untracked_at, tracked_at = self._rows_with_ordinal_42_clash()
        assigned = assign_detection_ids(rows)

        self.assertEqual(untracked_at, 42, msg="fixture must put the row on ordinal 42")
        self.assertEqual(assigned[untracked_at].get("track_id"), None)
        self.assertEqual(assigned[tracked_at]["track_id"], 42)

        # The numeric keys are identical; only the tag distinguishes them.
        self.assertEqual(
            assigned[untracked_at]["detection_id"], "D000007_N000042_000"
        )
        self.assertEqual(
            assigned[tracked_at]["detection_id"], "D000007_T000042_000"
        )
        self.assertNotEqual(
            assigned[untracked_at]["detection_id"], assigned[tracked_at]["detection_id"]
        )

        ids = [row["detection_id"] for row in assigned]
        self.assertEqual(len(ids), len(set(ids)), msg="detection_id collision")
        self.assertEqual(find_duplicate_detection_ids(assigned), [])

        # The untracked row is still not contract-valid at strict level: an
        # ordinal key keeps the ID unique but does not supply a track_id.
        tracked_errors, _ = validate_detection_record(assigned[tracked_at], strict=True)
        self.assertEqual(tracked_errors, [], msg=str(tracked_errors))
        untracked_errors, _ = validate_detection_record(assigned[untracked_at], strict=True)
        self.assertEqual(untracked_errors, ["detection: missing required field 'track_id'"])

    def test_assignment_is_still_deterministic_for_fixed_row_sequence(self):
        rows, _, _ = self._rows_with_ordinal_42_clash()
        first = [r["detection_id"] for r in assign_detection_ids(rows)]
        rows2, _, _ = self._rows_with_ordinal_42_clash()
        second = [r["detection_id"] for r in assign_detection_ids(rows2)]
        self.assertEqual(first, second)

    def test_key_kinds_are_disjoint(self):
        self.assertEqual(os_.detection_key_of(42, 7), ("T", 42))
        self.assertEqual(os_.detection_key_of(None, 42), ("N", 42))
        self.assertEqual(os_.detection_key_of("not-a-number", 42), ("N", 42))
        self.assertEqual(os_.detection_key_of(True, 42), ("N", 42))
        self.assertEqual(os_.detection_key_of(-5, 42), ("N", 42))
        # Same numeric key, different namespace.
        self.assertEqual(os_.detection_key_of(42, 0)[1], os_.detection_key_of(None, 42)[1])
        self.assertNotEqual(os_.detection_key_of(42, 0)[0], os_.detection_key_of(None, 42)[0])

    def test_negative_and_boolean_track_ids_cannot_collide_with_tracker_ids(self):
        rows = [
            legacy_row(frame=9, track_id=42),   # real tracker ID 42
            legacy_row(frame=9, track_id=None),  # unusable -> ordinal key
            legacy_row(frame=9, track_id=-1),    # unusable -> ordinal key
            legacy_row(frame=9, track_id=True),  # bool is not a valid ID
        ]
        assigned = assign_detection_ids(rows)
        ids = [r["detection_id"] for r in assigned]
        self.assertEqual(ids[0], "D000009_T000042_000")
        self.assertEqual(ids[1], "D000009_N000001_000")
        self.assertEqual(ids[2], "D000009_N000002_000")
        self.assertEqual(ids[3], "D000009_N000003_000")
        self.assertEqual(len(ids), len(set(ids)), msg=str(ids))
        self.assertTrue(all(is_valid_detection_id(value) for value in ids))

    def test_preexisting_id_clash_fails_loudly(self):
        # Two rows already carrying the same ID.
        duplicated = [
            dict(legacy_row(frame=7, track_id=1), detection_id="D000007_T000042_000"),
            dict(legacy_row(frame=8, track_id=2), detection_id="D000007_T000042_000"),
        ]
        with self.assertRaises(SchemaError):
            assign_detection_ids(duplicated)

        # A pre-existing ID that a synthesized ID would collide with.
        mixed = [
            dict(legacy_row(frame=7, track_id=99), detection_id="D000007_T000042_000"),
            legacy_row(frame=7, track_id=42),
        ]
        with self.assertRaises(SchemaError):
            assign_detection_ids(mixed)


class TeamVocabularyTests(unittest.TestCase):
    def test_canonical_vocabulary_is_exactly_three_labels(self):
        self.assertEqual(CANONICAL_TEAM_LABELS, ("TEAM_A", "TEAM_B", "UNKNOWN"))
        for label in CANONICAL_TEAM_LABELS:
            self.assertTrue(is_canonical_team_label(label))
            self.assertEqual(normalize_team_label(label), label)

    def test_player_sentinel_is_not_a_team_label(self):
        self.assertEqual(PLAYER_ID_UNRESOLVED, "UNK")
        self.assertNotIn(PLAYER_ID_UNRESOLVED, CANONICAL_TEAM_LABELS)
        # The old team sentinel normalizes to UNKNOWN, never to a player ID.
        self.assertEqual(normalize_team_label("UNK"), TEAM_UNKNOWN)

    def test_aliases_normalize_losslessly(self):
        cases = {
            None: TEAM_UNKNOWN,
            "": TEAM_UNKNOWN,
            "UNK": TEAM_UNKNOWN,
            "unk": TEAM_UNKNOWN,
            "unknown": TEAM_UNKNOWN,
            "team_a": TEAM_A,
            "Team_B": TEAM_B,
        }
        for raw, expected in cases.items():
            self.assertEqual(normalize_team_label(raw), expected, msg=repr(raw))

    def test_roster_club_labels_are_not_coerced(self):
        # Inventing HOME -> TEAM_A would fabricate a classification result.
        self.assertEqual(normalize_team_label("HOME"), "HOME")
        self.assertEqual(normalize_team_label("AWAY"), "AWAY")
        self.assertFalse(is_canonical_team_label("HOME"))

    def test_whitespace_is_stripped(self):
        self.assertEqual(normalize_team_label("  TEAM_A "), TEAM_A)

    def test_non_string_team_label_rejected(self):
        with self.assertRaises(SchemaError):
            normalize_team_label(3)

    def test_roster_labels_are_never_coerced_in_either_validation_level(self):
        for strict in (False, True):
            for roster_label in ("HOME", "AWAY"):
                row = normalize_detection_record(legacy_row())
                row["team_id"] = roster_label
                self.assertEqual(normalize_team_label(roster_label), roster_label)
                errors, warnings = validate_detection_record(row, strict=strict)
                self.assertEqual(errors, [], msg=f"{roster_label} strict={strict}: {errors}")
                self.assertTrue(
                    any("outside the anonymous team vocabulary" in w for w in warnings),
                    msg=f"{roster_label} strict={strict} should warn",
                )

    def test_unmappable_team_label_is_a_strict_error_only(self):
        row = normalize_detection_record(legacy_row())
        row["team_id"] = "TEAM_C"
        errors, warnings = validate_detection_record(row, strict=False)
        self.assertEqual(errors, [])
        self.assertTrue(any("outside the anonymous team vocabulary" in w for w in warnings))

        errors, _ = validate_detection_record(row, strict=True)
        self.assertTrue(any("outside the anonymous team vocabulary" in e for e in errors))

    def test_normalized_legacy_sentinel_row_becomes_strict_clean(self):
        row = legacy_row()
        row["team_id"] = "UNK"
        errors, warnings = validate_detection_record(row, strict=True)
        self.assertTrue(any("legacy player sentinel" in e for e in errors))

        normalized = normalize_detection_record(row)
        self.assertEqual(normalized["team_id"], TEAM_UNKNOWN)
        strict_errors, _ = validate_detection_record(normalized, strict=True)
        self.assertEqual(strict_errors, [], msg=str(strict_errors))

    def test_player_id_holding_a_team_label_is_an_error(self):
        row = normalize_detection_record(legacy_row())
        row["player_id"] = TEAM_A
        row["team_id"] = TEAM_UNKNOWN
        errors, _ = validate_detection_record(row, strict=True)
        self.assertTrue(any("player_id" in e and "team label" in e for e in errors))


class RecordValidationTests(unittest.TestCase):
    def test_legacy_row_passes_required_level_with_warnings(self):
        errors, warnings = validate_detection_record(legacy_row(), strict=False)
        self.assertEqual(errors, [])
        self.assertTrue(any("detection_id" in w for w in warnings))

    def test_normalized_row_passes_strict_level(self):
        row = normalize_detection_record(legacy_row())
        errors, warnings = validate_detection_record(row, strict=True)
        self.assertEqual(errors, [], msg=str(errors))
        self.assertTrue(is_valid_detection_id(row["detection_id"]))

    def test_legacy_team_sentinel_is_strict_error(self):
        row = normalize_detection_record(legacy_row())
        row["team_id"] = "UNK"
        errors, _ = validate_detection_record(row, strict=True)
        self.assertTrue(any("legacy player sentinel" in e for e in errors))

    def test_roster_club_label_warns_but_does_not_fail_strict(self):
        row = normalize_detection_record(legacy_row())
        row["team_id"] = "HOME"
        errors, warnings = validate_detection_record(row, strict=True)
        self.assertEqual(errors, [], msg=str(errors))
        self.assertTrue(any("outside the anonymous team vocabulary" in w for w in warnings))

    def test_person_uid_must_not_be_emitted(self):
        row = normalize_detection_record(legacy_row())
        row["person_uid"] = "P1"
        errors, _ = validate_detection_record(row, strict=True)
        self.assertTrue(any("person_uid is reserved" in e for e in errors))

    def test_missing_required_fields_are_errors(self):
        for field in ("frame", "track_id", "bbox_xyxy"):
            row = legacy_row()
            del row[field]
            errors, _ = validate_detection_record(row, strict=False)
            self.assertTrue(any("missing required field" in e for e in errors), msg=field)

    def test_bad_types_and_bbox_are_errors(self):
        row = normalize_detection_record(legacy_row())
        row["track_id"] = "3"
        errors, _ = validate_detection_record(row, strict=True)
        self.assertTrue(any("must be an integer" in e for e in errors))

        row = normalize_detection_record(legacy_row())
        row["bbox_xyxy"] = [100.0, 100.0, 90.0, 180.0]
        errors, _ = validate_detection_record(row, strict=True)
        self.assertTrue(any("bbox_xyxy must be ordered" in e for e in errors))

        row = normalize_detection_record(legacy_row())
        row["bbox_xyxy"] = [float("nan"), 0.0, 10.0, 20.0]
        errors, _ = validate_detection_record(row, strict=True)
        self.assertTrue(any("finite" in e for e in errors))

    def test_deprecated_aliases_are_warned_and_backfilled(self):
        row = normalize_detection_record(legacy_row())
        row["confidence"] = 0.91
        row["reason"] = "insufficient_or_conflicting_evidence"
        _, warnings = validate_detection_record(row, strict=True)
        self.assertTrue(any("deprecated field 'confidence'" in w for w in warnings))

        normalized = normalize_detection_record(row)
        self.assertEqual(normalized["identification_confidence"], 0.91)
        self.assertEqual(
            normalized["identification_reason"], "insufficient_or_conflicting_evidence"
        )
        # Legacy keys are retained so existing readers keep working.
        self.assertEqual(normalized["confidence"], 0.91)

    def test_batch_validation_summary(self):
        rows = [normalize_detection_record(legacy_row(frame=f)) for f in range(5)]
        for row in rows:
            row["team_id"] = TEAM_UNKNOWN
            row["player_id"] = PLAYER_ID_UNRESOLVED
        summary = validate_rows(rows, strict=True)
        self.assertTrue(summary["valid"])
        self.assertEqual(summary["records"], 5)
        self.assertEqual(summary["team_id_values"], {TEAM_UNKNOWN: 5})
        self.assertEqual(summary["player_id_values"], {PLAYER_ID_UNRESOLVED: 5})
        self.assertEqual(summary["detection_id"]["unique_detection_ids"], 5)

    def test_non_mapping_row_is_reported(self):
        errors, _ = validate_detection_record(["not", "a", "dict"], strict=True)
        self.assertTrue(any("not a mapping" in e for e in errors))


class LegacyFileCompatibilityTests(unittest.TestCase):
    def test_legacy_jsonl_loads_with_backfilled_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "boxes.jsonl"
            rows = [legacy_row(frame=f, track_id=f % 3) for f in range(12)]
            path.write_text("\n".join(json.dumps(r) for r in rows) + "\n\n", encoding="utf-8")

            loaded = load_detection_rows(path)
            self.assertEqual(len(loaded), 12)
            self.assertEqual(len({r["detection_id"] for r in loaded}), 12)
            self.assertTrue(all(is_valid_detection_id(r["detection_id"]) for r in loaded))

            # Re-reading the same file reproduces identical IDs.
            self.assertEqual(
                [r["detection_id"] for r in loaded],
                [r["detection_id"] for r in load_detection_rows(path)],
            )

    def test_written_current_file_round_trips_without_reassignment(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "boxes.jsonl"
            rows = [normalize_detection_record(legacy_row(frame=f)) for f in range(6)]
            write_jsonl(rows, path)
            reloaded = load_detection_rows(path)
            self.assertEqual(
                [r["detection_id"] for r in rows], [r["detection_id"] for r in reloaded]
            )

    def test_incomplete_legacy_row_is_rejected_with_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "boxes.jsonl"
            row = legacy_row()
            del row["track_id"]
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(SchemaError, "missing required fields"):
                load_detection_rows(path)

    def test_degenerate_bbox_is_rejected_on_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "boxes.jsonl"
            row = legacy_row()
            row["bbox_xyxy"] = [100.0, 100.0, 100.0, 180.0]
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaises(SchemaError):
                load_detection_rows(path)


class DownstreamCompatibilityTests(unittest.TestCase):
    def test_review_tool_accepts_legacy_rows_without_detection_id(self):
        from crop_filter import CropFilterPolicy
        from review_team_crops_v2 import filter_detections, load_detections

        # filter_detections is the function existing callers already use; it must
        # keep working on rows that carry no detection_id at all.
        detections = [legacy_row(frame=0, track_id=1), legacy_row(frame=0, track_id=2, conf=0.2)]
        by_track, stats = filter_detections(detections, CropFilterPolicy(), 1280, 720)
        self.assertIn(1, by_track)
        self.assertNotIn(2, by_track)
        self.assertEqual(stats.unique_accepted, 1)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "boxes.jsonl"
            path.write_text(
                "\n".join(json.dumps(legacy_row(frame=f, track_id=1)) for f in (0, 1, 2)),
                encoding="utf-8",
            )
            loaded = load_detections(path)
            self.assertTrue(all("detection_id" in r for r in loaded))
            self.assertEqual(
                find_duplicate_detection_ids(loaded), []
            )

    def test_clustering_reconstructs_ids_for_legacy_crop_index(self):
        from crop_filter import CropFilterPolicy
        from run_team_clustering import load_and_filter_samples

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            index = root / "crop_index.json"
            samples = []
            for track_id, frame in ((1, 30), (1, 45), (2, 60)):
                crop_path = root / f"c_{track_id}_{frame}.jpg"
                samples.append({
                    "file": str(crop_path),
                    "track_id": track_id,
                    "frame": frame,
                    "timestamp_s": frame / 15.0,
                    "detector_confidence": 0.8,
                    "box_width": 30.0,
                    "box_height": 80.0,
                    "crop_width": 20,
                    "crop_height": 30,
                })
            index.write_text(
                json.dumps({
                    "video_width": 1280,
                    "video_height": 720,
                    "filter_policy": CropFilterPolicy().to_dict(),
                    "samples": samples,
                }),
                encoding="utf-8",
            )

            report, retained = load_and_filter_samples(index, CropFilterPolicy())
            self.assertEqual(len(retained), 3)
            self.assertEqual(report["detection_id_source"], "reconstructed_from_samples")
            self.assertTrue(all(is_valid_detection_id(s["detection_id"]) for s in retained))

            # The reconstructed ID must equal what run.py would have written for
            # the same detection.
            self.assertEqual(
                retained[0]["detection_id"], build_detection_id(30, 1, 0)
            )

    def test_clustering_keeps_native_ids_from_new_crop_index(self):
        from crop_filter import CropFilterPolicy
        from run_team_clustering import load_and_filter_samples

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            index = root / "crop_index.json"
            samples = []
            for track_id, frame in ((4, 15), (4, 20), (5, 25)):
                row = normalize_detection_record(legacy_row(frame=frame, track_id=track_id))
                samples.append({
                    "file": str(root / f"c_{track_id}_{frame}.jpg"),
                    "detection_id": row["detection_id"],
                    "track_id": track_id,
                    "frame": frame,
                    "detector_confidence": 0.8,
                    "box_width": 30.0,
                    "box_height": 80.0,
                    "crop_width": 20,
                    "crop_height": 30,
                })
            index.write_text(
                json.dumps({
                    "schema_version": SCHEMA_VERSION,
                    "video_width": 1280,
                    "video_height": 720,
                    "filter_policy": CropFilterPolicy().to_dict(),
                    "samples": samples,
                }),
                encoding="utf-8",
            )

            report, retained = load_and_filter_samples(index, CropFilterPolicy())
            self.assertEqual(report["detection_id_source"], "crop_index")
            self.assertEqual(
                [s["detection_id"] for s in retained], [s["detection_id"] for s in samples]
            )
            self.assertEqual(report["index_schema_version"], SCHEMA_VERSION)

    def test_crop_index_sample_validation(self):
        rows = [normalize_detection_record(legacy_row(frame=f, track_id=f)) for f in (1, 2)]
        samples = [{"file": "a.jpg", "detection_id": r["detection_id"],
                    "track_id": r["track_id"], "frame": r["frame"],
                    "detector_confidence": r["detector_confidence"]} for r in rows]
        summary = validate_crop_index_samples(samples, strict=True)
        self.assertTrue(summary["valid"], msg=str(summary["errors"]))
        self.assertEqual(summary["detection_id"]["unique_detection_ids"], 2)

        broken = [dict(samples[0])]
        broken[0]["detection_id"] = "garbage"
        del broken[0]["frame"]
        summary = validate_crop_index_samples(broken, strict=True)
        self.assertFalse(summary["valid"])
        self.assertGreaterEqual(summary["error_count"], 2)

        # Legacy indexes simply report missing IDs instead of failing hard.
        legacy = [{k: v for k, v in samples[0].items() if k != "detection_id"}]
        summary = validate_crop_index_samples(legacy, strict=False)
        self.assertTrue(summary["valid"])
        self.assertEqual(summary["missing_detection_id"], 1)

    def test_new_samples_keep_every_key_the_legacy_sheet_tool_reads(self):
        # make_filtered_team_sheets.py reads exactly these keys from a crop
        # index sample; a contract-compliant sample must still provide them.
        from crop_filter import CropFilterPolicy

        row = normalize_detection_record(legacy_row(frame=30, track_id=4))
        sample = {
            "file": "crops/track_0004_frame_000030.jpg",
            "detection_id": row["detection_id"],
            "track_id": row["track_id"],
            "frame": row["frame"],
            "timestamp_s": row["timestamp_s"],
            "detector_confidence": row["detector_confidence"],
            "box_width": 30.0,
            "box_height": 80.0,
            "crop_width": 20,
            "crop_height": 30,
        }
        for key in ("box_width", "box_height", "detector_confidence", "file", "track_id", "frame"):
            self.assertIn(key, sample, msg=key)

        result = CropFilterPolicy().evaluate_sample_record(sample, 1280, 720)
        self.assertTrue(result.is_valid, msg=result.reasons)

        # The legacy tool's own filter constants accept the same record.
        self.assertGreaterEqual(sample["detector_confidence"], 0.45)
        self.assertLessEqual(sample["box_height"], 160)
        self.assertLessEqual(sample["box_width"] / sample["box_height"], 0.90)


class RunResolverVocabularyTests(unittest.TestCase):
    def test_unresolved_fragment_gets_unknown_team_and_unk_player(self):
        from run import resolve

        fragment = {
            "fragment_id": "F1",
            "numbers": [],
            "embedding": np.array([1.0, 0.0]),
            "start_s": 0,
            "end_s": 1,
            "start_frame": 0,
            "end_frame": 30,
        }
        resolve([fragment], {"players": []}, {}, 0.88, 0.08)
        self.assertEqual(fragment["player_id"], PLAYER_ID_UNRESOLVED)
        self.assertEqual(fragment["team_id"], TEAM_UNKNOWN)
        self.assertEqual(fragment["identification_reason"], fragment["reason"])
        self.assertEqual(fragment["identification_confidence"], fragment["confidence"])

    def test_resolved_fragment_keeps_roster_team_label_verbatim(self):
        from run import resolve

        fragment = {
            "fragment_id": "F1",
            "numbers": [],
            "embedding": np.array([1.0, 0.0]),
            "start_s": 0,
            "end_s": 1,
            "start_frame": 0,
            "end_frame": 30,
        }
        roster = {"players": [{"player_id": "P1", "team_id": "HOME", "jersey": 10}]}
        resolve([fragment], roster, {"P1": np.array([1.0, 0.0])}, 0.88, 0.08)
        self.assertEqual(fragment["player_id"], "P1")
        # HOME is roster metadata; it is not rewritten into an anonymous label.
        self.assertEqual(fragment["team_id"], "HOME")
        self.assertEqual(fragment["identification_confidence"], 1.0)


class SchemaDescriptorTests(unittest.TestCase):
    def test_descriptor_is_json_serializable_and_complete(self):
        descriptor = schema_descriptor()
        text = json.dumps(descriptor, indent=2, sort_keys=True)
        self.assertIn("detection_id_format", text)
        self.assertEqual(descriptor["schema_version"], SCHEMA_VERSION)
        self.assertEqual(descriptor["team_vocabulary"], list(CANONICAL_TEAM_LABELS))
        self.assertEqual(descriptor["player_id_unresolved"], PLAYER_ID_UNRESOLVED)
        self.assertIn("person_uid", descriptor["reserved_fields"])
        self.assertNotIn("person_uid", descriptor["detection_fields"])

    def test_descriptor_helper_emits_json(self):
        parsed = json.loads(os_.schema_descriptor_json(indent=None))
        self.assertEqual(parsed["schema_version"], SCHEMA_VERSION)

    def test_every_registry_field_is_documented(self):
        for name, (types, _required_in, note) in os_.FIELD_REGISTRY.items():
            self.assertTrue(types, msg=name)
            self.assertTrue(note, msg=name)


if __name__ == "__main__":
    unittest.main()
