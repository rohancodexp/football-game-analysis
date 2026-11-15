import copy
import json
import tempfile
import unittest
from pathlib import Path

import output_schema as schema


UID = "P0123456789abcdef0123456789abcdef"
INVALID_VERSIONS = (None, "", "1", "1.2", "2.0", " 1.1", 1, 1.0, 1.1, True, [], {})
IDENTITY_FIELDS = ("person_uid", "association_status", "association_reason", "association_method")


def identity_fragment(associated=False):
    return {
        "schema_version": "1.1",
        "fragment_id": "F000001",
        "person_uid": UID if associated else None,
        "association_status": "associated" if associated else "unresolved",
        "association_reason": "reviewed pairs accepted" if associated else "no human review",
        "association_method": "human_reviewed_pair_clique" if associated else "abstain",
    }


def legacy_detection(frame=0):
    return {
        "frame": frame,
        "timestamp_s": frame / 30.0,
        "detection_id": schema.build_detection_id(frame, 1),
        "track_id": 1,
        "fragment_id": "F000001",
        "bbox_xyxy": [10.0, 20.0, 30.0, 80.0],
        "detector_confidence": 0.9,
        "class_id": 0,
        "player_id": "UNK",
        "team_id": "UNKNOWN",
        "jersey": None,
        "identification_confidence": 0.0,
        "identification_reason": "unresolved roster identity",
    }


def enriched_detection(associated=False, frame=0):
    return dict(legacy_detection(frame), **identity_fragment(associated), team_score=0.0)


class IdentityFieldTests(unittest.TestCase):
    def test_enriched_identity_and_team_sentinels_cannot_be_missing(self):
        for key, invalid in [('team_id', (None, '', 'HOME', 'UNK', [], {})),
                             ('player_id', (None, '', [], {}))]:
            row = enriched_detection()
            del row[key]
            self.assertTrue(schema.validate_detection_record(row, strict=True)[0])
            for value in invalid:
                with self.subTest(key=key, value=value):
                    row[key] = value
                    self.assertTrue(schema.validate_detection_record(row, strict=True)[0])

    def test_associated_and_unresolved_fragments_are_valid(self):
        for associated in (False, True):
            with self.subTest(associated=associated):
                self.assertEqual(schema.validate_identity_fields(identity_fragment(associated)), [])

    def test_helper_is_independent_of_detection_and_team_fields(self):
        fragment = identity_fragment(True)
        fragment.update(frame=[], bbox_xyxy={}, team_score="not a detection score")
        self.assertEqual(schema.validate_identity_fields(fragment), [])
        self.assertTrue(schema.validate_detection_record(fragment)[0])

    def test_helper_requires_explicit_enriched_version(self):
        fragment = identity_fragment()
        del fragment["schema_version"]
        self.assertTrue(any("schema_version" in e for e in schema.validate_identity_fields(fragment)))
        for version in ("1.0", *INVALID_VERSIONS):
            with self.subTest(version=version):
                fragment["schema_version"] = version
                self.assertTrue(any("schema_version" in e for e in schema.validate_identity_fields(fragment)))

    def test_every_extension_field_is_required_even_nullable_uid(self):
        for associated in (False, True):
            for field in IDENTITY_FIELDS:
                with self.subTest(associated=associated, field=field):
                    fragment = identity_fragment(associated)
                    del fragment[field]
                    errors = schema.validate_identity_fields(fragment)
                    self.assertTrue(any("missing required field" in e and field in e for e in errors), errors)

    def test_uid_requires_exact_canonical_format(self):
        invalid = (
            "P1", "", "p" + "a" * 32, "P" + "A" * 32,
            "P" + "a" * 31, "P" + "a" * 33, "P" + "g" * 32,
            "P" + "０" * 32, UID + "\n", " " + UID, UID + " ",
            123, True, [], {},
        )
        for uid in invalid:
            with self.subTest(uid=uid):
                fragment = dict(identity_fragment(True), person_uid=uid)
                self.assertTrue(any("person_uid" in e for e in schema.validate_identity_fields(fragment)))
        for uid in (UID, "P" + "0" * 32, "P" + "f" * 32):
            self.assertEqual(schema.validate_identity_fields(dict(identity_fragment(True), person_uid=uid)), [])

    def test_associated_uid_cannot_be_null(self):
        errors = schema.validate_identity_fields(dict(identity_fragment(True), person_uid=None))
        self.assertTrue(any("person_uid" in e for e in errors))

    def test_unresolved_uid_must_be_null(self):
        for uid in (UID, "", "UNK", [], {}):
            with self.subTest(uid=uid):
                errors = schema.validate_identity_fields(dict(identity_fragment(), person_uid=uid))
                self.assertTrue(any("person_uid" in e for e in errors))

    def test_status_requires_known_string(self):
        for status in (None, "", "ASSOCIATED", "unknown", "abstain", 1, True, [], {}):
            with self.subTest(status=status):
                errors = schema.validate_identity_fields(dict(identity_fragment(), association_status=status))
                self.assertTrue(any("association_status" in e for e in errors))

    def test_methods_must_match_status(self):
        for associated, wrong_method in ((True, "abstain"), (False, "human_reviewed_pair_clique")):
            with self.subTest(associated=associated):
                fragment = dict(identity_fragment(associated), association_method=wrong_method)
                self.assertTrue(any("association_method" in e for e in schema.validate_identity_fields(fragment)))

    def test_method_requires_known_string(self):
        for method in (None, "", "automatic", "ABSTAIN", 1, True, [], {}):
            with self.subTest(method=method):
                fragment = dict(identity_fragment(), association_method=method)
                self.assertTrue(any("association_method" in e for e in schema.validate_identity_fields(fragment)))

    def test_reason_requires_nonempty_string(self):
        for reason in (None, "", " \t\n", 0, True, [], {}):
            with self.subTest(reason=reason):
                fragment = dict(identity_fragment(), association_reason=reason)
                self.assertTrue(any("association_reason" in e for e in schema.validate_identity_fields(fragment)))
        self.assertEqual(schema.validate_identity_fields(dict(identity_fragment(), association_reason=" review ")), [])

    def test_context_and_nonmapping_errors(self):
        for record in (None, [], [identity_fragment()], "record", 1):
            with self.subTest(record=record):
                errors = schema.validate_identity_fields(record, "fragment F9")
                self.assertTrue(errors)
                self.assertTrue(all(e.startswith("fragment F9:") for e in errors))
        errors = schema.validate_identity_fields({}, context="fragment F9")
        self.assertTrue(errors)
        self.assertTrue(all(e.startswith("fragment F9:") for e in errors))

    def test_validation_does_not_mutate_fragment(self):
        for associated in (False, True):
            fragment = identity_fragment(associated)
            before = copy.deepcopy(fragment)
            schema.validate_identity_fields(fragment)
            self.assertEqual(fragment, before)


class EnrichedDetectionTests(unittest.TestCase):
    def test_enriched_detections_pass_both_levels(self):
        for strict in (False, True):
            for associated in (False, True):
                with self.subTest(strict=strict, associated=associated):
                    row = enriched_detection(associated)
                    before = copy.deepcopy(row)
                    self.assertEqual(schema.validate_detection_record(row, strict=strict), ([], []))
                    self.assertEqual(row, before)

    def test_legacy_and_explicit_legacy_remain_valid(self):
        for strict in (False, True):
            for row in (legacy_detection(), dict(legacy_detection(), schema_version="1.0")):
                self.assertEqual(schema.validate_detection_record(row, strict=strict), ([], []))

    def test_legacy_uid_is_forbidden_including_null(self):
        for strict in (False, True):
            for explicit in (False, True):
                for uid in (None, UID, "P1", [], {}):
                    with self.subTest(strict=strict, explicit=explicit, uid=uid):
                        row = dict(legacy_detection(), person_uid=uid)
                        if explicit:
                            row["schema_version"] = "1.0"
                        errors, _ = schema.validate_detection_record(row, strict=strict)
                        self.assertTrue(any("person_uid is reserved" in e for e in errors))

    def test_unsupported_and_wrong_type_versions_are_errors(self):
        for strict in (False, True):
            for version in INVALID_VERSIONS:
                with self.subTest(strict=strict, version=version):
                    row = dict(enriched_detection(), schema_version=version)
                    errors, _ = schema.validate_detection_record(row, strict=strict)
                    self.assertTrue(any("schema_version" in e for e in errors))

    def test_enriched_extension_is_required_at_both_levels(self):
        for strict in (False, True):
            for field in IDENTITY_FIELDS:
                with self.subTest(strict=strict, field=field):
                    row = enriched_detection()
                    del row[field]
                    errors, _ = schema.validate_detection_record(row, strict=strict)
                    self.assertTrue(any(field in e for e in errors))

    def test_detection_uses_identity_validation(self):
        for field, value in (
            ("person_uid", "P1"), ("association_status", "unknown"),
            ("association_method", "automatic"), ("association_reason", ""),
        ):
            for strict in (False, True):
                with self.subTest(field=field, strict=strict):
                    row = dict(enriched_detection(), **{field: value})
                    identity_errors = schema.validate_identity_fields(row, context="detection")
                    detection_errors, _ = schema.validate_detection_record(row, strict=strict)
                    self.assertTrue(identity_errors)
                    self.assertTrue(set(identity_errors).issubset(detection_errors))

    def test_team_score_is_required_only_for_enriched_detections(self):
        row = enriched_detection()
        del row["team_score"]
        for strict in (False, True):
            errors, _ = schema.validate_detection_record(row, strict=strict)
            self.assertTrue(any("team_score" in e for e in errors))
            for legacy in (legacy_detection(), dict(legacy_detection(), schema_version="1.0", team_score=[])):
                self.assertEqual(schema.validate_detection_record(legacy, strict=strict)[0], [])

    def test_team_score_rejects_nonfinite_out_of_range_and_nonnumeric(self):
        for score in (None, True, False, "0.5", [], {}, -0.01, 1.01, float("nan"), float("inf"), -float("inf")):
            for strict in (False, True):
                with self.subTest(score=score, strict=strict):
                    row = dict(enriched_detection(), team_id="TEAM_A", team_score=score)
                    errors, _ = schema.validate_detection_record(row, strict=strict)
                    self.assertTrue(any("team_score" in e for e in errors))

    def test_known_team_score_accepts_range_and_endpoints(self):
        for team in ("TEAM_A", "TEAM_B", "HOME", "AWAY"):
            for score in (0, 0.0, 0.5, 1, 1.0):
                with self.subTest(team=team, score=score):
                    row = dict(enriched_detection(), team_id=team, team_score=score)
                    self.assertEqual(schema.validate_detection_record(row, strict=True)[0], [])

    def test_unknown_team_requires_zero_score(self):
        for strict in (False, True):
            for score in (0, 0.0, -0.0):
                row = dict(enriched_detection(), team_score=score)
                self.assertEqual(schema.validate_detection_record(row, strict=strict)[0], [])
            for score in (0.01, 0.5, 1):
                row = dict(enriched_detection(), team_score=score)
                errors, _ = schema.validate_detection_record(row, strict=strict)
                self.assertTrue(any("UNKNOWN" in e and "team_score" in e for e in errors))

    def test_malformed_containers_return_errors_without_exceptions(self):
        fields = ("schema_version", *IDENTITY_FIELDS, "team_score", "team_id", "player_id",
                  "frame", "track_id", "bbox_xyxy", "detection_id")
        for strict in (False, True):
            for field in fields:
                for value in ([], {}):
                    with self.subTest(strict=strict, field=field, value=value):
                        row = dict(enriched_detection(), **{field: value})
                        self.assertTrue(schema.validate_detection_record(row, strict=strict)[0])
                        self.assertFalse(schema.validate_rows([row], strict=strict)["valid"])


class VersionMigrationTests(unittest.TestCase):
    def test_batches_expose_encountered_versions_separately(self):
        for rows, versions in (
            ([], []),
            ([legacy_detection()], ["1.0"]),
            ([legacy_detection(), dict(legacy_detection(1), schema_version="1.0")], ["1.0"]),
            ([enriched_detection(), enriched_detection(True, 1)], ["1.1"]),
        ):
            with self.subTest(versions=versions):
                report = schema.validate_rows(rows, strict=True)
                self.assertTrue(report["valid"], report["errors"])
                self.assertEqual(report["schema_version"], "1.0")
                self.assertEqual(report["row_schema_versions"], versions)
                self.assertEqual(report["records"], len(rows))
                self.assertEqual(report["detection_id"]["unique_detection_ids"], len(rows))

    def test_mixed_artifact_versions_are_rejected(self):
        for strict in (False, True):
            for explicit in (False, True):
                legacy = legacy_detection()
                if explicit:
                    legacy["schema_version"] = "1.0"
                rows = [enriched_detection(True, 1), legacy]
                for ordered in (rows, list(reversed(rows))):
                    report = schema.validate_rows(ordered, strict=strict)
                    self.assertFalse(report["valid"])
                    self.assertEqual(report["row_schema_versions"], ["1.0", "1.1"])
                    self.assertTrue(any("mixed schema versions" in e for e in report["errors"]))

    def test_invalid_version_values_are_reported_without_coercion(self):
        for version in INVALID_VERSIONS:
            with self.subTest(version=version):
                report = schema.validate_rows([dict(legacy_detection(), schema_version=version)])
                self.assertFalse(report["valid"])
                self.assertEqual(report["row_schema_versions"], [version])
                json.dumps(report)
        report = schema.validate_rows([None, [], {}, dict(legacy_detection(), schema_version={})])
        self.assertFalse(report["valid"])
        json.dumps(report)

    def test_version_stamp_alone_is_not_migration(self):
        row = dict(legacy_detection(), schema_version="1.1")
        errors, _ = schema.validate_detection_record(row)
        for field in (*IDENTITY_FIELDS, "team_score"):
            self.assertTrue(any(field in e for e in errors))
        row.update(identity_fragment(), team_score=0.0)
        self.assertEqual(schema.validate_detection_record(row, strict=True)[0], [])
        for version in (None, "1.0"):
            downgraded = dict(row)
            if version is None:
                downgraded.pop("schema_version")
            else:
                downgraded["schema_version"] = version
            errors, _ = schema.validate_detection_record(downgraded)
            self.assertTrue(any("person_uid is reserved" in e for e in errors))

    def test_enriched_jsonl_round_trip_preserves_detection_and_identity(self):
        rows = [enriched_detection(False), enriched_detection(True, 1)]
        before = copy.deepcopy(rows)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "identified_boxes.jsonl"
            schema.write_jsonl(rows, path)
            loaded = schema.load_detection_rows(path)
        self.assertEqual(loaded, before)
        self.assertEqual(rows, before)
        self.assertTrue(schema.validate_rows(loaded, strict=True)["valid"])
        for row in loaded:
            self.assertEqual(schema.normalize_detection_record(row), row)
            legacy = legacy_detection(row["frame"])
            self.assertEqual({key: row[key] for key in legacy}, legacy)


class VersionedDescriptorTests(unittest.TestCase):
    def test_versions_and_default_descriptor_stay_legacy(self):
        self.assertEqual(schema.SCHEMA_VERSION, "1.0")
        self.assertEqual(schema.ENRICHED_SCHEMA_VERSION, "1.1")
        self.assertEqual(schema.ACCOUNTING_VERSION, "1.0")
        legacy = schema.schema_descriptor()
        self.assertEqual(legacy, schema.schema_descriptor("1.0"))
        self.assertEqual(legacy, json.loads(schema.schema_descriptor_json()))
        self.assertEqual(legacy["reserved_fields"], ["person_uid"])
        self.assertEqual(legacy["fields"]["person_uid"]["types"], ["str"])
        for field in ("team_score", "association_status", "association_reason", "association_method"):
            self.assertNotIn(field, legacy["fields"])
        self.assertNotIn("person_uid", legacy["detection_identity_fields"])

    def test_enriched_descriptor_describes_extension_and_score(self):
        descriptor = schema.schema_descriptor(version="1.1")
        self.assertEqual(json.loads(json.dumps(descriptor)), descriptor)
        self.assertEqual(descriptor["schema_version"], "1.1")
        self.assertNotIn("person_uid", descriptor["reserved_fields"])
        self.assertEqual(descriptor["fields"]["person_uid"]["types"], ["str", "NoneType"])
        self.assertRegex(UID, descriptor["person_uid_pattern"])
        self.assertEqual(descriptor["required_identity_fields"], ["schema_version", *IDENTITY_FIELDS])
        for field in (*IDENTITY_FIELDS, "team_score"):
            self.assertIn(field, descriptor["fields"])
            self.assertIn(field, descriptor["detection_identity_fields"])
            self.assertIn(field, descriptor["required_detection_fields"])
        self.assertEqual(descriptor["detection_fields"], list(schema.DETECTION_FIELDS))
        self.assertEqual(descriptor["player_id_unresolved"], "UNK")

    def test_enriched_descriptor_does_not_mutate_legacy_constants(self):
        legacy = schema.schema_descriptor()
        registry = copy.deepcopy(schema.FIELD_REGISTRY)
        reserved = schema.RESERVED_FIELDS
        identity_fields = schema.DETECTION_IDENTITY_FIELDS
        descriptor = schema.schema_descriptor("1.1")
        expected = copy.deepcopy(descriptor)
        descriptor["fields"]["person_uid"]["types"].append("int")
        descriptor["fields"]["frame"]["description"] = "changed"
        descriptor["team_label_aliases"].clear()
        descriptor["detection_fields"].clear()
        descriptor["detection_identity_fields"].clear()
        self.assertEqual(schema.schema_descriptor("1.1"), expected)
        self.assertEqual(schema.schema_descriptor(), legacy)
        self.assertEqual(schema.FIELD_REGISTRY, registry)
        self.assertEqual(schema.RESERVED_FIELDS, reserved)
        self.assertEqual(schema.DETECTION_IDENTITY_FIELDS, identity_fields)

    def test_descriptor_rejects_unsupported_and_malformed_versions(self):
        for version in INVALID_VERSIONS:
            with self.subTest(version=version):
                with self.assertRaises(schema.SchemaError):
                    schema.schema_descriptor(version)


if __name__ == "__main__":
    unittest.main()
