"""Synthetic cross-file identity invariants, not football accuracy evidence."""

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from identity_association import ACCEPTANCE_EXPLANATION, _person_uid, associate_fragments
from validate_tracking_outputs import analyse_identity_associations


VIDEO_SHA = 'a' * 64
BOXES_SHA = 'b' * 64


class IdentityOutputTests(unittest.TestCase):
    def setUp(self):
        self.uid = _person_uid(VIDEO_SHA, BOXES_SHA, ['F1', 'F2'])
        self.fragments = [self.fragment('F1', 0, 5), self.fragment('F2', 6, 10)]
        self.manifest = {'schema_version': '1.1', 'artifact_schema_versions': {
            'boxes.jsonl': '1.0', 'identified_boxes.jsonl': '1.1',
            'fragments.json': '1.1', 'detection_accounting.jsonl': '1.0'}}
        self.rows = [{**f, 'detection_id': f'D{i}'} for i, f in enumerate(self.fragments)]
        self.report = {'decisions': copy.deepcopy(self.fragments), 'groups': [], 'pairs': [],
                       'provenance': {'video_sha256': VIDEO_SHA, 'boxes_sha256': BOXES_SHA,
                                      'review': None}}

    def fragment(self, fid, start, end):
        return {'fragment_id': fid, 'schema_version': '1.1', 'start_frame': start,
                'team_id': 'UNKNOWN', 'team_classification': {'team_id': 'UNKNOWN', 'score': 0.0},
                'end_frame': end, 'person_uid': None, 'association_status': 'unresolved',
                'association_reason': 'no_review', 'association_method': 'abstain'}

    def sync(self):
        self.rows = [{**f, 'detection_id': f'D{i}'} for i, f in enumerate(self.fragments)]
        self.report['decisions'] = copy.deepcopy(self.fragments)

    def associate(self):
        for fragment in self.fragments:
            fragment.update(person_uid=self.uid, association_status='associated',
                            association_reason='reviewed_same_person',
                            association_method='human_reviewed_pair_clique')
        self.sync()
        self.report['groups'] = [{'person_uid': self.uid, 'fragment_ids': ['F1', 'F2']}]
        self.report['pairs'] = [{'left': 'F1', 'right': 'F2', 'decision': 'same',
                                 'accepted': True, 'reasons': []}]

    def validate(self, **kwargs):
        return analyse_identity_associations(self.fragments, self.rows, self.report,
                                             self.manifest, **kwargs)

    def test_legacy_is_not_claimed_as_identity_validation(self):
        result = analyse_identity_associations([{'fragment_id': 'F1'}], [], None)
        self.assertFalse(result['applicable'])
        self.assertTrue(result['valid'])

    def test_unresolved_and_positive_group_are_valid(self):
        self.assertTrue(self.validate()['valid'])
        self.associate()
        result = self.validate()
        self.assertTrue(result['valid'], result['errors'])
        self.assertEqual(result['associated_fragments'], 2)

    def test_missing_report_fails_for_enriched_output(self):
        self.report = None
        self.assertFalse(self.validate()['valid'])

    def test_row_fragment_decision_drift_fails(self):
        for field, value in [('person_uid', self.uid), ('association_reason', 'changed'),
                             ('association_method', 'unknown'), ('schema_version', '1.0')]:
            with self.subTest(field=field):
                self.sync()
                self.rows[0][field] = value
                self.assertFalse(self.validate()['valid'])

    def test_duplicate_and_missing_report_decisions_fail(self):
        for decisions in [[], self.fragments[:1], self.fragments + self.fragments[:1], None]:
            self.report['decisions'] = decisions
            self.assertFalse(self.validate()['valid'])

    def test_report_decision_disagreement_fails(self):
        self.report['decisions'][0]['association_reason'] = 'different'
        self.assertFalse(self.validate()['valid'])

    def test_duplicate_fragment_ids_fail(self):
        self.fragments.append(copy.deepcopy(self.fragments[0]))
        self.assertFalse(self.validate()['valid'])

    def test_group_must_contain_multiple_unique_known_members(self):
        self.associate()
        for members in [['F1'], ['F1', 'F1'], ['F1', 'absent'], None, [[], 'F1']]:
            self.report['groups'][0]['fragment_ids'] = members
            self.assertFalse(self.validate()['valid'])

    def test_inclusive_overlap_is_not_allowed(self):
        self.associate()
        self.fragments[1]['start_frame'] = 5
        self.sync()
        result = self.validate()
        self.assertFalse(result['valid'])
        self.assertIn('inclusive temporal overlap', ' '.join(result['errors']))

    def test_every_group_pair_requires_direct_accepted_evidence(self):
        self.associate()
        third = {**self.fragments[-1], 'fragment_id': 'F3', 'start_frame': 11, 'end_frame': 15}
        self.fragments.append(third)
        self.sync()
        self.report['groups'][0]['fragment_ids'].append('F3')
        self.report['pairs'].append({'left': 'F2', 'right': 'F3', 'decision': 'same',
                                     'accepted': True, 'reasons': []})
        result = self.validate()
        self.assertFalse(result['valid'])
        self.assertIn('missing accepted direct evidence', ' '.join(result['errors']))

    def test_accepted_pair_cannot_have_vetoes_or_different_judgment(self):
        self.associate()
        for change in [{'reasons': ['overlap']}, {'decision': 'different'}, {'accepted': 1}]:
            self.report['pairs'][0] = {'left': 'F1', 'right': 'F2', 'decision': 'same',
                                       'accepted': True, 'reasons': [], **change}
            self.assertFalse(self.validate()['valid'])

    def test_duplicate_pair_evidence_is_rejected(self):
        self.associate()
        self.report['pairs'].append({'left': 'F2', 'right': 'F1', 'decision': 'same',
                                     'accepted': True, 'reasons': []})
        self.assertFalse(self.validate()['valid'])

    def test_uid_requires_matching_group(self):
        self.associate()
        self.report['groups'] = []
        self.assertFalse(self.validate()['valid'])

    def test_unresolved_fragment_cannot_belong_to_group(self):
        self.report['groups'] = [{'person_uid': self.uid, 'fragment_ids': ['F1', 'F2']}]
        self.assertFalse(self.validate()['valid'])

    def test_manifest_migration_is_explicit(self):
        for manifest in [{}, {'schema_version': '1.0'},
                         {'schema_version': '1.1', 'artifact_schema_versions': None}]:
            self.manifest = manifest
            self.assertFalse(self.validate()['valid'])

    def test_validation_does_not_mutate_inputs(self):
        self.associate()
        before = copy.deepcopy((self.fragments, self.rows, self.report, self.manifest))
        self.validate()
        self.assertEqual(before, (self.fragments, self.rows, self.report, self.manifest))

    def test_provenance_rebinds_to_the_current_input_digests(self):
        self.associate()
        result = self.validate(provenance_digests={'video_sha256': VIDEO_SHA,
                                                   'boxes_sha256': BOXES_SHA})
        self.assertTrue(result['valid'], result['errors'])
        for key in ('video_sha256', 'boxes_sha256'):
            with self.subTest(key=key):
                result = self.validate(provenance_digests={key: 'c' * 64})
                self.assertFalse(result['valid'])
                self.assertIn('stale provenance', ' '.join(result['errors']))

    def test_group_without_provenance_is_rejected(self):
        self.associate()
        del self.report['provenance']
        result = self.validate()
        self.assertFalse(result['valid'])
        self.assertIn('require report provenance', ' '.join(result['errors']))

    def test_malformed_provenance_digest_is_rejected(self):
        self.associate()
        for digest in ('A' * 64, 'a' * 63, '', None, 7):
            with self.subTest(digest=digest):
                self.report['provenance']['video_sha256'] = digest
                result = self.validate()
                self.assertFalse(result['valid'])
                self.assertIn('lower-case SHA256', ' '.join(result['errors']))

    def test_forged_uid_survives_field_consistency_but_fails_recomputation(self):
        # Adversarial: fragments, decisions and groups all agree on a UID that
        # is not the deterministic digest of provenance digests plus members.
        self.uid = 'P' + '0' * 32
        self.associate()
        result = self.validate(provenance_digests={'video_sha256': VIDEO_SHA,
                                                   'boxes_sha256': BOXES_SHA})
        self.assertFalse(result['valid'])
        self.assertIn('deterministic digest', ' '.join(result['errors']))
        genuine = self.validate()
        self.assertEqual(genuine['error_count'], 1, genuine['errors'])


class ValidatorWriteSafetyTests(unittest.TestCase):
    def test_refuses_existing_reports_and_new_files_inside_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / 'run'
            run.mkdir()
            existing = root / 'existing.json'
            existing.write_text('preserve this evidence', encoding='utf-8')
            for out in (existing, run / 'new.json'):
                result = subprocess.run(
                    [sys.executable, '-B', str(Path(__file__).with_name('validate_tracking_outputs.py')),
                     '--video', str(root / 'absent.mp4'), '--run-dir', str(run), '--out', str(out)],
                    capture_output=True, text=True, check=False)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('error:', result.stderr)
            self.assertEqual(existing.read_text(encoding='utf-8'), 'preserve this evidence')
            self.assertFalse((run / 'new.json').exists())


class EngineReportIntegrationTests(unittest.TestCase):
    """Positive test using real association-engine output, not empty-reason fixtures."""

    VIDEO = hashlib.sha256(b"synthetic-video").hexdigest()
    BOXES = hashlib.sha256(b"synthetic-boxes").hexdigest()

    def engine_fragment(self, fid, start, end):
        return {'fragment_id': fid, 'track_id': 3, 'start_frame': start, 'end_frame': end,
                'embedding': [1.0, 0.0, 0.0],
                'appearance_evidence': {'status': 'ok', 'embedding_generated': True},
                'team_id': 'UNKNOWN',
                'team_classification': {'team_id': 'UNKNOWN', 'score': 0.0},
                'player_id': 'UNK'}

    def test_engine_accepted_pair_report_passes_the_validator(self):
        fragments = [self.engine_fragment('F1', 0, 5), self.engine_fragment('F2', 6, 10)]
        review = {
            'kind': 'human_fragment_identity_review', 'video_sha256': self.VIDEO,
            'boxes_sha256': self.BOXES, 'reviewer': 'Synthetic policy reviewer',
            'reviewed_at': '2026-11-15T00:00:00Z',
            'fragments': [{'fragment_id': f['fragment_id'], 'single_person': True,
                           'full_span_reviewed': True, 'shot_id': 'synthetic-shot-1'}
                          for f in fragments],
            'pairs': [{'left': 'F1', 'right': 'F2', 'decision': 'same',
                       'reason': 'Synthetic policy evidence, not real accuracy.'}],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'review.json'
            path.write_text(json.dumps(review, allow_nan=False), encoding='utf-8')
            report = associate_fragments(fragments, video_sha256=self.VIDEO,
                                         boxes_sha256=self.BOXES, review_path=path)
        accepted = [p for p in report['pairs'] if p['accepted']]
        self.assertEqual(len(accepted), 1)
        self.assertEqual(accepted[0]['reasons'], [ACCEPTANCE_EXPLANATION])
        self.assertEqual(len(report['groups']), 1)
        rows = [{**f, 'detection_id': f'D00000{i}'} for i, f in enumerate(fragments)]
        manifest = {'schema_version': '1.1', 'artifact_schema_versions': {
            'boxes.jsonl': '1.0', 'identified_boxes.jsonl': '1.1',
            'fragments.json': '1.1', 'detection_accounting.jsonl': '1.0'}}
        result = analyse_identity_associations(
            fragments, rows, report, manifest,
            provenance_digests={'video_sha256': self.VIDEO, 'boxes_sha256': self.BOXES})
        self.assertTrue(result['valid'], result['errors'])
        self.assertEqual(result['associated_fragments'], 2)
        self.assertEqual(result['groups'], 1)

    def test_validator_still_rejects_acceptance_flag_with_veto_reasons(self):
        fragments = [self.engine_fragment('F1', 0, 5), self.engine_fragment('F2', 6, 10)]
        report = {'decisions': copy.deepcopy(fragments), 'groups': [],
                  'pairs': [{'left': 'F1', 'right': 'F2', 'decision': 'same',
                             'accepted': True,
                             'reasons': [ACCEPTANCE_EXPLANATION, 'inclusive_frame_overlap']}]}
        result = analyse_identity_associations(fragments, [], report, None)
        self.assertFalse(result['valid'])
        self.assertIn('no vetoes', ' '.join(result['errors']))


if __name__ == '__main__':
    unittest.main()
