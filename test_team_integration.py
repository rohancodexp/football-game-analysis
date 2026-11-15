"""Stage 4A: independent team classification wired into run.py.

Synthetic kits are uniform colour patches plus pixel noise. "Two teams" fixtures use
two genuinely separate kit colours; nothing here asserts a two-team split on data that
does not contain two groups.
"""

import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from crop_filter import CropFilterPolicy, FilterStats
from output_schema import (
    PLAYER_ID_UNRESOLVED,
    TEAM_A,
    TEAM_B,
    TEAM_UNKNOWN,
    build_detection_id,
    validate_detection_record,
    validate_rows,
)
from run import (
    TEAM_MIN_SAMPLES,
    TEAM_RANDOM_STATE,
    TEAM_SCORE_KIND,
    TEAM_STATUS_CLUSTERING_REJECTED,
    TEAM_STATUS_CONFLICTING_VOTES,
    TEAM_STATUS_FRAGMENT_MAJORITY,
    TEAM_STATUS_INSUFFICIENT_CROPS,
    TEAM_STATUS_INSUFFICIENT_VOTES,
    TEAM_STATUS_NO_CROPS,
    TEAM_STATUS_ROLE_CANDIDATE,
    aggregate_fragment_team,
    build_accounting_record,
    build_team_classifier,
    classify_fragment_teams,
    collect_team_crop,
    new_fragment,
    release_team_crops,
    reliable_fragment_team,
    resolve,
    team_row_fields,
    write_team_crop_index,
)
from team_classifier import (
    ROLE_NONE,
    ROLE_REFEREE_CANDIDATE,
    RoleCandidate,
    TeamPrediction,
)

FRAME_H, FRAME_W = 720, 1280
BOX = (600.0, 300.0, 640.0, 400.0)  # 40x100: passes every CropFilterPolicy default

RED = (40, 40, 200)      # BGR
BLUE = (200, 60, 30)
WHITE = (235, 235, 235)
YELLOW = (0, 230, 240)   # fluorescent referee yellow

IDENTITY_KEYS = ('player_id', 'identification_confidence', 'confidence', 'reason',
                 'identification_reason', 'jersey')
TRACKING_KEYS = ('fragment_id', 'track_id', 'start_frame', 'end_frame', 'start_s', 'end_s')


def kit_frame(bgr, rng, box=BOX, noise=6.0):
    frame = np.full((FRAME_H, FRAME_W, 3), (40, 140, 50), dtype=np.uint8)
    x1, y1, x2, y2 = map(int, box)
    patch = np.full((y2 - y1, x2 - x1, 3), bgr, dtype=np.float32)
    patch += rng.normal(0.0, noise, patch.shape)
    frame[y1:y2, x1:x2] = np.clip(patch, 0, 255).astype(np.uint8)
    return frame


def make_fragment(fid, track_id, start_frame, length=20, fps=15.0):
    f = new_fragment(fid, track_id, start_frame)
    for key in ('_detections', '_sample_attempts', '_valid', '_invalid',
                '_encoding_failures', '_vectors'):
        f.pop(key)
    f['end_frame'] = start_frame + length - 1
    f['start_s'] = start_frame / fps
    f['end_s'] = f['end_frame'] / fps
    f['embedding'] = None
    f['appearance_evidence'] = {'detections': length, 'embedding_generated': False}
    return f


class Stats:
    def __init__(self):
        self.policy = CropFilterPolicy()
        self.box = FilterStats()
        self.crop = FilterStats()


def collect(fragment, colour, n, rng, stats, box=BOX, confidence=0.8):
    accepted = 0
    for i in range(n):
        frame_idx = fragment['start_frame'] + 5 * i
        meta = {'frame': frame_idx, 'timestamp_s': frame_idx / 15.0,
                'detection_id': build_detection_id(frame_idx, fragment['track_id'], 0),
                'detector_confidence': confidence}
        accepted += collect_team_crop(fragment, kit_frame(colour, rng, box), box, meta,
                                      policy=stats.policy, box_stats=stats.box,
                                      crop_stats=stats.crop, max_crops=8)
    return accepted


def two_kit_fragments(per_team=8, crops=4, seed=0, extra=()):
    """per_team fragments of RED and of BLUE; ``extra`` adds (colour, crops) fragments."""
    rng = np.random.default_rng(seed)
    stats = Stats()
    fragments, truth = [], {}
    plan = [(RED, crops)] * per_team + [(BLUE, crops)] * per_team + list(extra)
    for i, (colour, n) in enumerate(plan):
        f = make_fragment(i, 100 + i, 30 * i)
        collect(f, colour, n, rng, stats)
        fragments.append(f)
        truth[i] = colour
    resolve(fragments, {'players': []}, {}, 0.88, 0.08)
    return fragments, truth, stats


SYNTHETIC_CALIBRATION = {
    'kind': 'synthetic_logic_fixture_not_football_calibration',
    'thresholds': {'min_centroid_distance': .5, 'max_within_radius': .15,
                   'min_cluster_fraction': .25, 'min_mean_margin': .3,
                   'min_seed_agreement': 1., 'max_outlier_fraction': .25,
                   'min_crop_score': .05, 'min_crop_margin': .1},
}


def classify(fragments, stats, **overrides):
    kwargs = dict(policy=stats.policy, box_stats=stats.box, crop_stats=stats.crop,
                  collection_seconds=0.0, min_votes=2, min_majority=0.60,
                  max_per_fragment_fit=2, candidate_mode='multi', candidate_action='flag',
                  calibration=SYNTHETIC_CALIBRATION)
    kwargs.update(overrides)
    return classify_fragment_teams(fragments, **kwargs)


def rows_for(fragments):
    """Mirror the identified_boxes writer: identity fields, then team_row_fields."""
    rows = []
    for f in fragments:
        for frame in range(f['start_frame'], f['end_frame'] + 1, 7):
            row = {'frame': frame, 'timestamp_s': frame / 15.0,
                   'detection_id': build_detection_id(frame, f['track_id'], 0),
                   'track_id': f['track_id'], 'fragment_id': str(f['fragment_id']),
                   'bbox_xyxy': list(BOX), 'detector_confidence': 0.8, 'class_id': 0,
                   'player_id': f['player_id'], 'jersey': f['jersey'],
                   'identification_confidence': f['identification_confidence'],
                   'identification_reason': f['identification_reason']}
            row.update(team_row_fields(f))
            rows.append(row)
    return rows


def prediction(team, confidence=0.8, reason='nearest_team_appearance_cluster',
               role=ROLE_NONE, score=0.0, margin=.5):
    candidate = RoleCandidate(is_candidate=role != ROLE_NONE, role=role, score=score)
    return TeamPrediction(team_id=team, confidence=confidence, distance=0.1,
                          reason=reason, candidate=candidate, margin=margin)


class NoRosterTests(unittest.TestCase):
    """Tests 1 and 3."""

    def test_two_genuine_kits_are_separated_without_any_roster(self):
        fragments, truth, stats = two_kit_fragments()
        report = classify(fragments, stats)
        self.assertTrue(report['fit']['fitted'], report['fit']['error'])
        by_colour = {}
        for f in fragments:
            by_colour.setdefault(truth[f['fragment_id']], set()).add(f['team_id'])
        self.assertEqual(len(by_colour[RED]), 1)
        self.assertEqual(len(by_colour[BLUE]), 1)
        self.assertEqual(by_colour[RED] | by_colour[BLUE], {TEAM_A, TEAM_B})
        self.assertEqual(report['accuracy_claim'], None)

    def test_player_id_stays_unk_while_team_is_assigned(self):
        fragments, _, stats = two_kit_fragments()
        classify(fragments, stats)
        pairs = {(f['player_id'], f['team_id']) for f in fragments}
        self.assertIn((PLAYER_ID_UNRESOLVED, TEAM_A), pairs)
        self.assertIn((PLAYER_ID_UNRESOLVED, TEAM_B), pairs)
        self.assertTrue(all(f['player_id'] == PLAYER_ID_UNRESOLVED for f in fragments))

    def test_labels_do_not_depend_on_roster_state(self):
        bare, _, stats_a = two_kit_fragments(seed=3)
        with_roster, _, stats_b = two_kit_fragments(seed=3)
        proto = np.zeros(512, dtype=np.float32)
        proto[0] = 1.0
        with_roster[0]['embedding'] = proto
        roster = {'players': [{'player_id': 'P7', 'jersey': 7, 'team_id': 'HOME'}]}
        resolve(with_roster, roster, {'P7': proto}, 0.88, 0.08)
        self.assertEqual(with_roster[0]['player_id'], 'P7')
        self.assertEqual(with_roster[0]['team_id'], 'HOME')
        report_a = classify(bare, stats_a)
        report_b = classify(with_roster, stats_b)
        self.assertEqual([f['team_id'] for f in bare], [f['team_id'] for f in with_roster])
        self.assertEqual(report_a['fit']['cluster_fingerprint'],
                         report_b['fit']['cluster_fingerprint'])
        self.assertEqual(with_roster[0]['team_classification']['roster_inherited_team_id'], 'HOME')
        self.assertIn(with_roster[0]['team_id'], (TEAM_A, TEAM_B))


class PropagationTests(unittest.TestCase):
    """Test 2."""

    def test_fragment_label_reaches_every_detection_row(self):
        fragments, _, stats = two_kit_fragments()
        report = classify(fragments, stats)
        rows = rows_for(fragments)
        by_fragment = {str(f['fragment_id']): f for f in fragments}
        for row in rows:
            f = by_fragment[row['fragment_id']]
            self.assertEqual(row['team_id'], f['team_id'])
            self.assertEqual(row['team_evidence_status'],
                             f['team_classification']['evidence_status'])
            self.assertEqual(row['team_score_kind'], TEAM_SCORE_KIND)
            self.assertEqual(row['team_cluster_fingerprint'], report['fit']['cluster_fingerprint'])
        self.assertEqual({r['team_id'] for r in rows}, {TEAM_A, TEAM_B})

    def test_score_is_labelled_as_a_heuristic_not_a_probability(self):
        fragments, _, stats = two_kit_fragments()
        classify(fragments, stats)
        for f in fragments:
            record = f['team_classification']
            self.assertEqual(record['score_kind'], 'distance_to_radius_heuristic_not_probability')
            self.assertGreaterEqual(record['score'], 0.0)
            self.assertLessEqual(record['score'], 1.0)


class ScoreContractTests(unittest.TestCase):
    def test_team_a_score_uses_only_winning_crops(self):
        record = aggregate_fragment_team(
            [prediction(TEAM_A, .9), prediction(TEAM_A, .7), prediction(TEAM_B, .2)],
            min_votes=2, min_majority=.6)
        self.assertEqual(record['team_id'], TEAM_A)
        self.assertEqual(record['score'], .8)

    def test_team_b_score_uses_only_winning_crops(self):
        record = aggregate_fragment_team(
            [prediction(TEAM_A, .1), prediction(TEAM_B, .6), prediction(TEAM_B, .8)],
            min_votes=2, min_majority=.6)
        self.assertEqual(record['team_id'], TEAM_B)
        self.assertEqual(record['score'], .7)

    def test_unknown_aggregation_paths_have_zero_score(self):
        cases = {
            'empty': [],
            'abstained': [prediction(TEAM_UNKNOWN, 0)],
            'insufficient': [prediction(TEAM_A, .9)],
            'conflict': [prediction(TEAM_A, .9), prediction(TEAM_B, .8)],
            'excluded': [prediction(TEAM_A, .8, role=ROLE_REFEREE_CANDIDATE, score=.9)] * 3,
        }
        for name, predictions in cases.items():
            with self.subTest(name=name):
                record = aggregate_fragment_team(predictions, min_votes=2, min_majority=.6)
                self.assertEqual(record['team_id'], TEAM_UNKNOWN)
                self.assertEqual(record['score'], 0.0)
                self.assertEqual(team_row_fields({'team_classification': record})['team_score'], 0.0)

    def test_role_exclusion_preserves_pre_exclusion_score(self):
        record = aggregate_fragment_team(
            [prediction(TEAM_B, .8, role=ROLE_REFEREE_CANDIDATE, score=.9)] * 3,
            min_votes=2, min_majority=.6)
        self.assertEqual(record['team_id'], TEAM_UNKNOWN)
        self.assertEqual(record['score'], 0.0)
        self.assertEqual(record['score_before_candidate_exclusion'], .8)
        self.assertEqual(record['team_id_before_candidate_exclusion'], TEAM_B)

    def test_insufficient_clustering_evidence_has_zero_scores(self):
        fragments, _, stats = two_kit_fragments(per_team=1)
        report = classify(fragments, stats)
        self.assertFalse(report['fit']['fitted'])
        for fragment in fragments:
            self.assertEqual(fragment['team_id'], TEAM_UNKNOWN)
            self.assertEqual(fragment['team_classification']['score'], 0.0)


class IdentityIsolationTests(unittest.TestCase):
    """Test 4."""

    def test_identity_and_tracking_fields_are_untouched(self):
        fragments, _, stats = two_kit_fragments()
        proto = np.zeros(512, dtype=np.float32)
        proto[0] = 1.0
        fragments[2]['embedding'] = proto
        resolve(fragments, {'players': [{'player_id': 'P9', 'jersey': 9, 'team_id': 'AWAY'}]},
                {'P9': proto}, 0.88, 0.08)
        before = {f['fragment_id']: {k: copy.deepcopy(f.get(k)) for k in IDENTITY_KEYS + TRACKING_KEYS}
                  for f in fragments}
        embeddings = {f['fragment_id']: f['embedding'] for f in fragments}
        evidence = {f['fragment_id']: copy.deepcopy(f['appearance_evidence']) for f in fragments}
        classify(fragments, stats)
        for f in fragments:
            self.assertEqual({k: f.get(k) for k in IDENTITY_KEYS + TRACKING_KEYS},
                             before[f['fragment_id']])
            self.assertIs(f['embedding'], embeddings[f['fragment_id']])
            self.assertEqual(f['appearance_evidence'], evidence[f['fragment_id']])
        self.assertEqual(fragments[2]['player_id'], 'P9')


class InsufficientEvidenceTests(unittest.TestCase):
    """Test 5."""

    def test_below_classifier_minimum_gives_unknown_without_fitting(self):
        fragments, _, stats = two_kit_fragments(per_team=2)
        report = classify(fragments, stats)
        self.assertLess(report['fit']['fitting_crops'], TEAM_MIN_SAMPLES)
        self.assertFalse(report['fit']['fitted'])
        self.assertIsNone(report['fit']['cluster_fingerprint'])
        for f in fragments:
            self.assertEqual(f['team_id'], TEAM_UNKNOWN)
            self.assertEqual(f['team_classification']['evidence_status'],
                             TEAM_STATUS_INSUFFICIENT_CROPS)
            self.assertIsNone(f['team_classification']['cluster_fingerprint'])

    def test_minimum_is_the_classifier_default_and_is_not_lowered(self):
        self.assertEqual(TEAM_MIN_SAMPLES, 12)
        clf = build_team_classifier(candidate_mode='multi', candidate_action='flag')
        self.assertEqual(clf.min_samples, 12)
        self.assertEqual(clf.min_cluster_samples, 4)
        self.assertEqual(clf.random_state, TEAM_RANDOM_STATE)
        self.assertEqual(clf.random_state, 17)

    def test_fragment_without_crops_or_with_one_vote_stays_unknown(self):
        fragments, _, stats = two_kit_fragments(extra=[(RED, 0), (RED, 1)])
        classify(fragments, stats)
        no_crops, one_crop = fragments[-2], fragments[-1]
        self.assertEqual(no_crops['team_id'], TEAM_UNKNOWN)
        self.assertEqual(no_crops['team_classification']['evidence_status'], TEAM_STATUS_NO_CROPS)
        self.assertEqual(one_crop['team_id'], TEAM_UNKNOWN)
        self.assertEqual(one_crop['team_classification']['evidence_status'],
                         TEAM_STATUS_INSUFFICIENT_VOTES)


class RoleCandidateTests(unittest.TestCase):
    """Test 6."""

    def test_persistent_candidate_evidence_removes_the_label(self):
        preds = [prediction(TEAM_A, role=ROLE_REFEREE_CANDIDATE, score=0.9) for _ in range(4)]
        record = aggregate_fragment_team(preds, min_votes=2, min_majority=0.6)
        self.assertEqual(record['team_id'], TEAM_UNKNOWN)
        self.assertEqual(record['evidence_status'], TEAM_STATUS_ROLE_CANDIDATE)
        self.assertEqual(record['team_id_before_candidate_exclusion'], TEAM_A)

    def test_one_odd_crop_never_excludes_a_fragment(self):
        preds = [prediction(TEAM_B) for _ in range(3)]
        preds.append(prediction(TEAM_B, role=ROLE_REFEREE_CANDIDATE, score=0.95))
        record = aggregate_fragment_team(preds, min_votes=2, min_majority=0.6)
        self.assertEqual(record['team_id'], TEAM_B)
        self.assertEqual(record['evidence_status'], TEAM_STATUS_FRAGMENT_MAJORITY)
        self.assertFalse(record['candidate_evidence']['persistent_evidence'])

    def test_weak_candidate_scores_are_not_enough(self):
        preds = [prediction(TEAM_A, role=ROLE_REFEREE_CANDIDATE, score=0.55) for _ in range(4)]
        record = aggregate_fragment_team(preds, min_votes=2, min_majority=0.6)
        self.assertEqual(record['team_id'], TEAM_A)

    def test_yellow_kit_fragment_is_not_given_a_team(self):
        fragments, _, stats = two_kit_fragments(extra=[(YELLOW, 4)])
        report = classify(fragments, stats)
        self.assertTrue(report['fit']['fitted'], report['fit']['error'])
        self.assertGreater(report['fit']['fitting_set_candidate_gated'], 0)
        self.assertEqual(fragments[-1]['team_id'], TEAM_UNKNOWN)


class InvalidCropTests(unittest.TestCase):
    """Test 7."""

    def test_bad_boxes_are_rejected_and_counted_without_raising(self):
        stats = Stats()
        rng = np.random.default_rng(1)
        frame = kit_frame(RED, rng)
        f = make_fragment(0, 1, 0)
        bad = [
            ((float('nan'), 300.0, 640.0, 400.0), 0.8),
            ((640.0, 300.0, 600.0, 400.0), 0.8),     # inverted
            ((600.0, 300.0, 604.0, 310.0), 0.8),     # tiny
            ((2000.0, 900.0, 2040.0, 1000.0), 0.8),  # fully outside the frame
            (BOX, 0.1),                              # below min confidence
            ((0.0, 0.0, 1280.0, 720.0), 0.9),        # whole frame
        ]
        for box, conf in bad:
            meta = {'frame': 0, 'timestamp_s': 0.0,
                    'detection_id': build_detection_id(0, 1, 0), 'detector_confidence': conf}
            self.assertFalse(collect_team_crop(f, frame, box, meta, policy=stats.policy,
                                               box_stats=stats.box, crop_stats=stats.crop,
                                               max_crops=8))
        self.assertEqual(f.get('_team_crops', []), [])
        self.assertEqual(stats.box.total_evaluated, len(bad))
        self.assertEqual(stats.box.unique_rejected + stats.crop.unique_rejected, len(bad))
        self.assertEqual(stats.crop.unique_accepted, 0)

    def test_stored_crop_is_a_copy_not_a_frame_view(self):
        stats = Stats()
        frame = kit_frame(RED, np.random.default_rng(2))
        f = make_fragment(0, 1, 0)
        meta = {'frame': 0, 'timestamp_s': 0.0,
                'detection_id': build_detection_id(0, 1, 0), 'detector_confidence': 0.8}
        self.assertTrue(collect_team_crop(f, frame, BOX, meta, policy=stats.policy,
                                          box_stats=stats.box, crop_stats=stats.crop,
                                          max_crops=8))
        crop = f['_team_crops'][0]['crop']
        self.assertFalse(np.shares_memory(crop, frame))
        self.assertTrue(crop.flags['OWNDATA'])

    def test_per_fragment_crop_cap_is_respected(self):
        stats = Stats()
        f = make_fragment(0, 1, 0, length=200)
        accepted = collect(f, RED, 12, np.random.default_rng(4), stats)
        self.assertEqual(accepted, 8)
        self.assertEqual(len(f['_team_crops']), 8)

    def test_pipeline_survives_fragments_with_only_invalid_crops(self):
        fragments, _, stats = two_kit_fragments()
        broken = make_fragment(99, 999, 5000)
        collect(broken, RED, 4, np.random.default_rng(5), stats, confidence=0.05)
        broken.update(player_id=PLAYER_ID_UNRESOLVED, team_id=TEAM_UNKNOWN)
        fragments.append(broken)
        report = classify(fragments, stats)
        self.assertTrue(report['fit']['fitted'])
        self.assertEqual(broken['team_id'], TEAM_UNKNOWN)
        self.assertEqual(broken['team_classification']['evidence_status'], TEAM_STATUS_NO_CROPS)


class DeterminismTests(unittest.TestCase):
    """Tests 8 and 9."""

    def test_cluster_fingerprint_is_reproducible(self):
        prints = set()
        for _ in range(3):
            fragments, _, stats = two_kit_fragments(seed=11)
            prints.add(classify(fragments, stats)['fit']['cluster_fingerprint'])
        self.assertEqual(len(prints), 1)
        self.assertIsNotNone(prints.pop())

    def test_repeated_fits_give_identical_labels(self):
        runs = []
        for _ in range(3):
            fragments, _, stats = two_kit_fragments(seed=11)
            report = classify(fragments, stats)
            runs.append(([(f['fragment_id'], f['team_id'], f['team_classification']['score'])
                          for f in fragments], report['fragment_label_distribution']))
        self.assertEqual(runs[0], runs[1])
        self.assertEqual(runs[1], runs[2])

    def test_fragment_order_does_not_change_labels_or_fingerprint(self):
        forward, _, stats_f = two_kit_fragments(seed=11)
        backward, _, stats_b = two_kit_fragments(seed=11)
        backward.reverse()
        report_f = classify(forward, stats_f)
        report_b = classify(backward, stats_b)
        self.assertEqual(report_f['fit']['cluster_fingerprint'], report_b['fit']['cluster_fingerprint'])
        self.assertEqual({f['fragment_id']: f['team_id'] for f in forward},
                         {f['fragment_id']: f['team_id'] for f in backward})


class AmbiguityTests(unittest.TestCase):
    """Test 10."""

    def test_out_of_radius_crops_do_not_receive_a_team(self):
        fragments, _, stats = two_kit_fragments()
        clf = build_team_classifier(candidate_mode='off', candidate_action='flag')
        report = classify(fragments, stats, classifier=clf)
        self.assertTrue(report['fit']['fitted'])
        rng = np.random.default_rng(9)
        white = make_fragment(50, 500, 9000)
        collect(white, WHITE, 4, rng, Stats())
        preds = [clf.predict(entry['crop']) for entry in white['_team_crops']]
        self.assertTrue(all(p.team_id == TEAM_UNKNOWN for p in preds),
                        [(p.team_id, p.reason) for p in preds])
        self.assertTrue(all(p.reason == 'appearance_outlier' for p in preds))
        record = aggregate_fragment_team(preds, min_votes=2, min_majority=0.6)
        self.assertEqual(record['team_id'], TEAM_UNKNOWN)

    def test_split_votes_are_ambiguous(self):
        preds = [prediction(TEAM_A), prediction(TEAM_A), prediction(TEAM_B), prediction(TEAM_B)]
        record = aggregate_fragment_team(preds, min_votes=2, min_majority=0.6)
        self.assertEqual(record['team_id'], TEAM_UNKNOWN)
        self.assertEqual(record['evidence_status'], TEAM_STATUS_CONFLICTING_VOTES)
        self.assertEqual(record['majority_ratio'], 0.5)

    def test_abstaining_crops_do_not_count_as_votes(self):
        preds = [prediction(TEAM_A), prediction(TEAM_UNKNOWN, 0.0, 'appearance_outlier'),
                 prediction(TEAM_UNKNOWN, 0.0, 'appearance_outlier')]
        record = aggregate_fragment_team(preds, min_votes=2, min_majority=0.6)
        self.assertEqual(record['team_id'], TEAM_UNKNOWN)
        self.assertEqual(record['evidence_status'], TEAM_STATUS_INSUFFICIENT_VOTES)
        self.assertEqual(record['crop_abstention_reasons'], {'appearance_outlier': 2})

    def test_single_kit_population_is_not_forced_into_two_teams(self):
        fragments, _, stats = two_kit_fragments(per_team=0, extra=[(RED, 4)] * 16)
        report = classify(fragments, stats)
        labels = {f['team_id'] for f in fragments} - {TEAM_UNKNOWN}
        self.assertLessEqual(len(labels), 1, report['fit'])


class ReliabilityIntegrationTests(unittest.TestCase):
    def test_uncalibrated_real_pipeline_abstains_even_for_two_distinct_kits(self):
        fragments, _, stats = two_kit_fragments()
        report = classify(fragments, stats, calibration=None)
        self.assertTrue(report['fit']['fitted'])
        self.assertEqual(report['cluster_separation_check']['decision'], 'inconclusive')
        self.assertEqual({f['team_id'] for f in fragments}, {TEAM_UNKNOWN})
        self.assertTrue(all(f['team_classification']['score'] == 0.0 for f in fragments))
        self.assertEqual(set(report['uncalibrated_proposal_distribution']), {TEAM_A, TEAM_B})

    def test_mixed_fragment_is_rejected_even_with_a_majority(self):
        predictions = [prediction(TEAM_A)] * 3 + [prediction(TEAM_B)]
        record = reliable_fragment_team(predictions, {'decision': 'accepted'}, SYNTHETIC_CALIBRATION,
                                        min_votes=2, min_majority=.6)
        self.assertEqual(record['team_id'], TEAM_UNKNOWN)
        self.assertEqual(record['score'], 0.0)
        self.assertEqual(record['evidence_status'], 'mixed_fragment_appearance')
        self.assertEqual(record['proposal']['team_id'], TEAM_A)

    def test_weak_and_ambiguous_crop_predictions_do_not_vote(self):
        for predictions, reason in (([prediction(TEAM_A, .01)] * 3, 'weak_crop_score'),
                                    ([prediction(TEAM_B, margin=.01)] * 3, 'ambiguous_crop_distance')):
            with self.subTest(reason=reason):
                record = reliable_fragment_team(predictions, {'decision': 'accepted'}, SYNTHETIC_CALIBRATION,
                                                min_votes=2, min_majority=.6)
                self.assertEqual(record['team_id'], TEAM_UNKNOWN)
                self.assertEqual(record['score'], 0.0)
                self.assertEqual(record['crop_abstention_reasons'], {reason: 3})

    def test_role_evidence_survives_uncalibrated_separation(self):
        predictions = [prediction(TEAM_B, role=ROLE_REFEREE_CANDIDATE, score=.9)] * 4
        record = reliable_fragment_team(predictions,
                                        {'decision': 'inconclusive', 'reason': 'uncalibrated_team_separation'},
                                        None, min_votes=2, min_majority=.6)
        self.assertEqual(record['evidence_status'], TEAM_STATUS_ROLE_CANDIDATE)
        self.assertEqual(record['candidate_evidence']['candidate_crops'], 4)
        self.assertEqual(record['score'], 0.0)
        self.assertEqual(len(record['crop_predictions']), 4)

    def test_production_no_calibration_for_single_nearly_identical_and_third_kits(self):
        populations = ([(RED, 4)] * 16, [(RED, 4)] * 8 + [((41, 41, 201), 4)] * 8,
                       [(RED, 4)] * 8 + [(BLUE, 4)] * 8 + [((200, 20, 200), 4)] * 16)
        for population in populations:
            fragments, _, stats = two_kit_fragments(per_team=0, extra=population)
            classify(fragments, stats, calibration=None)
            self.assertTrue(all(f['team_id'] == TEAM_UNKNOWN for f in fragments))
            self.assertTrue(all(f['team_classification']['score'] == 0.0 for f in fragments))


class SchemaCompatibilityTests(unittest.TestCase):
    """Test 11."""

    def test_rows_pass_strict_validation_with_unk_players_on_both_teams(self):
        fragments, _, stats = two_kit_fragments()
        classify(fragments, stats)
        rows = rows_for(fragments)
        result = validate_rows(rows, strict=True)
        self.assertTrue(result['valid'], result['errors'])
        self.assertEqual(result['warning_count'], 0, result['warnings'])
        for team in (TEAM_A, TEAM_B):
            row = next(r for r in rows if r['team_id'] == team)
            self.assertEqual(row['player_id'], PLAYER_ID_UNRESOLVED)
            self.assertEqual(validate_detection_record(row, strict=True), ([], []))
        self.assertTrue(all('person_uid' not in r for r in rows))

    def test_unknown_rows_from_a_failed_fit_are_valid(self):
        fragments, _, stats = two_kit_fragments(per_team=1)
        classify(fragments, stats)
        result = validate_rows(rows_for(fragments), strict=True)
        self.assertTrue(result['valid'], result['errors'])
        self.assertEqual(result['team_id_values'], {TEAM_UNKNOWN: len(rows_for(fragments))})

    def test_fragments_serialize_after_crops_are_released(self):
        fragments, _, stats = two_kit_fragments()
        classify(fragments, stats, keep_crops=True)
        self.assertIn('_crops', fragments[0]['team_classification'])
        release_team_crops(fragments)
        payload = [{k: v for k, v in f.items() if k != 'embedding'} for f in fragments]
        json.dumps(payload)
        self.assertTrue(all('_team_crops' not in f for f in fragments))

    def test_saved_crop_index_is_readable_by_the_clustering_driver(self):
        from run_team_clustering import load_and_filter_samples
        fragments, _, stats = two_kit_fragments()
        classify(fragments, stats, keep_crops=True)
        with tempfile.TemporaryDirectory() as tmp:
            index = write_team_crop_index(Path(tmp), fragments, video='synthetic.mp4',
                                          frame_wh=(FRAME_W, FRAME_H), frames=600,
                                          rows_read=0, sampled=64, policy=stats.policy)
            self.assertEqual(index['crop_write_failures'], 0)
            _, retained = load_and_filter_samples(Path(tmp) / 'team_crop_index.json')
        release_team_crops(fragments)
        self.assertEqual(len(retained), len(index['samples']))
        self.assertEqual(len(retained), sum(f['team_classification']['crops_collected']
                                            for f in fragments))


class FailureIsolationTests(unittest.TestCase):
    """Test 12."""

    class RejectingClassifier:
        max_iterations = 100
        random_state = 17
        outlier_multiplier = 1.35
        minimum_radius = 0.12
        use_grass_mask = False
        min_foreground_ratio = 0.25
        candidate_mode = 'multi'
        candidate_action = 'flag'
        candidate_threshold = 0.5

        def fit(self, crops, crop_ids=None):
            raise ValueError('Each team cluster needs at least 4 samples; got [30, 2]')

        def predict(self, crop):
            raise AssertionError('predict must not run after a rejected fit')

    class BrokenClassifier(RejectingClassifier):
        def fit(self, crops, crop_ids=None):
            raise RuntimeError('unexpected classifier bug')

    def test_rejected_fit_yields_unknown_and_leaves_tracking_untouched(self):
        fragments, _, stats = two_kit_fragments()
        before = [{k: f.get(k) for k in IDENTITY_KEYS + TRACKING_KEYS} for f in fragments]
        report = classify(fragments, stats, classifier=self.RejectingClassifier())
        self.assertFalse(report['fit']['fitted'])
        self.assertIn('at least 4 samples', report['fit']['error'])
        self.assertEqual(before, [{k: f.get(k) for k in IDENTITY_KEYS + TRACKING_KEYS}
                                  for f in fragments])
        for f in fragments:
            self.assertEqual(f['team_id'], TEAM_UNKNOWN)
            self.assertEqual(f['team_classification']['evidence_status'],
                             TEAM_STATUS_CLUSTERING_REJECTED)
            self.assertNotIn('_team_crops', f)
        self.assertTrue(validate_rows(rows_for(fragments), strict=True)['valid'])

    def test_unexpected_classifier_errors_are_not_swallowed(self):
        fragments, _, stats = two_kit_fragments()
        with self.assertRaises(RuntimeError):
            classify(fragments, stats, classifier=self.BrokenClassifier())

    def test_accounting_records_do_not_depend_on_team_state(self):
        def record():
            return build_accounting_record(frame_idx=10, fps=15.0, scores=[0.9, 0.4, 0.05],
                                           emitted_track_ids=[1, 2], tracker_frame_id=11,
                                           previous_tracker_frame_id=10, active_tracks=2,
                                           lost_tracks=0, high_thresh=0.5, low_thresh=0.1)
        baseline = record()
        fragments, _, stats = two_kit_fragments()
        classify(fragments, stats, classifier=self.RejectingClassifier())
        self.assertEqual(record(), baseline)


if __name__ == '__main__':
    unittest.main()
