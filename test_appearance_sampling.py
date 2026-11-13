import unittest
from pathlib import Path
import numpy as np
import torch
from run import (
    EMBEDDING_FAILED, EMBEDDING_INSUFFICIENT, EMBEDDING_NO_VALID, EMBEDDING_OK,
    MIN_CROP_HEIGHT, MIN_CROP_WIDTH, clip_crop, finalize_embedding, is_sampling_turn,
    new_fragment, record_observation, resolve,
)

FRAME = np.full((200, 300, 3), 127, dtype=np.uint8)
GOOD_BOX = [10., 10., 50., 100.]


def mean_colour_encoder(crop):
    v = crop.reshape(-1, 3).mean(axis=0).astype(np.float32) + 1.
    return v / np.linalg.norm(v)


def simulate(fragment, start_frame, n_detections, sample_every, encoder=mean_colour_encoder, box=GOOD_BOX, max_observations=32):
    """Feed n consecutive detections of one fragment through the run.py sampling path."""
    for frame_idx in range(start_frame, start_frame + n_detections):
        index = fragment['_detections']; fragment['_detections'] += 1
        fragment['end_frame'] = frame_idx
        if is_sampling_turn(index, sample_every):
            record_observation(fragment, FRAME, box, encoder, max_observations)
    return fragment


class SamplingTests(unittest.TestCase):
    def test_first_detection_always_sampled(self):
        self.assertTrue(is_sampling_turn(0, 15))
        self.assertFalse(is_sampling_turn(1, 15))
        self.assertTrue(is_sampling_turn(15, 15))

    def test_sampling_independent_of_absolute_frame(self):
        # Frames 16..29 contain no multiple of 15: the old frame-based rule sampled nothing here.
        f = simulate(new_fragment('F1', 1, 16), 16, 14, 15)
        self.assertEqual(f['_sample_attempts'], 1)
        self.assertEqual(len(f['_vectors']), 1)

    def test_same_count_regardless_of_start_frame(self):
        counts = {len(simulate(new_fragment('F', 1, s), s, 46, 15)['_vectors']) for s in (0, 1, 7, 14, 16, 1000)}
        self.assertEqual(counts, {4})

    def test_stride(self):
        f = simulate(new_fragment('F1', 1, 0), 0, 10, 3)
        self.assertEqual(f['_sample_attempts'], 4)  # indices 0,3,6,9

    def test_sample_every_one_samples_all(self):
        f = simulate(new_fragment('F1', 1, 0), 0, 5, 1)
        self.assertEqual(len(f['_vectors']), 5)

    def test_cap_counts_valid_but_does_not_encode(self):
        calls = []
        enc = lambda c: (calls.append(1), mean_colour_encoder(c))[1]
        f = simulate(new_fragment('F1', 1, 0), 0, 10, 1, encoder=enc, max_observations=4)
        self.assertEqual(len(f['_vectors']), 4)
        self.assertEqual(len(calls), 4)
        self.assertEqual(f['_valid'], 10)


class CropTests(unittest.TestCase):
    def test_valid_crop(self):
        crop, reason = clip_crop(FRAME, GOOD_BOX)
        self.assertIsNone(reason)
        self.assertEqual(crop.shape[:2], (90, 40))

    def test_clipped_to_frame(self):
        crop, reason = clip_crop(FRAME, [-20., -20., 40., 90.])
        self.assertIsNone(reason)
        self.assertEqual(crop.shape[:2], (90, 40))

    def test_missing_frame(self):
        self.assertEqual(clip_crop(None, GOOD_BOX), (None, 'empty_frame'))
        self.assertEqual(clip_crop(np.zeros((0, 0, 3), np.uint8), GOOD_BOX)[1], 'empty_frame')

    def test_box_outside_frame(self):
        self.assertEqual(clip_crop(FRAME, [400., 400., 500., 600.])[1], 'empty_crop')

    def test_inverted_box(self):
        self.assertEqual(clip_crop(FRAME, [50., 100., 10., 10.])[1], 'empty_crop')

    def test_too_small(self):
        self.assertEqual(clip_crop(FRAME, [0., 0., MIN_CROP_WIDTH - 1, 100.])[1], 'crop_too_small')
        self.assertEqual(clip_crop(FRAME, [0., 0., 50., MIN_CROP_HEIGHT - 1])[1], 'crop_too_small')
        self.assertIsNone(clip_crop(FRAME, [0., 0., MIN_CROP_WIDTH, MIN_CROP_HEIGHT])[1])

    def test_invalid_bbox(self):
        self.assertEqual(clip_crop(FRAME, [np.nan, 0., 50., 100.])[1], 'invalid_bbox')
        self.assertEqual(clip_crop(FRAME, [0., 0., 50.])[1], 'invalid_bbox')
        self.assertEqual(clip_crop(FRAME, [0., 0., np.inf, 100.])[1], 'invalid_bbox')

    def test_invalid_crop_recorded_not_encoded(self):
        f = new_fragment('F1', 1, 0)
        enc = lambda c: self.fail('encoder must not be called')
        self.assertIsNone(record_observation(f, FRAME, [0., 0., 5., 5.], enc, 32))
        self.assertEqual(f['_invalid'], {'crop_too_small': 1})
        self.assertEqual((f['_valid'], f['_sample_attempts']), (0, 1))


class MinObservationTests(unittest.TestCase):
    def test_below_minimum_is_unknown(self):
        f = finalize_embedding(simulate(new_fragment('F1', 1, 0), 0, 31, 15), 4)  # 3 samples
        ev = f['appearance_evidence']
        self.assertIsNone(f['embedding'])
        self.assertEqual(ev['status'], EMBEDDING_INSUFFICIENT)
        self.assertEqual((ev['samples_used'], ev['valid_observations'], ev['min_observations']), (3, 3, 4))
        self.assertFalse(ev['embedding_generated'])
        self.assertIn('< min_observations 4', ev['reason'])

    def test_at_minimum_generates_embedding(self):
        f = finalize_embedding(simulate(new_fragment('F1', 1, 0), 0, 31, 15), 3)
        ev = f['appearance_evidence']
        self.assertEqual(ev['status'], EMBEDDING_OK)
        self.assertTrue(ev['embedding_generated'])
        self.assertAlmostEqual(float(np.linalg.norm(f['embedding'])), 1., places=5)
        self.assertEqual(ev['detections'], 31)

    def test_min_one_matches_single_sample(self):
        f = finalize_embedding(simulate(new_fragment('F1', 1, 0), 0, 1, 15), 1)
        self.assertEqual(f['appearance_evidence']['status'], EMBEDDING_OK)
        np.testing.assert_allclose(f['embedding'], mean_colour_encoder(FRAME[10:100, 10:50]), rtol=1e-6)

    def test_no_valid_observations(self):
        f = finalize_embedding(simulate(new_fragment('F1', 1, 0), 0, 20, 5, box=[0., 0., 5., 5.]), 1)
        ev = f['appearance_evidence']
        self.assertIsNone(f['embedding'])
        self.assertEqual(ev['status'], EMBEDDING_NO_VALID)
        self.assertEqual(ev['invalid_observations'], {'crop_too_small': 4})
        self.assertIn('crop_too_small=4', ev['reason'])

    def test_encoding_failure(self):
        bad = lambda c: np.full(4, np.nan, dtype=np.float32)
        f = finalize_embedding(simulate(new_fragment('F1', 1, 0), 0, 3, 1, encoder=bad), 1)
        ev = f['appearance_evidence']
        self.assertEqual(ev['status'], EMBEDDING_FAILED)
        self.assertEqual((ev['encoding_failures'], ev['valid_observations'], ev['samples_used']), (3, 3, 0))
        self.assertIsNone(f['embedding'])

    def test_zero_vector_is_failure(self):
        f = finalize_embedding(simulate(new_fragment('F1', 1, 0), 0, 2, 1, encoder=lambda c: np.zeros(4)), 1)
        self.assertEqual(f['appearance_evidence']['status'], EMBEDDING_FAILED)

    def test_working_fields_removed(self):
        f = finalize_embedding(simulate(new_fragment('F1', 1, 0), 0, 3, 1), 1)
        self.assertFalse([k for k in f if k.startswith('_')])

    def test_unknown_reason_for_missing_embedding(self):
        f = finalize_embedding(new_fragment('F1', 1, 0), 3)
        f.update(start_s=0, end_s=1)
        roster = {'players': [{'player_id': 'P1', 'team_id': 'HOME', 'jersey': 10}]}
        resolve([f], roster, {'P1': np.ones(4) / 2}, .88, .08)
        self.assertEqual((f['player_id'], f['team_id'], f['reason']), ('UNK', 'UNKNOWN', 'insufficient_appearance_evidence'))


class DeterminismTests(unittest.TestCase):
    def test_fake_encoder_deterministic(self):
        run = lambda: finalize_embedding(simulate(new_fragment('F1', 1, 0), 0, 60, 15), 3)['embedding']
        np.testing.assert_array_equal(run(), run())

    @unittest.skipUnless((Path(torch.hub.get_dir()) / 'checkpoints' / 'resnet18-f37072fd.pth').exists(), 'ResNet18 weights not cached')
    def test_resnet_encoder_deterministic(self):
        from run import Encoder
        enc = Encoder('cpu')
        rng = np.random.default_rng(0)
        frame = rng.integers(0, 255, (200, 300, 3), dtype=np.uint8)
        def run():
            f = new_fragment('F1', 1, 0)
            for i in range(4):
                f['_detections'] += 1
                record_observation(f, frame, [10. + i, 10., 60. + i, 120.], enc, 32)
            return finalize_embedding(f, 3)['embedding']
        a, b = run(), run()
        self.assertEqual(a.shape, (512,))
        np.testing.assert_array_equal(a, b)


if __name__ == '__main__':
    unittest.main()
