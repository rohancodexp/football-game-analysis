"""
Unit tests for the unified crop filtering module (crop_filter.py).

Covers deterministic boundaries for:
- Detector confidence thresholds.
- Minimum and maximum box height.
- Minimum and maximum box width.
- Bounding-box area fraction.
- Aspect ratio boundaries.
- Invalid bounding boxes and non-finite numbers.
- Missing or invalid metadata.
- Multiple simultaneous rejection reasons and stats accounting.
- Torso crop dimension evaluation.
- Serialization (to_dict / from_dict).
"""

import math
import unittest
import numpy as np

from crop_filter import (
    CropFilterPolicy,
    FilterResult,
    FilterStats,
    REASON_AREA_TOO_LARGE,
    REASON_ASPECT_RATIO_TOO_HIGH,
    REASON_BOX_TOO_NARROW,
    REASON_BOX_TOO_SHORT,
    REASON_BOX_TOO_TALL,
    REASON_CROP_INVALID,
    REASON_CROP_TOO_NARROW,
    REASON_CROP_TOO_SHORT,
    REASON_INVALID_BOX,
    REASON_INVALID_METADATA,
    REASON_LOW_CONFIDENCE,
)


class TestCropFilterPolicy(unittest.TestCase):
    def setUp(self):
        # 1280x720 video frame (standard InFactory test clips)
        self.frame_w = 1280
        self.frame_h = 720
        self.policy = CropFilterPolicy(
            min_confidence=0.45,
            min_box_height=50,
            max_box_height=160,
            min_box_width=14,
            max_box_area_fraction=0.025,
            max_aspect_ratio=0.90,
            min_crop_height=14,
            min_crop_width=8,
        )

    # 1. Valid box baseline
    def test_valid_box_passes(self):
        # Box: x=100, y=100, w=30, h=80 (area = 2400, area_frac = 2400/921600 = 0.0026, aspect_ratio = 30/80 = 0.375)
        res = self.policy.evaluate_box([100, 100, 130, 180], 0.75, self.frame_w, self.frame_h)
        self.assertTrue(res.is_valid)
        self.assertEqual(res.reasons, [])
        self.assertEqual(res.metrics["box_width"], 30.0)
        self.assertEqual(res.metrics["box_height"], 80.0)

    # 2. Confidence boundaries
    def test_detector_confidence_boundary(self):
        # Exactly at boundary 0.45 -> valid
        res_at = self.policy.evaluate_box([100, 100, 130, 180], 0.45, self.frame_w, self.frame_h)
        self.assertTrue(res_at.is_valid)

        # Just below boundary 0.449 -> rejected
        res_below = self.policy.evaluate_box([100, 100, 130, 180], 0.449, self.frame_w, self.frame_h)
        self.assertFalse(res_below.is_valid)
        self.assertIn(REASON_LOW_CONFIDENCE, res_below.reasons)

        # High confidence 0.99 -> valid
        res_high = self.policy.evaluate_box([100, 100, 130, 180], 0.99, self.frame_w, self.frame_h)
        self.assertTrue(res_high.is_valid)

    # 3. Minimum and maximum box height
    def test_box_height_boundaries(self):
        # Exactly min_box_height = 50 -> valid
        res_min = self.policy.evaluate_box([100, 100, 120, 150], 0.60, self.frame_w, self.frame_h)
        self.assertTrue(res_min.is_valid)

        # Height = 49 (below 50) -> rejected
        res_short = self.policy.evaluate_box([100, 100, 120, 149], 0.60, self.frame_w, self.frame_h)
        self.assertFalse(res_short.is_valid)
        self.assertIn(REASON_BOX_TOO_SHORT, res_short.reasons)

        # Exactly max_box_height = 160 -> valid (with narrow width to stay under area_fraction)
        # width = 30, height = 160 -> area = 4800 (area fraction 0.0052, aspect ratio 0.1875)
        res_max = self.policy.evaluate_box([100, 100, 130, 260], 0.60, self.frame_w, self.frame_h)
        self.assertTrue(res_max.is_valid)

        # Height = 161 (above 160) -> rejected
        res_tall = self.policy.evaluate_box([100, 100, 130, 261], 0.60, self.frame_w, self.frame_h)
        self.assertFalse(res_tall.is_valid)
        self.assertIn(REASON_BOX_TOO_TALL, res_tall.reasons)

    # 4. Minimum box width
    def test_box_width_boundaries(self):
        # Exactly min_box_width = 14 (with height = 60, aspect ratio = 14/60 = 0.233)
        res_min = self.policy.evaluate_box([100, 100, 114, 160], 0.60, self.frame_w, self.frame_h)
        self.assertTrue(res_min.is_valid)

        # Width = 13 (below 14) -> rejected
        res_narrow = self.policy.evaluate_box([100, 100, 113, 160], 0.60, self.frame_w, self.frame_h)
        self.assertFalse(res_narrow.is_valid)
        self.assertIn(REASON_BOX_TOO_NARROW, res_narrow.reasons)

    # 5. Area fraction
    def test_area_fraction_boundary(self):
        # frame_area = 1280 * 720 = 921,600 px. max_fraction = 0.025 -> max_area = 23,040 px
        # Try box: width = 140, height = 160 -> area = 22,400 (< 23,040, fraction = 0.0243, aspect = 140/160 = 0.875)
        res_valid = self.policy.evaluate_box([100, 100, 240, 260], 0.60, self.frame_w, self.frame_h)
        self.assertTrue(res_valid.is_valid)

        # Try box exceeding max area fraction:
        # width = 144, height = 160 -> area = 23,040.0 (fraction = 0.025)
        # width = 146, height = 160 -> area = 23,360 (> 23,040, fraction = 0.02535)
        res_large = self.policy.evaluate_box([100, 100, 246, 260], 0.60, self.frame_w, self.frame_h)
        self.assertFalse(res_large.is_valid)
        self.assertIn(REASON_AREA_TOO_LARGE, res_large.reasons)

    # 6. Aspect ratio
    def test_aspect_ratio_boundary(self):
        # Height = 100. max_aspect_ratio = 0.90 -> max_width = 90
        # Exactly width = 90 -> aspect_ratio = 0.90 -> valid
        res_valid = self.policy.evaluate_box([100, 100, 190, 200], 0.60, self.frame_w, self.frame_h)
        self.assertTrue(res_valid.is_valid)

        # Width = 91 -> aspect_ratio = 0.91 (> 0.90) -> rejected
        res_wide = self.policy.evaluate_box([100, 100, 191, 200], 0.60, self.frame_w, self.frame_h)
        self.assertFalse(res_wide.is_valid)
        self.assertIn(REASON_ASPECT_RATIO_TOO_HIGH, res_wide.reasons)

    # 7. Invalid bounding boxes
    def test_invalid_box_coordinates(self):
        # Negative width (x2 < x1)
        res1 = self.policy.evaluate_box([150, 100, 120, 200], 0.60, self.frame_w, self.frame_h)
        self.assertFalse(res1.is_valid)
        self.assertIn(REASON_INVALID_BOX, res1.reasons)

        # Zero height (y2 == y1)
        res2 = self.policy.evaluate_box([100, 100, 140, 100], 0.60, self.frame_w, self.frame_h)
        self.assertFalse(res2.is_valid)
        self.assertIn(REASON_INVALID_BOX, res2.reasons)

        # Non-finite coordinate (NaN, inf)
        res3 = self.policy.evaluate_box([100, float("nan"), 140, 200], 0.60, self.frame_w, self.frame_h)
        self.assertFalse(res3.is_valid)
        self.assertIn(REASON_INVALID_BOX, res3.reasons)

        # Malformed list length
        res4 = self.policy.evaluate_box([100, 100, 140], 0.60, self.frame_w, self.frame_h)
        self.assertFalse(res4.is_valid)
        self.assertIn(REASON_INVALID_BOX, res4.reasons)

    # 8. Missing or invalid metadata
    def test_invalid_metadata(self):
        # Invalid frame dimensions
        res1 = self.policy.evaluate_box([100, 100, 130, 180], 0.60, 0, 720)
        self.assertFalse(res1.is_valid)
        self.assertIn(REASON_INVALID_METADATA, res1.reasons)

        # Non-finite confidence
        res2 = self.policy.evaluate_box([100, 100, 130, 180], float("inf"), self.frame_w, self.frame_h)
        self.assertFalse(res2.is_valid)
        self.assertIn(REASON_INVALID_METADATA, res2.reasons)

        # Sample record missing required key
        res3 = self.policy.evaluate_sample_record({"box_width": 30.0}, self.frame_w, self.frame_h)
        self.assertFalse(res3.is_valid)
        self.assertIn(REASON_INVALID_METADATA, res3.reasons)

    # 9. Multiple simultaneous rejection reasons
    def test_multiple_simultaneous_rejection_reasons(self):
        # Box: width = 10 (< 14: box_too_narrow), height = 40 (< 50: box_too_short), conf = 0.20 (< 0.45: low_confidence)
        res = self.policy.evaluate_box([100, 100, 110, 140], 0.20, self.frame_w, self.frame_h)
        self.assertFalse(res.is_valid)
        self.assertIn(REASON_LOW_CONFIDENCE, res.reasons)
        self.assertIn(REASON_BOX_TOO_NARROW, res.reasons)
        self.assertIn(REASON_BOX_TOO_SHORT, res.reasons)
        self.assertEqual(len(res.reasons), 3)

    # 10. Torso crop dimension evaluation
    def test_torso_crop_evaluation(self):
        # Valid crop: 20x15x3
        valid_crop = np.zeros((20, 15, 3), dtype=np.uint8)
        res_valid = self.policy.evaluate_torso_crop(valid_crop)
        self.assertTrue(res_valid.is_valid)

        # Too short: 12x15x3 (height 12 < 14)
        short_crop = np.zeros((12, 15, 3), dtype=np.uint8)
        res_short = self.policy.evaluate_torso_crop(short_crop)
        self.assertFalse(res_short.is_valid)
        self.assertIn(REASON_CROP_TOO_SHORT, res_short.reasons)

        # Too narrow: 20x6x3 (width 6 < 8)
        narrow_crop = np.zeros((20, 6, 3), dtype=np.uint8)
        res_narrow = self.policy.evaluate_torso_crop(narrow_crop)
        self.assertFalse(res_narrow.is_valid)
        self.assertIn(REASON_CROP_TOO_NARROW, res_narrow.reasons)

        # None or 0-size crop
        res_none = self.policy.evaluate_torso_crop(None)
        self.assertFalse(res_none.is_valid)
        self.assertIn(REASON_CROP_INVALID, res_none.reasons)

    # 11. Accounting: FilterStats unique vs overlapping counts
    def test_filter_stats_accounting(self):
        stats = FilterStats()

        # 1 valid
        stats.record(FilterResult(is_valid=True))
        # 1 rejected with 1 reason
        stats.record(FilterResult(is_valid=False, reasons=[REASON_LOW_CONFIDENCE]))
        # 1 rejected with 2 simultaneous reasons
        stats.record(FilterResult(is_valid=False, reasons=[REASON_LOW_CONFIDENCE, REASON_BOX_TOO_SHORT]))

        d = stats.to_dict()
        self.assertEqual(d["total_evaluated"], 3)
        self.assertEqual(d["unique_accepted"], 1)
        self.assertEqual(d["unique_rejected"], 2)

        # Overlapping counts: low_confidence = 2, box_too_short = 1
        self.assertEqual(d["rejection_reasons_overlapping"][REASON_LOW_CONFIDENCE], 2)
        self.assertEqual(d["rejection_reasons_overlapping"][REASON_BOX_TOO_SHORT], 1)

        # Primary rejection counts sum exactly to unique_rejected:
        self.assertEqual(sum(d["primary_rejection_reasons"].values()), 2)

        # Combination tracking
        self.assertEqual(d["rejection_combinations"]["box_too_short+low_detector_confidence"], 1)

    # 12. Serialization & Roundtrip
    def test_policy_serialization(self):
        p_dict = self.policy.to_dict()
        reconstructed = CropFilterPolicy.from_dict(p_dict)
        self.assertEqual(self.policy, reconstructed)


if __name__ == "__main__":
    unittest.main()

