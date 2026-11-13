import unittest
from pathlib import Path
import tempfile
import subprocess
import sys

import cv2
import numpy as np


class ReviewToolSyntaxTests(unittest.TestCase):
    def test_script_compiles(self):
        source = Path(__file__).with_name("review_team_crops_v2.py")
        compile(source.read_text(encoding="utf-8"), str(source), "exec")

    def test_contact_sheet_dependency_available(self):
        self.assertTrue(hasattr(cv2, "imwrite"))
        self.assertTrue(np.zeros((2, 2, 3), dtype=np.uint8).size)

    def test_filter_detections_with_policy(self):
        from crop_filter import CropFilterPolicy
        from review_team_crops_v2 import filter_detections

        policy = CropFilterPolicy()
        detections = [
            # Valid detection
            {"frame": 0, "track_id": 1, "bbox_xyxy": [100, 100, 140, 200], "detector_confidence": 0.8},
            # Low confidence
            {"frame": 0, "track_id": 2, "bbox_xyxy": [100, 100, 140, 200], "detector_confidence": 0.2},
        ]
        by_track, stats = filter_detections(detections, policy, 1280, 720)
        self.assertIn(1, by_track)
        self.assertNotIn(2, by_track)
        self.assertEqual(stats.unique_accepted, 1)
        self.assertEqual(stats.unique_rejected, 1)


if __name__ == "__main__":
    unittest.main()
