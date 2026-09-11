import tempfile
from pathlib import Path
import unittest

import cv2
import numpy as np

from experiments.benchmark import centerline_measurements
from experiments.run_baseline import run_baseline


class BenchmarkMeasurementsTests(unittest.TestCase):
    def test_measurements_detect_missing_tail_and_extra_branch(self):
        points = [[10, 30], [50, 30]]
        complete = np.zeros((64, 64), np.uint8)
        cv2.line(complete, (10, 30), (50, 30), 1, 1)
        exact = centerline_measurements(complete, points)
        self.assertEqual(exact['f1'], 1.0)
        short = complete.copy()
        short[:, 31:] = 0
        self.assertLess(centerline_measurements(short, points)['recall'], .65)
        branch = complete.copy()
        cv2.line(branch, (30, 30), (30, 60), 1, 1)
        extra = centerline_measurements(branch, points)
        self.assertEqual(extra['recall'], 1.0)
        self.assertLess(extra['precision'], .7)
        self.assertEqual(centerline_measurements(np.zeros_like(complete), points)['recall'], 0.0)

    def test_tolerance_is_spatial_not_polyline_vertex_count(self):
        prediction = np.zeros((64, 64), np.uint8)
        cv2.line(prediction, (10, 31), (50, 31), 1, 1)
        sparse = centerline_measurements(prediction, [[10, 30], [50, 30]])
        dense = centerline_measurements(prediction, [[10, 30], [20, 30], [30, 30], [50, 30]])
        self.assertEqual(sparse, dense)
        self.assertEqual(sparse['f1'], 1.0)

    def test_runner_refuses_canonical_or_source_output(self):
        project = Path(__file__).resolve().parents[1]
        config = project / 'configs/overlap_demo_baseline.json'
        for relative in ['Raw_Ward_Data/new', 'outputs/updated_outputs_path_v2/new', 'annotations/new']:
            with self.assertRaises(ValueError):
                run_baseline(config, project / 'Algorithmic_Pixel_Mask_For_Tails' / relative)


if __name__ == '__main__':
    unittest.main()
