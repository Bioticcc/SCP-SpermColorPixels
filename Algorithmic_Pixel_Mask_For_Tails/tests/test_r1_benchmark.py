"""Contract tests for the R1 fixture benchmark."""
from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

import cv2
import numpy as np

from experiments.benchmark import centerline_measurements
from experiments.r1_benchmark import run_r1_fixtures


class R1BenchmarkTests(unittest.TestCase):
    def test_centerline_measurement_rejects_missing_route(self):
        truth = [[5, 20], [55, 20]]
        missing = np.zeros((64, 64), np.uint8)
        self.assertEqual(centerline_measurements(missing, truth)["recall"], 0.0)
        partial = missing.copy()
        cv2.line(partial, (5, 20), (25, 20), 1, 1)
        self.assertLess(centerline_measurements(partial, truth)["recall"], 0.5)

    def test_fixture_report_preserves_partial_and_indeterminate_semantics(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report = run_r1_fixtures(root, {"fixture_seed": 0, "max_expansions": 20000})
            self.assertTrue((root / "fixture_report.json").is_file())
            self.assertEqual(len(report["cases"]), 8)
            t_contact = next(row for row in report["cases"] if row["name"] == "t_contact")
            partial = next(row for row in t_contact["explicit_graph"] if row["instance_id"] == "contact_fragment")
            self.assertEqual(partial["expected_status"], "partial")
            self.assertTrue(partial["k5_contains_declared_route"])
            self.assertEqual(partial["k5_termination"], "partial")
            indeterminate = next(row for row in report["cases"] if row["name"] == "indeterminate_identity")
            self.assertFalse(indeterminate["expected_behavior"]["determinate"])
            self.assertTrue(all(solution["all_routes_available"]
                                for solution in indeterminate["indeterminate_solution_availability"]))
            self.assertTrue((root / "fixtures" / "indeterminate_identity" / "comparison.png").is_file())

    def test_report_contains_nonempty_search_and_transform_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report = run_r1_fixtures(root, {})
            isolated = next(row for row in report["cases"] if row["name"] == "isolated_curve")
            row = isolated["explicit_graph"][0]
            self.assertTrue(row["route_available"])
            self.assertTrue(row["k5_contains_declared_route"])
            self.assertEqual(set(isolated["transformations"]),
                             {"horizontal_flip", "vertical_flip", "rotate_180", "padded_translation"})
            self.assertTrue((root / "fixtures" / "isolated_curve" / "graph.json").is_file())
            self.assertTrue((root / "fixtures" / "isolated_curve" / "search.json").is_file())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
