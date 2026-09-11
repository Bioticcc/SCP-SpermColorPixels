"""Artifact and semantic checks for the bounded R2 fixture benchmark."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from experiments.r2_benchmark import _metric_summary, _selected_metrics, _truth_image, _PALETTE, run_r2_fixtures
from global_assignment import Hypothesis
from tests.fixtures.r2_competing import generate_competing_case


class R2BenchmarkTests(unittest.TestCase):
    def test_declared_paths_use_same_head_colors_as_prediction_panels(self):
        case = generate_competing_case()
        image = _truth_image(case)
        self.assertEqual(image[38, 20].tolist(), list(_PALETTE[1]))
        self.assertEqual(image[90, 20].tolist(), list(_PALETTE[0]))

    def _run(self):
        temporary = tempfile.TemporaryDirectory()
        root = Path(temporary.name) / "r2"
        report = run_r2_fixtures(root, {"search_max_expansions": 5000, "solver_max_expansions": 10000})
        return temporary, root, report

    def test_competing_case_replaces_independent_endpoint_conflict(self) -> None:
        temporary, root, report = self._run()
        self.addCleanup(temporary.cleanup)
        case = next(row for row in report["cases"] if row["name"] == "r2_competing_endpoint_width_3" and row["mode"] == "explicit")
        self.assertEqual(["r2_competing_endpoint_width_3:end_right"], case["independent"]["conflicts"]["endpoints"])
        self.assertEqual([], case["joint"]["conflicts"]["endpoints"])
        self.assertEqual([], case["joint"]["conflicts"]["ordinary_segments"])
        self.assertEqual("optimal", case["joint"]["result"]["status"])
        self.assertTrue(all(not value.endswith(":null") for value in case["joint"]["selected_by_head"].values()))
        by_id = {row["id"]: row for row in case["hypotheses"]}
        lower = by_id[case["joint"]["selected_by_head"]["h_lower"]]
        upper = by_id[case["joint"]["selected_by_head"]["h_upper"]]
        self.assertEqual(["lower_j", "j_top"], lower["metadata"]["r1_route"]["segment_ids"])
        self.assertEqual(["upper_j", "j_right"], upper["metadata"]["r1_route"]["segment_ids"])
        self.assertEqual({"h_upper": True, "h_lower": True}, case["declared_path_available"])
        self.assertTrue((root / "fixtures" / "r2_competing_endpoint_width_3" / "comparison.png").is_file())
        self.assertTrue((root / "fixtures" / "r2_competing_endpoint_width_3" / "joint_centerlines.png").is_file())

    def test_report_contains_constructed_semantics_and_refuses_overwrite(self) -> None:
        temporary, root, report = self._run()
        self.addCleanup(temporary.cleanup)
        self.assertEqual(22, len(report["cases"]))
        self.assertTrue((root / "fixture_report.json").is_file())
        saved = json.loads((root / "fixture_report.json").read_text())
        self.assertEqual(report["schema_version"], saved["schema_version"])
        indeterminate = next(row for row in report["cases"] if not row["expected_behavior"]["determinate"])
        self.assertIn("without a unique correctness claim", indeterminate["indeterminate_note"])
        self.assertEqual(2, len(indeterminate["indeterminate_identity_options"]))
        determined = next(row for row in report["cases"] if row["name"] == "r2_competing_endpoint_width_3_rendered")
        self.assertEqual({"full"}, {row["expected_status"] for row in determined["selected_centerline_metrics"]})
        self.assertTrue(any(
            row["mode"] == "rendered" and any(metric["expected_status"] == "partial" for metric in (row["selected_centerline_metrics"] or ()))
            for row in report["cases"]
        ))
        self.assertEqual({"r1_rank1", "independent", "joint"}, set(report["rendered_centerline_summary"]))
        rendered = next(row for row in report["cases"] if row["name"] == "r2_competing_endpoint_width_3_rendered")
        rendered_by_id = {row["id"]: row for row in rendered["hypotheses"]}
        self.assertEqual("partial", rendered_by_id[rendered["joint"]["selected_by_head"]["h_upper"]]["termination"])
        self.assertEqual(["segment_004", "segment_005", "segment_003"],
                         rendered_by_id[rendered["joint"]["selected_by_head"]["h_lower"]]["metadata"]["r1_route"]["segment_ids"])
        self.assertEqual(["segment_005"], rendered["unmodeled_extended_corridor_segments"])
        with self.assertRaises(FileExistsError):
            run_r2_fixtures(root, {})

    def test_null_prediction_counts_as_empty_all_instance_metric(self) -> None:
        case = generate_competing_case()
        nulls = {
            "upper-null": Hypothesis("upper-null", "h_upper", 0.0, termination="null"),
            "lower-null": Hypothesis("lower-null", "h_lower", 0.0, termination="null"),
        }
        rows = _selected_metrics(case, {"h_upper": "upper-null", "h_lower": "lower-null"}, nulls)
        self.assertTrue(all(not row["selected_route_available"] for row in rows))
        self.assertTrue(all(row["selected_null_or_missing"] for row in rows))
        self.assertEqual([0.0, 0.0], [row["centerline"]["f1"] for row in rows])
        summary = _metric_summary([{"mode": "rendered", "selection_centerline_metrics": {"joint": rows}}])
        full = summary["joint"]["full"]
        self.assertEqual(2, full["declared_instances"])
        self.assertEqual(0, full["selected_route_available"])
        self.assertEqual(2, full["number_null_or_missing_selections"])
        self.assertEqual(0.0, full["all_instance_mean_recall"])
        self.assertEqual(0.0, full["all_instance_mean_f1"])
        self.assertIsNone(full["conditional_on_route_mean_f1"])


if __name__ == "__main__":
    unittest.main()
