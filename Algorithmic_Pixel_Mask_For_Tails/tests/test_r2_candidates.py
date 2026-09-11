"""Focused contracts for R2 candidate-pool construction."""
from __future__ import annotations

import math
import unittest

from experiments.r2_candidates import build_candidate_pool, build_graph_pool
from graph_construction import from_explicit_graph


def _graph(*, junction: bool = True):
    return from_explicit_graph({"nodes": [
        {"id": "h", "xy": [0, 0], "kind": "head_anchor"},
        {"id": "j", "xy": [10, 0], "kind": "junction" if junction else "ordinary"},
        {"id": "e", "xy": [20, 0], "kind": "endpoint"},
    ], "edges": [
        {"id": "s1", "start": "h", "end": "j", "points": [[0, 0], [10, 0]]},
        {"id": "s2", "start": "j", "end": "e", "points": [[10, 0], [20, 0]]},
    ]})


def _route(route_id, segments, nodes, points, termination="endpoint", direction=0.0, curvature=0.0, evidence=0.0):
    return {"id": route_id, "head_id": "h", "segment_ids": segments, "directions": [True] * len(segments),
            "node_ids": nodes, "points_xy": points, "termination": termination,
            "costs": {"direction": direction, "curvature": curvature, "evidence": evidence}, "score": 99.0}


class R2CandidateTests(unittest.TestCase):
    def test_same_head_across_components_has_one_null_and_scoped_resources(self):
        graph = _graph()
        route = _route("h:h1", ["s1", "s2"], ["h", "j", "e"], [[0, 0], [10, 0], [20, 0]])
        longer_route = _route("h:h2", ["s1", "s2"], ["h", "j", "e"], [[0, 0], [20, 0], [40, 0]])
        record = {"components": [
            {"tail_id": 4, "roi_xyxy": [5, 6, 30, 30], "graph": graph.to_dict(), "searches": {"h": {"hypotheses": [route]}}, "baseline": []},
            {"tail_id": 8, "roi_xyxy": [40, 6, 70, 30], "graph": graph.to_dict(), "searches": {"h": {"hypotheses": [longer_route]}}, "baseline": []},
        ]}
        pool = build_candidate_pool(record, {})
        nulls = [item for item in pool["hypotheses"] if item.is_null]
        self.assertEqual([(item.id, item.head_id, item.cost) for item in nulls], [("r2:null:h", "h", 1.5)])
        paths = [item for item in pool["hypotheses"] if item.metadata.get("kind") == "r1_route"]
        self.assertEqual({item.id for item in paths}, {"r2:4:h:h:h1", "r2:8:h:h:h2"})
        self.assertEqual({item.exclusive_segments[0].split(":")[0] for item in paths}, {"4", "8"})
        self.assertEqual({item.endpoint_id for item in paths}, {"4:e", "8:e"})
        by_tail = {item.metadata["tail_id"]: item.metadata["cost_decomposition"] for item in paths}
        self.assertAlmostEqual(by_tail["4"]["coverage_deficit"], .5)
        self.assertEqual(by_tail["8"]["coverage_deficit"], 0.0)

    def test_partial_and_null_costs_expose_every_requested_term(self):
        graph = _graph()
        partial = _route("partial", ["s1", "s2"], ["h", "j"], [[0, 0], [10, 0], [20, 0]], "partial", .6, .3, .2)
        longer = _route("long", ["s1", "s2"], ["h", "j", "e"], [[0, 0], [20, 0], [40, 0]], "endpoint")
        items = build_graph_pool(graph, {"h": {"hypotheses": [partial, longer]}}, {})
        result = next(item for item in items if item.id.endswith(":partial"))
        detail = result.metadata["cost_decomposition"]
        self.assertAlmostEqual(detail["mean_transition_cost"], .9)
        self.assertAlmostEqual(detail["coverage_deficit"], .5)
        self.assertAlmostEqual(detail["termination_cost"], .85)
        self.assertAlmostEqual(result.cost, .9 + .2 + .5 + .85)
        null = next(item for item in items if item.is_null)
        self.assertEqual(null.cost, 1.5)
        self.assertEqual(null.metadata["cost_decomposition"], {"null_cost": 1.5})

    def test_isolated_control_retains_exact_baseline_raster_only(self):
        graph = _graph(junction=False)
        route = _route("unused", ["s1", "s2"], ["h", "j", "e"], [[0, 0], [10, 0], [20, 0]])
        pixels = [[2, 3], [3, 3], [4, 3]]
        pool = build_candidate_pool({"components": [{
            "tail_id": 2, "roi_xyxy": [100, 200, 120, 220], "graph": graph.to_dict(),
            "searches": {"h": {"hypotheses": [route]}},
            "baseline": [{"crop_id": "crop-7", "head_id": "h", "pixels_xy": pixels, "status": "accepted"}],
        }]}, {})
        self.assertEqual(len(pool["hypotheses"]), 2)
        control = next(item for item in pool["hypotheses"] if item.metadata.get("kind") == "baseline_control")
        self.assertEqual(control.cost, 0.0)
        self.assertEqual(control.metadata["pixels_xy_unordered"], pixels)
        self.assertEqual(control.metadata["source_roi_offset_xy"], [100, 200])
        self.assertEqual(control.exclusive_segments, ("2:s1", "2:s2"))
        self.assertEqual(pool["pool_diagnostics"]["isolated_controls"][0]["r1_alternatives_retained_for_inspection"], [route])

    def test_long_collinear_route_uses_mean_transition_cost_not_segment_count(self):
        graph = from_explicit_graph({"nodes": [
            {"id": "h", "xy": [0, 0], "kind": "head_anchor"}, {"id": "a", "xy": [10, 0]},
            {"id": "b", "xy": [20, 0]}, {"id": "e", "xy": [30, 0], "kind": "endpoint"},
        ], "edges": [
            {"id": "a", "start": "h", "end": "a", "points": [[0, 0], [10, 0]]},
            {"id": "b", "start": "a", "end": "b", "points": [[10, 0], [20, 0]]},
            {"id": "c", "start": "b", "end": "e", "points": [[20, 0], [30, 0]]},
        ]})
        route = _route("long", ["a", "b", "c"], ["h", "a", "b", "e"], [[0, 0], [10, 0], [20, 0], [30, 0]], direction=.6, curvature=.3)
        item = next(item for item in build_graph_pool(graph, {"h": {"hypotheses": [route]}}, {}) if not item.is_null)
        self.assertAlmostEqual(item.cost, .45)
        self.assertAlmostEqual(item.metadata["r1_raw_score"], 99.0)

    def test_invalid_cost_preferences_are_rejected(self):
        for key, value in (("null_cost", -1), ("partial_cost", math.nan), ("coverage_weight", True)):
            with self.subTest(key=key):
                with self.assertRaises(ValueError):
                    build_graph_pool(_graph(), {"h": {"hypotheses": []}}, {key: value})


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
