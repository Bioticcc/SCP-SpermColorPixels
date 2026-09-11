"""Regression checks for bounded R1 search materialization."""
from __future__ import annotations

import unittest
from unittest.mock import patch

from graph_construction import from_explicit_graph
from path_hypotheses import search_hypotheses
from path_hypotheses import search as search_module
from tests.fixtures import generate_case


# Frozen from the R1 route-search artifacts before deferred materialization.
# Each row is (segment ids, directions, termination, score), including null.
EXPECTED = {
    "isolated_curve": ("h1", [([], [], "null", 0.0), (["e_h_b1", "e_b1_b2", "e_b2_end"], [True, True, True], "endpoint", 0.05134170759050163)]),
    "same_color_x": ("h_left", [([], [], "null", 0.0), (["left_j", "j_right"], [True, True], "endpoint", 0.0), (["left_j"], [True], "partial", 0.25), (["left_j", "j_bottom"], [True, True], "endpoint", 0.5), (["left_j", "top_j"], [True, False], "partial", 0.75)]),
    "curved_crossing": ("h_left", [([], [], "null", 0.0), (["left_j", "j_right"], [True, True], "endpoint", 0.024437251972198197), (["left_j"], [True], "partial", 0.25), (["left_j", "j_bottom"], [True, True], "endpoint", 0.6589469981442511), (["left_j", "top_j"], [True, False], "partial", 0.6944001122142162)]),
    "competing_heads": ("h_left", [([], [], "null", 0.0), (["left_j", "j_right"], [True, True], "endpoint", 0.06604810022015548), (["left_j"], [True], "partial", 0.25), (["left_j", "j_top"], [True, True], "endpoint", 0.5660481002201555), (["left_j", "bottom_j"], [True, False], "partial", 0.683951899779845)]),
    "indeterminate_identity": ("h_upper", [([], [], "null", 0.0), (["upper_merge"], [True], "partial", 0.25), (["upper_merge", "corridor_1", "corridor_2", "corridor_3", "split_lower"], [True, True, True, True, True], "endpoint", 0.477502323870715), (["upper_merge", "corridor_1", "corridor_2", "corridor_3", "split_upper"], [True, True, True, True, True], "endpoint", 0.4775023238707157), (["upper_merge", "corridor_1", "corridor_2", "corridor_3"], [True, True, True, True], "partial", 0.5127256113479921), (["upper_merge", "lower_merge"], [True, False], "partial", 0.7245487773040165)]),
    "self_loop": ("h1", [([], [], "null", 0.0), (["head_j"], [True], "partial", 0.25), (["head_j", "exit"], [True, True], "endpoint", 0.6542141144503336), (["head_j", "loop_return", "loop_across", "loop_up", "exit"], [True, False, False, False, True], "endpoint", 0.9063814917996825), (["head_j", "loop_return", "loop_across", "loop_up"], [True, False, False, False], "partial", 1.1045837596989152), (["head_j", "loop_up", "loop_across", "loop_return", "exit"], [True, True, True, True, True], "endpoint", 1.5979532628990154)]),
    "t_contact": ("h_left", [([], [], "null", 0.0), (["left_j", "j_right"], [True, True], "endpoint", 0.0), (["left_j"], [True], "partial", 0.25), (["left_j", "top_j"], [True, False], "partial", 0.75)]),
    "sharp_supported_bend": ("h1", [([], [], "null", 0.0), (["head_bend", "bend_end"], [True, True], "endpoint", 0.5413817659656779)]),
}


def _summary(result):
    return [(item["segment_ids"], item["directions"], item["termination"], item["score"])
            for item in result["hypotheses"]]


def _branching_graph(depth: int = 14):
    nodes = [{"id": "h", "xy": [0, 0], "kind": "head_anchor"}]
    edges = []
    previous = "h"
    for index in range(depth):
        current = "n%02d" % index
        nodes.append({"id": current, "xy": [index + 1, 0], "kind": "junction" if index else "ordinary"})
        for choice in ("a", "b"):
            edges.append({"id": "e%02d%s" % (index, choice), "start": previous, "end": current,
                          "points": [[index, 0], [index + 1, 0]]})
        previous = current
    nodes[-1]["kind"] = "endpoint"
    return from_explicit_graph({"nodes": nodes, "edges": edges})


class SearchPerformanceTests(unittest.TestCase):
    def test_fixed_r1_fixtures_match_frozen_routes_and_scores(self):
        for name, (head_id, expected) in EXPECTED.items():
            with self.subTest(name=name):
                graph = from_explicit_graph(generate_case(name).truth["graph"])
                actual = _summary(search_hypotheses(graph, head_id, k=5, max_expansions=20000))
                self.assertEqual(len(actual), len(expected))
                for received, wanted in zip(actual, expected):
                    self.assertEqual(received[:3], wanted[:3])
                    self.assertAlmostEqual(received[3], wanted[3], places=14)

    def test_large_branching_search_materializes_only_retained_routes(self):
        graph = _branching_graph()
        with patch.object(search_module, "_join", wraps=search_module._join) as joined:
            first = search_hypotheses(graph, "h", k=5, max_expansions=2000)
        second = search_hypotheses(_branching_graph(), "h", k=5, max_expansions=2000)
        self.assertEqual(_summary(first), _summary(second))
        self.assertEqual(joined.call_count, first["diagnostics"]["returned_nonnull_count"])
        self.assertEqual([row[0] for row in _summary(first)[1:]], [
            ["e00a", "e01a"],
            ["e00a", "e01a", "e02a"],
            ["e00a", "e01a", "e02a", "e03a"],
            ["e00a", "e01a", "e02a", "e03a", "e04a"],
            ["e00a", "e01a", "e02a", "e03a", "e04a", "e05a"],
        ])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
