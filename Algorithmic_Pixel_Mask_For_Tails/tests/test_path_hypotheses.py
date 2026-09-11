"""R1 route-search checks against explicit geometry truth."""
from __future__ import annotations

import unittest

from graph_construction import from_explicit_graph
from junction_transitions import transition_cost
from path_hypotheses import search_hypotheses
from path_hypotheses.search import _join
from tests.fixtures import generate_case


def _result(name: str, head: str):
    fixture = generate_case(name)
    graph = from_explicit_graph(fixture.truth["graph"])
    return fixture, search_hypotheses(graph, head, k=5)


class PathHypothesisTests(unittest.TestCase):
    def test_isolated_and_supported_bend_retain_declared_route(self):
        for name, head, expected in [
            ("isolated_curve", "h1", ["e_h_b1", "e_b1_b2", "e_b2_end"]),
            ("sharp_supported_bend", "h1", ["head_bend", "bend_end"]),
        ]:
            _, result = _result(name, head)
            routes = [h["segment_ids"] for h in result["hypotheses"]]
            self.assertIn(expected, routes)

    def test_x_and_indeterminate_keep_multiple_endings(self):
        _, x = _result("same_color_x", "h_left")
        endpoints = [h for h in x["hypotheses"] if h["termination"] == "endpoint"]
        self.assertGreaterEqual(len(endpoints), 2)
        _, ambiguous = _result("indeterminate_identity", "h_upper")
        endings = {tuple(h["segment_ids"][-1:]) for h in ambiguous["hypotheses"] if h["termination"] == "endpoint"}
        self.assertGreaterEqual(len(endings), 2)

    def test_self_loop_revisits_junction_without_repeating_edges(self):
        _, result = _result("self_loop", "h1")
        routes = [h for h in result["hypotheses"] if h["termination"] != "null"]
        self.assertTrue(any(r["node_ids"].count("j") >= 2 for r in routes))
        for route in routes:
            self.assertEqual(len(route["segment_ids"]), len(set(route["segment_ids"])))

    def test_expansion_bound_is_reported(self):
        fixture = generate_case("self_loop")
        graph = from_explicit_graph(fixture.truth["graph"])
        result = search_hypotheses(graph, "h1", k=5, max_expansions=2)
        self.assertLessEqual(result["diagnostics"]["expansions"], 20000)
        self.assertIn("truncated", result["diagnostics"])
        self.assertEqual(result["diagnostics"]["ranking_scope"], "top_k_among_explored_candidates")

    def test_all_declared_fixture_routes_survive_when_determinate(self):
        for name in ("isolated_curve", "same_color_x", "curved_crossing", "competing_heads", "self_loop", "t_contact", "sharp_supported_bend"):
            fixture = generate_case(name)
            graph = from_explicit_graph(fixture.truth["graph"])
            for truth_path in fixture.truth["true_edge_paths"]:
                result = search_hypotheses(graph, truth_path["node_ids"][0], k=5)
                actual = {tuple(h["segment_ids"]) for h in result["hypotheses"]}
                self.assertIn(tuple(truth_path["edge_ids"]), actual, msg=f"{name}: {truth_path['instance_id']}")

    def test_indeterminate_preserves_both_declared_endings(self):
        fixture = generate_case("indeterminate_identity")
        graph = from_explicit_graph(fixture.truth["graph"])
        for head in ("h_upper", "h_lower"):
            result = search_hypotheses(graph, head, k=5)
            endings = {h["segment_ids"][-1] for h in result["hypotheses"] if h["termination"] == "endpoint"}
            self.assertEqual({"split_upper", "split_lower"}, endings)

    def test_straight_x_pairing_is_cheaper_than_turn(self):
        fixture = generate_case("same_color_x")
        graph = from_explicit_graph(fixture.truth["graph"])
        straight = transition_cost(graph, "j", ("left_j", True), ("j_right", True))["total"]
        turn = transition_cost(graph, "j", ("left_j", True), ("j_bottom", True))["total"]
        self.assertLess(straight, turn)

    def test_resampling_and_local_radius_keep_straight_cost_low(self):
        fixture = generate_case("same_color_x")
        raw = fixture.truth["graph"]
        for edge in raw["edges"]:
            if edge["id"] in {"left_j", "j_right"}:
                edge["radii"] = [2.0] * len(edge["points"])
        graph = from_explicit_graph(raw)
        cost = transition_cost(graph, "j", ("left_j", True), ("j_right", True))
        self.assertLess(cost["direction"], 0.02)
        self.assertLess(cost["curvature"], 0.02)

    def test_join_refuses_unsupported_chord(self):
        graph = from_explicit_graph({"nodes": [
            {"id": "a", "xy": [0, 0], "kind": "head_anchor"},
            {"id": "j", "xy": [1, 0], "kind": "junction", "pixels_xy": [[1, 0]]},
            {"id": "b", "xy": [4, 0], "kind": "endpoint"},
        ], "edges": [
            {"id": "in", "start": "a", "end": "j", "points": [[0, 0], [0, 0]]},
            {"id": "out", "start": "j", "end": "b", "points": [[4, 0], [4, 0]]},
        ]})
        self.assertIsNone(_join(graph, [("in", True), ("out", True)]))

    def test_k_one_prefers_full_route_over_junction_partial(self):
        _, result = _result("same_color_x", "h_left")
        fixture = generate_case("same_color_x")
        one = search_hypotheses(from_explicit_graph(fixture.truth["graph"]), "h_left", k=1)
        self.assertEqual(len(one["hypotheses"]), 2)
        self.assertEqual(one["hypotheses"][1]["termination"], "endpoint")

    def test_foreign_anchor_and_boundary_have_explicit_terminations(self):
        graph = from_explicit_graph({"nodes": [
            {"id": "start", "xy": [0, 0], "kind": "head_anchor"},
            {"id": "foreign", "xy": [1, 0], "kind": "endpoint"},
            {"id": "border", "xy": [0, 2], "kind": "boundary_port"},
        ], "edges": [
            {"id": "to_foreign", "start": "start", "end": "foreign", "points": [[0, 0], [1, 0]]},
            {"id": "to_border", "start": "start", "end": "border", "points": [[0, 0], [0, 2]]},
        ]}, anchors={"a": "start", "b": "foreign"})
        result = search_hypotheses(graph, "a", k=5)
        by_edge = {h["segment_ids"][0]: h["termination"] for h in result["hypotheses"] if h["segment_ids"]}
        self.assertEqual(by_edge["to_foreign"], "partial")
        self.assertEqual(by_edge["to_border"], "boundary")

    def test_out_of_range_evidence_is_rejected_before_search(self):
        graph = from_explicit_graph({"nodes": [
            {"id": "h", "xy": [0, 0], "kind": "head_anchor"}, {"id": "e", "xy": [1, 0], "kind": "endpoint"},
        ], "edges": [{"id": "bad", "start": "h", "end": "e", "points": [[0, 0], [1, 0]], "evidence": 1.1}]})
        with self.assertRaisesRegex(ValueError, "evidence"):
            search_hypotheses(graph, "h")

    def test_subdividing_equal_evidence_does_not_change_route_evidence_cost(self):
        base = {"nodes": [{"id": "h", "xy": [0, 0], "kind": "head_anchor"}, {"id": "e", "xy": [4, 0], "kind": "endpoint"}],
                "edges": [{"id": "whole", "start": "h", "end": "e", "points": [[0, 0], [4, 0]], "evidence": 0.5}]}
        split = {"nodes": [{"id": "h", "xy": [0, 0], "kind": "head_anchor"}, {"id": "m", "xy": [2, 0]}, {"id": "e", "xy": [4, 0], "kind": "endpoint"}],
                 "edges": [{"id": "one", "start": "h", "end": "m", "points": [[0, 0], [2, 0]], "evidence": 0.5}, {"id": "two", "start": "m", "end": "e", "points": [[2, 0], [4, 0]], "evidence": 0.5}]}
        first = search_hypotheses(from_explicit_graph(base), "h", k=1)["hypotheses"][1]
        second = search_hypotheses(from_explicit_graph(split), "h", k=1)["hypotheses"][1]
        self.assertAlmostEqual(first["costs"]["evidence"], second["costs"]["evidence"])

    def test_mirror_transform_preserves_directed_truth_route(self):
        fixture = generate_case("curved_crossing")
        truth = fixture.truth["graph"]
        mirrored = {"nodes": [], "edges": []}
        for node in truth["nodes"]:
            copy = dict(node); copy["xy"] = [127 - node["xy"][0], node["xy"][1]]
            mirrored["nodes"].append(copy)
        for edge in truth["edges"]:
            copy = dict(edge); copy["points"] = [[127 - x, y] for x, y in edge["points"]]
            mirrored["edges"].append(copy)
        result = search_hypotheses(from_explicit_graph(mirrored), "h_left", k=5)
        self.assertIn(["left_j", "j_right"], [h["segment_ids"] for h in result["hypotheses"]])

    def test_signed_curvature_consistency_is_reflection_invariant(self):
        fixture = generate_case("curved_crossing")
        graph = from_explicit_graph(fixture.truth["graph"])
        original = transition_cost(graph, "j", ("left_j", True), ("j_right", True))["curvature"]
        mirrored = {"nodes": [], "edges": []}
        for node in fixture.truth["graph"]["nodes"]:
            copy = dict(node); copy["xy"] = [127 - node["xy"][0], node["xy"][1]]; mirrored["nodes"].append(copy)
        for edge in fixture.truth["graph"]["edges"]:
            copy = dict(edge); copy["points"] = [[127 - x, y] for x, y in edge["points"]]; mirrored["edges"].append(copy)
        reflected = transition_cost(from_explicit_graph(mirrored), "j", ("left_j", True), ("j_right", True))["curvature"]
        self.assertAlmostEqual(original, reflected)

    def test_returning_to_start_is_a_partial_terminal(self):
        graph = from_explicit_graph({"nodes": [
            {"id": "h", "xy": [0, 0], "kind": "head_anchor"}, {"id": "j", "xy": [1, 0], "kind": "junction"},
        ], "edges": [
            {"id": "out", "start": "h", "end": "j", "points": [[0, 0], [1, 0]]},
            {"id": "back", "start": "j", "end": "h", "points": [[1, 0], [0, 0]]},
        ]})
        result = search_hypotheses(graph, "h", k=5)
        returned = [h for h in result["hypotheses"] if h["segment_ids"] == ["out", "back"]]
        self.assertEqual(["partial"], [h["termination"] for h in returned])


if __name__ == "__main__":
    unittest.main()
