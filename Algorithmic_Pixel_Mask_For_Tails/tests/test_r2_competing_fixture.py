"""Contract checks for the bounded R2 endpoint-competition fixture."""
from __future__ import annotations

import unittest

import numpy as np

from graph_construction import from_explicit_graph
from path_hypotheses import search_hypotheses
from tests.fixtures.r2_competing import generate_competing_case


class R2CompetingFixtureTests(unittest.TestCase):
    def test_independent_endpoint_minima_conflict_at_right_endpoint(self):
        fixture = generate_competing_case()
        graph = from_explicit_graph(fixture.truth["graph"])
        for head_id, incoming in (("h_upper", "upper_j"), ("h_lower", "lower_j")):
            with self.subTest(head_id=head_id):
                endpoints = [item for item in search_hypotheses(graph, head_id, k=5)["hypotheses"]
                             if item["termination"] == "endpoint"]
                best = min(endpoints, key=lambda item: item["score"])
                self.assertEqual(best["segment_ids"], [incoming, "j_right"])
        self.assertEqual(fixture.truth["expected_behavior"]["local_conflict_endpoint"], "end_right")

    def test_declared_joint_paths_are_disjoint_and_rendered(self):
        fixture = generate_competing_case()
        paths = fixture.truth["true_edge_paths"]
        self.assertEqual(set(paths[0]["edge_ids"]) & set(paths[1]["edge_ids"]), set())
        self.assertEqual({path["node_ids"][1] for path in paths}, {"j"})
        self.assertEqual(
            fixture.truth["expected_behavior"]["declared_smooth_pairing_assumption"],
            "Synthetic assignment convention only; not real biological gold.",
        )
        for edge in fixture.truth["graph"]["edges"]:
            for x, y in edge["points"]:
                self.assertTrue(fixture.tail_mask[y, x], msg=edge["id"])
        for path in paths:
            self.assertTrue(np.any(fixture.instance_masks[path["instance_id"]]))

    def test_seed_repeat_and_supported_width_variations_are_stable(self):
        first, second = generate_competing_case(seed=17), generate_competing_case(seed=17)
        np.testing.assert_array_equal(first.rgb, second.rgb)
        np.testing.assert_array_equal(first.tail_mask, second.tail_mask)
        np.testing.assert_array_equal(first.head_mask, second.head_mask)
        widths = [generate_competing_case(width_px=width) for width in (1, 3, 5)]
        self.assertEqual([fixture.truth["graph"]["nodes"] for fixture in widths], [first.truth["graph"]["nodes"]] * 3)
        self.assertLess(widths[0].tail_mask.sum(), widths[1].tail_mask.sum())
        self.assertLess(widths[1].tail_mask.sum(), widths[2].tail_mask.sum())
        with self.assertRaises(ValueError):
            generate_competing_case(width_px=2)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
