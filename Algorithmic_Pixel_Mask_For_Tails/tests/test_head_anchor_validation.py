import unittest

import numpy as np

from graph_construction import from_explicit_graph
from head_anchors.candidates import propose_head_anchors


def graph(nodes, edges):
    return from_explicit_graph({"nodes": nodes, "edges": edges})


class HeadAnchorValidation(unittest.TestCase):
    def test_baseline_fractional_coordinates_preserved_for_junction_and_isolated(self):
        junction = graph(
            [{"id": "j", "xy": [4.5, 4.25], "kind": "junction"},
             {"id": "a", "xy": [2, 2]}, {"id": "b", "xy": [7, 2]}, {"id": "c", "xy": [5, 8]}],
            [{"id": "ja", "start": "j", "end": "a", "points": [[4.5, 4.25], [2, 2]]},
             {"id": "jb", "start": "j", "end": "b", "points": [[4.5, 4.25], [7, 2]]},
             {"id": "jc", "start": "j", "end": "c", "points": [[4.5, 4.25], [5, 8]]}],
        )
        isolated = graph([{"id": "i", "xy": [2.5, 3.75], "kind": "ordinary"}], [])
        mask = np.zeros((12, 12), bool); mask[3:7, 3:7] = True
        score = np.full((12, 12), 0.5, np.float32)
        self.assertEqual(propose_head_anchors(junction, mask, score, ["j"])[0]["point_xy"], [4.5, 4.25])
        self.assertEqual(propose_head_anchors(isolated, mask, score, ["i"])[0]["point_xy"], [2.5, 3.75])

    def test_coarse_and_dense_same_geometry_have_same_tangent(self):
        coarse = graph([{"id": "e", "xy": [2, 5]}, {"id": "z", "xy": [8, 5]}],
                       [{"id": "ez", "start": "e", "end": "z", "points": [[2, 5], [8, 5]]}])
        dense = graph([{"id": "e", "xy": [2, 5]}, {"id": "z", "xy": [8, 5]}],
                      [{"id": "ez", "start": "e", "end": "z", "points": [[2, 5], [4, 5], [6, 5], [8, 5]]}])
        mask = np.zeros((12, 12), bool); mask[4:7, 4:7] = True
        score = np.full((12, 12), 0.5, np.float32)
        a = propose_head_anchors(coarse, mask, score, ["e"])[0]
        b = propose_head_anchors(dense, mask, score, ["e"])[0]
        self.assertEqual(a["tail_direction"], b["tail_direction"])
        self.assertEqual(a["tangent_options"], b["tangent_options"])

    def test_bent_coarse_and_dense_same_geometry_have_same_tangent(self):
        coarse = graph([{"id": "e", "xy": [2, 5]}, {"id": "z", "xy": [10, 9]}],
                       [{"id": "ez", "start": "e", "end": "z", "points": [[2, 5], [4, 5], [8, 8], [10, 9]]}])
        dense = graph([{"id": "e", "xy": [2, 5]}, {"id": "z", "xy": [10, 9]}],
                      [{"id": "ez", "start": "e", "end": "z", "points": [[2, 5], [3, 5], [4, 5], [4.8, 5.6], [5.6, 6.2], [6.4, 6.8], [7.2, 7.4], [8, 8], [8.89442719, 8.4472136], [9.4472136, 8.7236068], [10, 9]]}])
        mask = np.zeros((16, 16), bool); mask[4:7, 4:7] = True
        score = np.full((16, 16), 0.5, np.float32)
        a = propose_head_anchors(coarse, mask, score, ["e"])[0]
        b = propose_head_anchors(dense, mask, score, ["e"])[0]
        self.assertAlmostEqual(a["tail_direction"][0], b["tail_direction"][0], places=5)
        self.assertAlmostEqual(a["tail_direction"][1], b["tail_direction"][1], places=5)

    def test_inside_head_with_incident_segment_has_unknown_alignment(self):
        g = graph([{"id": "i", "xy": [5, 5], "kind": "ordinary"}, {"id": "z", "xy": [10, 5]}],
                  [{"id": "iz", "start": "i", "end": "z", "points": [[5, 5], [10, 5]]}])
        mask = np.zeros((16, 16), bool); mask[3:8, 3:8] = True
        out = propose_head_anchors(g, mask, np.ones((16, 16), np.float32), ["i"])[0]
        self.assertEqual(out["uncertainty"], "inside_head_or_direct_touch")
        self.assertIsNone(out["alignment"])

    def test_diagonal_unsupported_run_uses_physical_distance(self):
        g = graph([{"id": "e", "xy": [9, 9]}, {"id": "z", "xy": [10, 10], "kind": "boundary"}],
                  [{"id": "ez", "start": "e", "end": "z", "points": [[9, 9], [10, 10]]}])
        mask = np.zeros((16, 16), bool); mask[3:8, 3:8] = True
        score = np.zeros((16, 16), np.float32); score[8, 8] = 0.5
        out = propose_head_anchors(g, mask, score, [], max_distance_px=8)
        self.assertEqual(["e"], [x["node_id"] for x in out])
        self.assertAlmostEqual(out[0]["longest_unsupported_run_px"], np.sqrt(2), delta=0.2)

    def test_head_only_support_does_not_create_attachment(self):
        g = graph([{"id": "e", "xy": [8, 5]}, {"id": "z", "xy": [10, 5]}],
                  [{"id": "ez", "start": "e", "end": "z", "points": [[8, 5], [10, 5]]}])
        mask = np.zeros((12, 12), bool); mask[4:7, 4:7] = True
        self.assertEqual([], propose_head_anchors(g, mask, np.zeros((12, 12), np.float32), []))

    def test_positive_faint_support_outside_neck_retains_attachment(self):
        g = graph([{"id": "e", "xy": [8, 5]}, {"id": "z", "xy": [10, 5]}],
                  [{"id": "ez", "start": "e", "end": "z", "points": [[8, 5], [10, 5]]}])
        mask = np.zeros((12, 12), bool); mask[4:7, 4:7] = True
        score = np.zeros((12, 12), np.float32); score[5, 7:9] = 0.3
        out = propose_head_anchors(g, mask, score, [])
        self.assertEqual(["e"], [x["node_id"] for x in out])
        self.assertGreaterEqual(out[0]["mean_color_support"], 0.25)

    def test_inside_head_has_unknown_alignment(self):
        g = graph([{"id": "i", "xy": [5.5, 5.5], "kind": "ordinary"}], [])
        mask = np.zeros((12, 12), bool); mask[3:8, 3:8] = True
        out = propose_head_anchors(g, mask, np.ones((12, 12), np.float32), ["i"])[0]
        self.assertEqual(out["uncertainty"], "inside_head_or_direct_touch")
        self.assertIsNone(out["alignment"])

    def test_opposite_tangent_is_soft_penalty_and_candidate_remains(self):
        g = graph(
            [{"id": "l", "xy": [2, 5]}, {"id": "lr", "xy": [1, 5], "kind": "boundary"},
             {"id": "r", "xy": [8, 5]}, {"id": "rl", "xy": [6, 5], "kind": "boundary"}],
            [{"id": "lr", "start": "l", "end": "lr", "points": [[2, 5], [1, 5]]},
             {"id": "rl", "start": "r", "end": "rl", "points": [[8, 5], [6, 5]]}],
        )
        mask = np.zeros((12, 12), bool); mask[4:7, 4:7] = True
        score = np.zeros((12, 12), np.float32); score[5, 1:3] = 0.5; score[5, 7:9] = 0.5
        out = propose_head_anchors(g, mask, score, [], max_distance_px=8)
        self.assertEqual({"l", "r"}, {x["node_id"] for x in out})
        by = {x["node_id"]: x for x in out}
        self.assertIsNotNone(by["r"]["alignment"])
        self.assertGreater(by["r"]["score"], by["l"]["score"])

    def test_invalid_baseline_node_and_malformed_masks_raise(self):
        g = graph([], [])
        mask = np.ones((4, 4), bool)
        with self.assertRaises(ValueError):
            propose_head_anchors(g, mask, np.zeros((4, 4), np.float32), ["missing"])
        with self.assertRaises(ValueError):
            propose_head_anchors(g, np.zeros((4, 4), bool), np.zeros((3, 4), np.float32), [])

    def test_valid_empty_head_returns_empty(self):
        self.assertEqual([], propose_head_anchors(graph([], []), np.zeros((2, 2), bool), np.zeros((2, 2), np.float32), []))


if __name__ == "__main__":
    unittest.main()
