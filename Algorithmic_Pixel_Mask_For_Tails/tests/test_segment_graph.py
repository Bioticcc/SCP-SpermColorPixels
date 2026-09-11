import json
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from graph_construction import build_segment_graph, from_explicit_graph
from algorithmic_tail_mask import zhang_suen_thinning


class SegmentGraphTests(unittest.TestCase):
    def test_x_has_one_junction_and_four_port_segments(self):
        skeleton = np.zeros((51, 51), np.uint8)
        cv2.line(skeleton, (5, 25), (45, 25), 255, 1)
        cv2.line(skeleton, (25, 5), (25, 45), 255, 1)
        graph = build_segment_graph(skeleton, tail_mask=cv2.dilate(skeleton, np.ones((3, 3), np.uint8)))
        junctions = [node for node in graph.nodes.values() if node.kind == "junction"]
        self.assertEqual(len(junctions), 1)
        self.assertEqual(len(graph.outgoing(junctions[0].id)), 4)
        self.assertEqual(graph.diagnostics["unrepresented_skeleton_pixels"], 0)
        self.assertTrue(all(len(segment.points_xy) > 0 for segment in graph.segments.values()))

    def test_curved_line_is_one_segment_between_endpoints(self):
        skeleton = np.zeros((60, 60), np.uint8)
        cv2.polylines(skeleton, [np.array([(4, 50), (15, 35), (31, 29), (54, 7)])], False, 255, 1)
        graph = build_segment_graph(skeleton)
        self.assertEqual(len(graph.segments), 1)
        self.assertEqual(sorted(node.kind for node in graph.nodes.values()), ["endpoint", "endpoint"])

    def test_multi_pixel_junction_is_one_neighborhood(self):
        skeleton = np.zeros((51, 51), np.uint8)
        cv2.line(skeleton, (5, 25), (45, 25), 255, 3)
        cv2.line(skeleton, (25, 5), (25, 45), 255, 3)
        skeleton = zhang_suen_thinning(skeleton)
        graph = build_segment_graph(skeleton)
        self.assertEqual(len([node for node in graph.nodes.values() if node.kind == "junction"]), 1)
        self.assertEqual(graph.diagnostics["unrepresented_skeleton_pixels"], 0)

    def test_dense_unthinned_junction_is_rejected_before_it_can_swallow_tracks(self):
        skeleton = np.zeros((51, 51), np.uint8)
        cv2.line(skeleton, (5, 25), (45, 25), 255, 3)
        cv2.line(skeleton, (25, 5), (25, 45), 255, 3)
        with self.assertRaisesRegex(ValueError, "oversized dense junction core"):
            build_segment_graph(skeleton)

    def test_realistic_thinning_keeps_x_as_four_ports_not_a_junction_ladder(self):
        mask = np.zeros((61, 61), np.uint8)
        cv2.line(mask, (6, 30), (54, 30), 255, 3)
        cv2.line(mask, (30, 6), (30, 54), 255, 3)
        graph = build_segment_graph(zhang_suen_thinning(mask))
        junction = next(node for node in graph.nodes.values() if node.kind == "junction")
        self.assertLessEqual(len(junction.pixels_xy), 5)
        self.assertEqual(len(graph.outgoing(junction.id)), 4)
        self.assertEqual(len(graph.segments), 4)
        self.assertEqual(graph.diagnostics["unrepresented_skeleton_edges"], 0)

    def test_degree_two_cycle_is_explicit_self_loop(self):
        skeleton = np.zeros((60, 60), np.uint8)
        cv2.circle(skeleton, (30, 30), 15, 255, 1)
        graph = build_segment_graph(skeleton)
        loops = [node for node in graph.nodes.values() if node.kind == "loop"]
        self.assertEqual(len(loops), 1)
        self.assertEqual(len(graph.segments), 1)
        segment_id = next(iter(graph.segments))
        self.assertEqual(graph.destination(segment_id, True), loops[0].id)
        self.assertEqual(len(graph.outgoing(loops[0].id)), 2)

    def test_interior_anchor_splits_segment_and_records_snap(self):
        skeleton = np.zeros((30, 50), np.uint8)
        cv2.line(skeleton, (3, 15), (45, 15), 255, 1)
        graph = build_segment_graph(skeleton, anchors_xy={"head_a": (21.4, 15.1)})
        anchor_id = graph.anchors["head_a"]
        self.assertEqual(graph.nodes[anchor_id].kind, "head_anchor")
        self.assertEqual(len(graph.outgoing(anchor_id)), 2)
        self.assertLess(graph.diagnostics["anchor_snaps"]["head_a"]["distance_px"], 1.0)

    def test_empty_and_disconnected_are_accounted_for(self):
        empty = build_segment_graph(np.zeros((10, 10), np.uint8))
        self.assertEqual(empty.nodes, {})
        skeleton = np.zeros((30, 30), np.uint8)
        cv2.line(skeleton, (2, 4), (10, 4), 255, 1)
        cv2.line(skeleton, (18, 20), (27, 20), 255, 1)
        graph = build_segment_graph(skeleton)
        self.assertEqual(len(graph.segments), 2)
        self.assertEqual(graph.diagnostics["unrepresented_skeleton_pixels"], 0)
        self.assertEqual(graph.diagnostics["unrepresented_skeleton_edges"], 0)

    def test_boundary_endpoint_is_explicit_port(self):
        skeleton = np.zeros((20, 20), np.uint8)
        cv2.line(skeleton, (0, 8), (14, 8), 255, 1)
        graph = build_segment_graph(skeleton)
        self.assertEqual(sorted(node.kind for node in graph.nodes.values()), ["boundary_port", "endpoint"])

    def test_anchors_on_one_junction_have_one_accurate_collision(self):
        skeleton = np.zeros((31, 31), np.uint8)
        cv2.line(skeleton, (3, 15), (27, 15), 255, 1)
        cv2.line(skeleton, (15, 3), (15, 27), 255, 1)
        graph = build_segment_graph(skeleton, anchors_xy={"a": (15, 15), "b": (15.1, 15.1)})
        junction_id = next(node.id for node in graph.nodes.values() if node.kind == "junction")
        self.assertEqual(graph.anchors, {"a": junction_id, "b": junction_id})
        self.assertEqual(len(graph.diagnostics["anchor_on_node"]), 1)
        self.assertEqual(graph.diagnostics["anchor_collisions"], [{
            "anchor_id": "b", "node_id": junction_id, "with_anchor_id": "a",
            "reason": "shared_structural_node",
        }])

    def test_explicit_parser_rejects_silent_geometry_corruption(self):
        base = {
            "nodes": [
                {"id": "a", "xy": [1, 2]},
                {"id": "b", "xy": [3, 4]},
            ],
            "segments": [{"id": "s", "start": "a", "end": "b", "points_xy": [[1, 2], [3, 4]]}],
        }
        cases = [
            {**base, "nodes": [*base["nodes"], {"id": "a", "xy": [5, 6]}]},
            {**base, "segments": [*base["segments"], {"id": "s", "start": "a", "end": "b"}]},
            {**base, "segments": [{"id": "s", "start": "a", "end": "b", "points_xy": [[1, 2], [3, float("nan")]]}]},
            {**base, "segments": [{"id": "s", "start": "a", "end": "b", "points_xy": [[1, 2], [3, 4]], "radii": [1.0]}]},
        ]
        for malformed in cases:
            with self.subTest(malformed=malformed):
                with self.assertRaises(ValueError):
                    from_explicit_graph(malformed)

    def test_transform_and_explicit_graph_round_trip(self):
        skeleton = np.zeros((31, 31), np.uint8)
        cv2.line(skeleton, (3, 15), (27, 15), 255, 1)
        graph = build_segment_graph(skeleton)
        flipped = build_segment_graph(cv2.flip(skeleton, 1))
        self.assertEqual((len(graph.nodes), len(graph.segments)), (len(flipped.nodes), len(flipped.segments)))
        explicit = from_explicit_graph(graph.to_dict(), shape=graph.shape)
        self.assertEqual(explicit.to_dict()["segments"], graph.to_dict()["segments"])
        json.dumps(explicit.to_dict())

    def test_noninterpolating_raster_transforms_preserve_topology(self):
        skeleton = np.zeros((41, 53), np.uint8)
        cv2.line(skeleton, (3, 20), (48, 20), 255, 1)
        cv2.line(skeleton, (25, 3), (25, 37), 255, 1)
        reference = build_segment_graph(skeleton)
        for transformed in (cv2.flip(skeleton, 0), cv2.flip(skeleton, 1), cv2.rotate(skeleton, cv2.ROTATE_180)):
            graph = build_segment_graph(transformed)
            with self.subTest(shape=transformed.shape):
                self.assertEqual(
                    (len(graph.nodes), len(graph.segments), graph.diagnostics["junction_count"]),
                    (len(reference.nodes), len(reference.segments), reference.diagnostics["junction_count"]),
                )
                self.assertEqual(graph.diagnostics["unrepresented_skeleton_edges"], 0)


if __name__ == "__main__":
    unittest.main()
