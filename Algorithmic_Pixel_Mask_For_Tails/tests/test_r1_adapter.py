"""Identity and source-coordinate safeguards for frozen-evidence integration."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from experiments.r1_adapter import baseline_record, restore_component_labels, source_boundary_graph
from graph_construction import Node, Segment, SegmentGraph


class AdapterTests(unittest.TestCase):
    def test_labels_follow_saved_identity_not_connected_component_order(self):
        mask = np.zeros((20, 30), np.uint8)
        mask[2:5, 3:7] = 255
        mask[10:12, 20:25] = 255
        records = [{'component_id': 9, 'bbox_xywh': [3, 2, 4, 3], 'area_px': 12},
                   {'component_id': 2, 'bbox_xywh': [20, 10, 5, 2], 'area_px': 10}]
        labels = restore_component_labels(mask, list(reversed(records)))
        self.assertEqual(labels[3, 4], 9)
        self.assertEqual(labels[10, 21], 2)
        with self.assertRaises(ValueError):
            restore_component_labels(mask, records[:1])
        with self.assertRaises(ValueError):
            restore_component_labels(mask, [records[0], records[0]])

    def test_baseline_pixels_roundtrip_from_crop_to_roi(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            mask = np.zeros((7, 9), np.uint8)
            mask[2:6, 3] = 255
            path = root/'skeleton.png'
            Image.fromarray(mask).save(path)
            row = {'skeleton_mask': str(path), 'bbox_xyxy': [100, 200, 109, 207],
                   'crop_id': 'old', 'head_label_id': 4, 'candidate_status': 'ambiguous'}
            result = baseline_record(row, (80, 180), root)
            self.assertEqual(result['pixels_xy'], [[23, 22], [23, 23], [23, 24], [23, 25]])
            self.assertEqual(result['status'], 'ambiguous')
            with self.assertRaises(ValueError):
                baseline_record({**row, 'bbox_xyxy': [100, 200, 108, 207]}, (80, 180), root)

    def test_component_roi_edge_is_not_source_boundary(self):
        nodes = {'a': Node('a', (0, 3), 'boundary'), 'b': Node('b', (9, 3), 'endpoint')}
        graph = SegmentGraph(nodes, {'ab': Segment('ab', 'a', 'b', ((0, 3), (9, 3)))}, {}, (10, 10))
        interior = source_boundary_graph(graph, (40, 30), (100, 100))
        boundary = source_boundary_graph(graph, (0, 30), (100, 100))
        self.assertEqual(interior.nodes['a'].kind, 'endpoint')
        self.assertEqual(boundary.nodes['a'].kind, 'boundary')


if __name__ == '__main__':
    unittest.main()
