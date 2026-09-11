"""Frozen-artifact adaptation checks for R3 export records."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from experiments.r3_adapter import reconstruct_image


class R3AdapterTests(unittest.TestCase):
    def test_reconstruction_keeps_source_coordinates_and_writes_local_assets(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); rgb = np.zeros((20, 30, 3), np.uint8); rgb[:, :, 0] = np.arange(30, dtype=np.uint8)
            tail = np.zeros((20, 30), np.uint8); tail[10, 5:16] = 255
            head = np.zeros((20, 30), np.uint8); head[8:11, 3:6] = 255
            Image.fromarray(rgb).save(root / "source.png"); Image.fromarray(tail).save(root / "tail.png"); Image.fromarray(head).save(root / "head.png")
            source = {"image": str(root / "source.png"), "relative_image": "source.png", "outputs": {"tail_mask": str(root / "tail.png"), "head_mask": str(root / "head.png")},
                      "tails": [{"component_id": 1, "bbox_xywh": [5, 10, 11, 1], "area_px": 11}], "heads": [{"head_id": 1}]}
            r1 = {"relative_image": "source.png", "width": 30, "height": 20, "components": [{"tail_id": 1, "roi_xyxy": [4, 9, 17, 12], "searches": {"1": {"diagnostics": {"truncated": False}}},
                                  "graph": {"nodes": [{"id": "j", "kind": "junction", "xy": [6, 1], "pixels_xy": [[6, 1]]}], "segments": [{"start": "j", "end": "e", "radii": [1.0]}]}}]}
            hypothesis = {"id": "r2:1:1:route", "head_id": "1", "termination": "endpoint", "metadata": {"kind": "r1_route", "tail_id": "1", "roi_xyxy": [4, 9, 17, 12], "points_xy_ordered": [[1, 1], [11, 1]], "r1_route": {"node_ids": ["j"]}}}
            r2 = {"relative_image": "source.png", "width": 30, "height": 20, "hypotheses": [hypothesis], "assignment": {"status": "optimal", "selected_by_head": {"1": hypothesis["id"]}, "groups": [{"head_ids": ["1"], "assignment": {"status": "optimal", "head_alternatives": {"1": {"score_margin": 0.2}}}}]}}
            output = reconstruct_image(root / "out", source, r1, r2, {"reconstruction": {}})
            self.assertEqual(1, len(output["instances"])); instance = output["instances"][0]
            self.assertEqual([0, 4, 20, 15], instance["roi_xyxy"])
            self.assertTrue(instance["files"]["original_crop"].startswith("instances/"))
            self.assertTrue((root / "out" / instance["files"]["instance_mask"]).is_file())
            exported = np.asarray(Image.open(root / "out" / instance["files"]["original_crop"]).convert("RGB"))
            x0, y0, x1, y1 = instance["roi_xyxy"]
            self.assertTrue(np.array_equal(exported, rgb[y0:y1, x0:x1]))
            self.assertEqual(["j"], instance["uncertainty"]["crossing_ids"])
            self.assertTrue((root / "out" / output["components"][0]["evidence_mask_file"]).is_file())
            self.assertTrue((root / "out" / output["components"][0]["crossing_regions"][0]["mask_file"]).is_file())
            highlighted = np.asarray(Image.open(root / "out" / instance["files"]["highlighted_crop"]).convert("RGB"))
            self.assertEqual(exported.shape, highlighted.shape)
            self.assertFalse(output["source_comparisons"]["production_statuses_changed"])

    def test_null_choice_does_not_create_an_instance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); image = np.zeros((4, 4), np.uint8)
            for name in ("source.png", "tail.png", "head.png"): Image.fromarray(image).save(root / name)
            source = {"image": str(root / "source.png"), "relative_image": "blank.png", "outputs": {"tail_mask": str(root / "tail.png"), "head_mask": str(root / "head.png")}, "tails": [], "heads": []}
            null = {"id": "null-anything", "head_id": "a", "termination": "null", "metadata": {}}
            output = reconstruct_image(root / "out", source, {"relative_image": "blank.png", "width": 4, "height": 4, "components": []}, {"relative_image": "blank.png", "width": 4, "height": 4, "hypotheses": [null], "assignment": {"selected_by_head": {"a": "null-anything"}}}, {"reconstruction": {}})
            self.assertEqual([], output["instances"])
            self.assertEqual("a", output["null_choices"][0]["head_id"])

    def test_missing_or_wrong_head_selection_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); image = np.zeros((4, 4), np.uint8)
            for name in ("source.png", "tail.png", "head.png"): Image.fromarray(image).save(root / name)
            source = {"image": str(root / "source.png"), "relative_image": "blank.png", "outputs": {"tail_mask": str(root / "tail.png"), "head_mask": str(root / "head.png")}, "tails": [], "heads": []}
            base = {"relative_image": "blank.png", "width": 4, "height": 4}
            with self.assertRaises(ValueError):
                reconstruct_image(root / "out", source, {**base, "components": []}, {**base, "hypotheses": [], "assignment": {"selected_by_head": {"a": "missing"}}}, {"reconstruction": {}})


if __name__ == "__main__":
    unittest.main()
