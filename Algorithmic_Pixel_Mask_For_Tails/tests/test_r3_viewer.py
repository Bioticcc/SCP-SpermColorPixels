"""Focused output contracts for the R3 static mask viewer."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from experiments.r3_viewer import write_image_viewer, write_index


def png(path: Path, pixels: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("L", (2, 2)); image.putdata(pixels); image.save(path)


class R3ViewerTests(unittest.TestCase):
    def test_mask_overlays_use_original_roi_and_share_assets(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); page = root / "images" / "image"; r1 = root / "r1" / "image"; r2 = root / "r2" / "image"
            for asset in ("instances/a/tail.png", "instances/a/head.png", "instances/a/instance.png", "instances/a/centerline.png",
                          "instances/b/tail.png", "instances/b/head.png", "instances/b/instance.png", "instances/b/centerline.png",
                          "components/tail_t1/evidence.png", "components/tail_t1/crossing_j.png"):
                png(page / asset, [0, 255, 255, 0])
            payload = {"relative_image": "sample.jpg", "width": 100, "height": 80, "instances": [
                {"id": "a", "head_id": "h1", "tail_id": "t1", "termination": "endpoint", "roi_xyxy": [10, 20, 12, 22],
                 "files": {"tail_mask": "instances/a/tail.png", "head_mask": "instances/a/head.png", "instance_mask": "instances/a/instance.png", "centerline": "instances/a/centerline.png"}, "measurements": {}, "uncertainty": {"solver_status": "optimal", "upstream_r1_truncated": False}},
                {"id": "b", "head_id": "h2", "tail_id": "t2", "termination": "partial", "roi_xyxy": [30, 40, 32, 42],
                 "files": {"tail_mask": "instances/b/tail.png", "head_mask": "instances/b/head.png", "instance_mask": "instances/b/instance.png", "centerline": "instances/b/centerline.png"}, "measurements": {}, "uncertainty": {"solver_status": "limit", "upstream_r1_truncated": True}},
            ], "null_choices": [{"head_id": "h3", "reason": "null"}], "components": [{"tail_id": "t1", "source_roi_xyxy": [10, 20, 12, 22], "evidence_mask_file": "components/tail_t1/evidence.png", "crossing_regions": [{"id": "j", "mask_file": "components/tail_t1/crossing_j.png"}], "core_diagnostics": {"ordinary_duplicate_pixels": 0, "shared_pixels": 1, "unsupported_centerline_pixels": {"a": 0}, "width_unknown_samples": {"a": 1}}}]}
            summary = write_image_viewer(page, r1, r2, {"relative_image": "sample.jpg"}, payload)
            text = (page / "index.html").read_text()
            self.assertIn('x="10" y="20" width="2" height="2"', text)
            self.assertIn('x="30" y="40" width="2" height="2"', text)
            self.assertIn('overlays/000_a_instance_mask.png', text)
            self.assertIn('overlays/001_b_instance_mask.png', text)
            self.assertIn('overlays/000_a_centerline.png', text)
            self.assertIn('showBoth()', text)
            self.assertIn('showMasks()', text)
            self.assertIn('showCenterlines()', text)
            self.assertIn('components/tail_t1/evidence.png', text)
            self.assertIn('components/tail_t1/crossing_j.png', text)
            self.assertIn('instances/a/centerline.png', text)
            self.assertIn('../r2/image/index.html', text)
            self.assertEqual(2, summary["displayable_masks"])
            self.assertTrue((page / "overlays" / "000_a_instance_mask.png").is_file())

    def test_missing_binary_asset_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); page = root / "images" / "image"
            payload = {"instances": [{"id": "a", "roi_xyxy": [0, 0, 2, 2], "files": {}, "measurements": {}, "uncertainty": {}}]}
            with self.assertRaises(ValueError):
                write_image_viewer(page, root / "r1", root / "r2", {}, payload)

    def test_v2_exports_add_head_priority_and_uncertainty_layers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); page = root / "images" / "image"
            files = {}
            for key in ("tail_mask", "head_mask", "instance_mask", "centerline", "head_evidence_mask", "ownership_uncertainty_mask", "head_priority_instance_mask"):
                asset = f"instances/a/{key}.png"; png(page / asset, [0, 255, 255, 0]); files[key] = asset
            payload = {"schema_version": "scp.r3.reconstruction.v2", "width": 10, "height": 10, "instances": [{"id": "a", "head_id": "1", "tail_id": "1", "roi_xyxy": [3, 4, 5, 6], "files": files, "measurements": {}, "uncertainty": {}}]}
            write_image_viewer(page, root / "r1", root / "r2", {}, payload)
            text = (page / "index.html").read_text()
            self.assertIn('showHeadPriority()', text)
            self.assertIn('showUncertainty()', text)
            self.assertIn('head-priority-layer', text)
            self.assertIn('uncertainty-layer', text)
            self.assertIn('Full detected head evidence', text)
            self.assertIn('head_evidence_mask', text)

    def test_index_has_local_assets_and_concise_scientific_limits(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_index(root, [{"relative_image": "fixed.jpg", "page": "images/fixed/index.html", "instances": 1, "null_choices": 0, "displayable_masks": 1}],
                        {"cases": [{"name": "curve", "directory": "curve"}]}, {"images": [{"relative_image": "fixed.jpg", "reason": "Fixed panel"}]})
            text = (root / "index.html").read_text()
            self.assertIn('fixtures/curve/comparison.png', text)
            self.assertIn('fixed_panel_contact_sheet.png', text)
            self.assertIn('Experimental reconstruction from selected R2 paths', text)


if __name__ == "__main__":
    unittest.main()
