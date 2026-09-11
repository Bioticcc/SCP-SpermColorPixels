"""Focused schema-to-page checks for the R2 review viewer."""
from __future__ import annotations

import tempfile
import unittest
import re
from pathlib import Path

from experiments.r2_viewer import write_image_viewer, write_index


def route(identifier: str, head: str, points: list[list[int]], *, offset=(10, 20)) -> dict:
    return {"id": identifier, "head_id": head, "cost": 1.0, "termination": "endpoint",
            "metadata": {"kind": "r1_route", "source_roi_offset_xy": list(offset),
                         "roi_xyxy": [999, 999, 1000, 1000],
                         "points_xy_ordered": points,
                         "r1_route": {"points_xy": points, "termination": "endpoint"}}}


class R2ViewerTests(unittest.TestCase):
    def test_page_uses_metadata_offset_and_selected_baseline_control(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); page = root / "images" / "image_a"; r1 = root / "r1" / "image_a"
            r1.mkdir(parents=True)
            control = {"id": "control", "head_id": "a", "cost": 0.0, "termination": "baseline_control",
                       "metadata": {"kind": "baseline_control", "source_roi_offset_xy": [30, 40],
                                    "roi_xyxy": [999, 999, 1000, 1000], "pixels_xy_unordered": [[1, 2], [2, 2]]}}
            first, second, other = route("first", "a", [[1, 2], [3, 4]]), route("second", "a", [[5, 6], [7, 8]]), route("other", "b", [[2, 3], [4, 5]])
            record = {"relative_image": "image_a.jpg", "width": 100, "height": 80, "components": [
                {"tail_id": "one", "roi_xyxy": [10, 20, 30, 40], "searches": {"a": {"hypotheses": [{"termination": "null"}, {"termination": "endpoint", "points_xy": [[1, 2], [3, 4]]}]}}},
                {"tail_id": "two", "roi_xyxy": [50, 60, 70, 80], "searches": {"a": {"hypotheses": [{"termination": "endpoint", "points_xy": [[4, 5], [6, 7]]}]}}},
            ]}
            assignment = {"selected_by_head": {"a": "control", "b": "other"}, "independent_by_head": {"a": "first", "b": "other"},
                          "groups": [{"id": "group_001", "assignment": {"status": "optimal", "head_alternatives": {
                              "a": {"selected_id": "control", "status": "optimal", "score_margin": 1.0,
                                    "alternative": {"selected_by_head": {"a": "second"}}}}}}]}
            write_image_viewer(page, r1, record, {"hypotheses": [control, first, second, other], "assignment": assignment, "baseline_changes": []})
            text = (page / "index.html").read_text()
            self.assertIn('d="M11.000,22.000 L13.000,24.000"', text)  # source_roi_offset_xy, not roi_xyxy
            self.assertIn('M31.000,42.000h0', text)  # retained unordered baseline-control raster
            self.assertIn('routes_joint', text)
            joint_layer = re.search(r'<g id="routes_joint"[^>]*>(.*?)</g>', text).group(1)
            control_geometry = re.search(r'<path id="(route_\d+)" d="M31\.000,42\.000h0', text).group(1)
            self.assertIn(f'href="#{control_geometry}"', joint_layer)  # selected joint control uses shared SVG geometry
            self.assertIn('2 component/head R1 rank-1 routes', text)
            self.assertIn('../../index.html', text)
            self.assertIn('assignment.json', text)
            self.assertIn('id="routes_alt_a"', text)
            # The complete alternative retains b from the untouched group/image incumbent.
            self.assertIn('href="#route_2"', text)

    def test_root_index_has_local_fixture_and_contact_sheet_links(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixtures = {"cases": [{"name": "case"}, {"name": "case_rendered"}],
                        "rendered_centerline_summary": {"case": {"full": {"declared_instances": 2}}}}
            write_index(root, [{"relative_image": "fixed.jpg", "page": "images/fixed/index.html", "heads": 1, "hypotheses": 2, "r1_rank1_routes": 1}], fixtures,
                        {"images": [{"relative_image": "fixed.jpg", "reason": "Fixed panel"}]})
            text = (root / "index.html").read_text()
            self.assertIn('fixtures/case/comparison.png', text)
            self.assertIn('fixtures/case/rendered/comparison.png', text)
            self.assertIn('fixed_panel_contact_sheet.png', text)
            self.assertNotIn('"cases"', text)


if __name__ == "__main__":
    unittest.main()
