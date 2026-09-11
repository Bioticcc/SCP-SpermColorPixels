import html.parser
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from experiments.r4_viewer import write_image_viewer


class _Parser(html.parser.HTMLParser):
    def __init__(self):
        super().__init__()
        self.buttons = []

    def handle_starttag(self, tag, attrs):
        if tag == "button":
            self.buttons.append(dict(attrs))


class R4ViewerValidation(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.r1 = self.root / "r1"
        self.r1.mkdir()
        Image.fromarray(np.zeros((9, 9, 3), np.uint8)).save(self.r1 / "original.jpg")

    def tearDown(self):
        self.tmp.cleanup()

    @staticmethod
    def route(route_id, head="a", termination="endpoint", points=None, roi=None):
        return {
            "id": route_id,
            "head_id": head,
            "termination": termination,
            "metadata": {
                "points_xy_ordered": points if points is not None else [[1, 1], [7, 1]],
                "roi_xyxy": roi if roi is not None else [0, 0, 9, 9],
            },
        }

    def payload(self, hypotheses, selected, frozen=None, runner=None):
        frozen = frozen or hypotheses
        return {
            "relative_image": "x",
            "width": 9,
            "height": 9,
            "hypotheses": hypotheses,
            "assignment": {"selected_by_head": selected, **({"runner_up": runner} if runner else {})},
            "frozen_r2": {
                "hypotheses": frozen,
                "assignment": {"selected_by_head": {"a": frozen[0]["id"]}},
            },
        }

    def write(self, payload):
        out = self.root / "images" / "x"
        write_image_viewer(out, self.r1, payload)
        return (out / "index.html").read_text()

    def test_html_parser_route_button_and_colon_id(self):
        item = self.route("a:r4_attachment:1")
        page = self.write(self.payload([item], {"a": item["id"]}))
        parser = _Parser()
        parser.feed(page)
        self.assertIn({"data-route": "a:r4_attachment:1", "onclick": "one(this.dataset.route)"}, parser.buttons)

    def test_null_count_excludes_unselected_nulls(self):
        chosen = self.route("a:r4_attachment:1", termination="partial")
        selected_null = self.route("b:null", head="b", termination="null", points=[])
        unselected_null = self.route("c:null", head="c", termination="null", points=[])
        payload = self.payload([chosen, selected_null, unselected_null], {"a": chosen["id"], "b": selected_null["id"]})
        payload["frozen_r2"]["assignment"]["selected_by_head"] = {"a": chosen["id"]}
        page = self.write(payload)
        self.assertEqual(page.count('&quot;termination&quot;: &quot;null&quot;'), 1)

    def test_frozen_old_mapping_is_supported_when_absent_from_new(self):
        new = self.route("a:r4_attachment:1")
        old = self.route("a:old", points=[[2, 2], [6, 2]])
        page = self.write(self.payload([new], {"a": new["id"]}, frozen=[old]))
        self.assertIn('class="old"', page)

    def test_wrong_selected_head_raises(self):
        item = self.route("a:r4_attachment:1", head="a")
        with self.assertRaisesRegex(ValueError, "selected ID/head"):
            self.write(self.payload([item], {"b": item["id"]}))

    def test_bad_coordinates_and_roi_raise(self):
        item = self.route("a:r4_attachment:1", points=[[math.nan, 1], [7, 1]])
        with self.assertRaisesRegex(ValueError, "invalid route coordinate"):
            self.write(self.payload([item], {"a": item["id"]}))
        item = self.route("a:r4_attachment:1", points=[[1, 1], [10, 1]])
        with self.assertRaisesRegex(ValueError, "invalid route coordinate"):
            self.write(self.payload([item], {"a": item["id"]}))
        item = self.route("a:r4_attachment:1", roi=[0, 0, 10, 9])
        with self.assertRaisesRegex(ValueError, "invalid route ROI"):
            self.write(self.payload([item], {"a": item["id"]}))

    def test_missing_source_and_wrong_source_dimensions_raise(self):
        item = self.route("a:r4_attachment:1")
        self.r1.joinpath("original.jpg").unlink()
        with self.assertRaisesRegex(ValueError, "missing original"):
            self.write(self.payload([item], {"a": item["id"]}))
        Image.fromarray(np.zeros((8, 9, 3), np.uint8)).save(self.r1 / "original.jpg")
        with self.assertRaisesRegex(ValueError, "source dimensions"):
            self.write(self.payload([item], {"a": item["id"]}))

    def test_runner_up_button_and_layer_are_complete(self):
        item = self.route("a:r4_attachment:1")
        runner = {"selected_by_head": {"a": item["id"]}}
        page = self.write(self.payload([item], {"a": item["id"]}, runner=runner))
        self.assertIn("Runner-up complete assignment", page)
        self.assertIn('class="runner"', page)

    def test_scaled_preview_preserves_source_coordinate_canvas(self):
        item = self.route("a:r4_attachment:1")
        payload = self.payload([item], {"a": item["id"]})
        payload.update(width=2800, height=1800)
        Image.fromarray(np.zeros((900, 1400, 3), np.uint8)).save(self.r1 / "original.jpg")
        page = self.write(payload)
        self.assertIn('viewBox="0 0 2800 1800"', page)

    def test_r4_runner_up_does_not_use_frozen_runner_up(self):
        item = self.route("a:r4_attachment:1")
        payload = self.payload([item], {"a": item["id"]})
        payload["frozen_r2"]["assignment"]["runner_up"] = {"selected_by_head": {"a": "missing-old-option"}}
        self.assertNotIn('class="runner"', self.write(payload))


if __name__ == "__main__":
    unittest.main()
