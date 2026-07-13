import unittest

from annotation.annotation_gui import UndoStack, ViewTransform
from annotation.annotation_models import SCHEMA_VERSION, ImageAnnotation


class AnnotationGuiCoreTests(unittest.TestCase):
    def test_view_transform_round_trip_and_cursor_zoom(self) -> None:
        transform = ViewTransform(scale=2.0, offset_x=15.0, offset_y=25.0)
        image_point = (42.5, 77.25)
        canvas_point = transform.image_to_canvas(image_point)

        self.assertAlmostEqual(transform.canvas_to_image(canvas_point)[0], image_point[0])
        self.assertAlmostEqual(transform.canvas_to_image(canvas_point)[1], image_point[1])

        cursor = (100.0, 120.0)
        before = transform.canvas_to_image(cursor)
        transform.zoom_at(cursor, 1.25)
        after = transform.canvas_to_image(cursor)

        self.assertAlmostEqual(before[0], after[0])
        self.assertAlmostEqual(before[1], after[1])

    def test_undo_redo_restores_annotation_snapshots(self) -> None:
        annotation = ImageAnnotation(
            schema_version=SCHEMA_VERSION,
            dataset_id="ward_gold_v1",
            image_id="synthetic",
            source_path="Raw_Ward_Data/synthetic.tif",
            source_sha256="0" * 64,
            width=10,
            height=10,
        )
        stack = UndoStack()
        stack.push(annotation)
        annotation.notes = "changed"

        previous = stack.undo(annotation)
        self.assertIsNotNone(previous)
        self.assertEqual(previous.notes, "")

        redone = stack.redo(previous)
        self.assertIsNotNone(redone)
        self.assertEqual(redone.notes, "changed")


if __name__ == "__main__":
    unittest.main()
