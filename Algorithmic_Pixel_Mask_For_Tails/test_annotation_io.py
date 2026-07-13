import json
import tempfile
import unittest
from pathlib import Path

from annotation.annotation_io import (
    collect_source_images,
    create_blank_annotation,
    load_annotation_for_image,
    read_annotation_file,
    save_annotation,
)
from annotation.annotation_models import SCHEMA_VERSION, HeadAnnotation, SpermInstance


class AnnotationIOTests(unittest.TestCase):
    def setUp(self) -> None:
        self.input_root = Path("Raw_Ward_Data")
        self.image_path = collect_source_images(self.input_root)[0]

    def test_blank_annotation_uses_real_ward_image_metadata(self) -> None:
        annotation = create_blank_annotation(self.image_path, self.input_root)

        self.assertEqual(annotation.schema_version, SCHEMA_VERSION)
        self.assertTrue(annotation.image_id)
        self.assertTrue(annotation.source_path.startswith("Raw_Ward_Data/"))
        self.assertEqual(len(annotation.source_sha256), 64)
        self.assertGreater(annotation.width, 0)
        self.assertGreater(annotation.height, 0)

    def test_atomic_save_round_trip_and_backup_rotation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            annotation = create_blank_annotation(self.image_path, self.input_root)
            annotation.sperm_instances.append(
                SpermInstance(
                    instance_id="sperm_001",
                    head=HeadAnnotation(type="point_only", center=[10, 20]),
                    neck_point=[11, 21],
                    tail_centerline=[[11, 21], [30, 40]],
                    distal_endpoint=[30, 40],
                    instance_status="full",
                    head_visibility="full",
                    tail_visibility="full",
                    certainty="high",
                )
            )

            first = save_annotation(annotation, root)
            self.assertTrue(first.path.exists())
            loaded = read_annotation_file(first.path)
            self.assertEqual(loaded.sperm_instances[0].instance_id, "sperm_001")

            mtime = first.mtime_ns
            for index in range(7):
                annotation.notes = f"version {index}"
                result = save_annotation(annotation, root, mtime)
                self.assertIsNone(result.conflict_path)
                mtime = result.mtime_ns

            backups = list((root / "backups" / annotation.image_id).glob("*.json"))
            self.assertLessEqual(len(backups), 5)

    def test_conflict_copy_when_disk_file_changed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            annotation = create_blank_annotation(self.image_path, self.input_root)
            first = save_annotation(annotation, root)

            disk_copy = read_annotation_file(first.path)
            disk_copy.notes = "external edit"
            second = save_annotation(disk_copy, root, first.mtime_ns)

            annotation.notes = "stale edit"
            conflict = save_annotation(annotation, root, first.mtime_ns)

            self.assertIsNotNone(conflict.conflict_path)
            self.assertTrue(conflict.conflict_path.exists())  # type: ignore[union-attr]
            self.assertEqual(read_annotation_file(first.path).notes, "external edit")
            self.assertNotEqual(second.mtime_ns, first.mtime_ns)

    def test_recovery_from_latest_valid_backup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            annotation = create_blank_annotation(self.image_path, self.input_root)
            first = save_annotation(annotation, root)
            annotation.notes = "backup version"
            second = save_annotation(annotation, root, first.mtime_ns)
            self.assertIsNotNone(second.backup_path)
            first.path.write_text("{bad json", encoding="utf-8")

            recovered = load_annotation_for_image(self.image_path, self.input_root, root)

            self.assertIsNotNone(recovered.recovered_from_backup)
            self.assertIsInstance(json.loads(recovered.recovered_from_backup.read_text(encoding="utf-8")), dict)  # type: ignore[union-attr]


if __name__ == "__main__":
    unittest.main()
