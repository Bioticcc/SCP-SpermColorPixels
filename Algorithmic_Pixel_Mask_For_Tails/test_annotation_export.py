import tempfile
import unittest
from pathlib import Path

from PIL import Image

from annotation.annotation_export import export_dataset
from annotation.annotation_io import create_blank_annotation, save_annotation
from annotation.annotation_models import HeadAnnotation, SpermInstance


class AnnotationExportTests(unittest.TestCase):
    def test_export_writes_manifest_summary_preview_rasters_and_orientation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            input_root = Path(tmp) / "Raw_Ward_Data"
            image_dir = input_root / "Sperm 40x Raw Photos"
            image_dir.mkdir(parents=True)
            image_path = image_dir / "tiny.tif"
            Image.new("RGB", (80, 60), (240, 240, 240)).save(image_path)
            annotation_root = Path(tmp) / "annotations"
            export_root = Path(tmp) / "exports"
            annotation = create_blank_annotation(image_path, input_root)
            annotation.sperm_instances.append(
                SpermInstance(
                    instance_id="sperm_001",
                    head=HeadAnnotation(type="ellipse", center=[20, 20], axes=[5, 4]),
                    neck_point=[24, 20],
                    tail_centerline=[[24, 20], [50, 30], [75, 45]],
                    distal_endpoint=[75, 45],
                    instance_status="full",
                    head_visibility="full",
                    tail_visibility="full",
                    certainty="high",
                )
            )
            save_annotation(annotation, annotation_root)

            manifest = export_dataset(input_root, annotation_root, export_root)

            self.assertEqual(manifest["image_count"], 1)
            self.assertTrue((export_root / "manifest.json").exists())
            self.assertTrue((annotation_root / "manifest.json").exists())
            self.assertTrue((export_root / "summary.csv").exists())
            self.assertTrue((export_root / "previews" / f"{annotation.image_id}_preview.png").exists())
            self.assertTrue((export_root / "rasters" / "head" / f"{annotation.image_id}_head.png").exists())
            self.assertTrue((export_root / "orientation_maps" / f"{annotation.image_id}_orientation.npz").exists())


if __name__ == "__main__":
    unittest.main()
