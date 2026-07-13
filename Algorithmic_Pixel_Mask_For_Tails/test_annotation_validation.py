import unittest

from annotation.annotation_models import (
    SCHEMA_VERSION,
    CrossingAnnotation,
    CrossingContinuation,
    HeadAnnotation,
    ImageAnnotation,
    SpermInstance,
)
from annotation.annotation_validation import validate_annotation


def base_annotation() -> ImageAnnotation:
    return ImageAnnotation(
        schema_version=SCHEMA_VERSION,
        dataset_id="ward_gold_v1",
        image_id="synthetic",
        source_path="Raw_Ward_Data/synthetic.tif",
        source_sha256="0" * 64,
        width=100,
        height=100,
    )


class AnnotationValidationTests(unittest.TestCase):
    def test_valid_complete_instance_has_no_errors(self) -> None:
        annotation = base_annotation()
        annotation.sperm_instances.append(
            SpermInstance(
                instance_id="sperm_001",
                head=HeadAnnotation(type="ellipse", center=[20, 20], axes=[6, 4]),
                neck_point=[24, 20],
                tail_centerline=[[24, 20], [45, 25], [70, 40]],
                distal_endpoint=[70, 40],
                instance_status="full",
                head_visibility="full",
                tail_visibility="full",
                certainty="high",
            )
        )

        report = validate_annotation(annotation)

        self.assertEqual(report.errors, [])

    def test_invalid_schema_duplicate_ids_and_coordinates_are_errors(self) -> None:
        annotation = base_annotation()
        annotation.schema_version = "0.9"
        bad = SpermInstance(
            instance_id="sperm_001",
            head=HeadAnnotation(type="point_only", center=[200, 20]),
            tail_centerline=[[10, 10]],
            instance_status="full",
            head_visibility="full",
            tail_visibility="full",
        )
        annotation.sperm_instances.extend([bad, bad])

        report = validate_annotation(annotation)
        codes = {issue.code for issue in report.errors}

        self.assertIn("invalid_schema_version", codes)
        self.assertIn("duplicate_instance_id", codes)
        self.assertIn("coordinate_out_of_bounds", codes)
        self.assertIn("tail_centerline_too_short", codes)

    def test_broken_crossing_references_are_errors(self) -> None:
        annotation = base_annotation()
        annotation.crossings.append(
            CrossingAnnotation(
                crossing_id="crossing_001",
                center=[50, 50],
                radius_pixels=8,
                involved_instance_ids=["missing"],
                continuations=[
                    CrossingContinuation(
                        instance_id="missing",
                        incoming_point=[40, 50],
                        outgoing_point=[60, 50],
                    )
                ],
            )
        )

        report = validate_annotation(annotation)
        codes = {issue.code for issue in report.errors}

        self.assertIn("broken_crossing_instance_reference", codes)
        self.assertIn("broken_continuation_instance_reference", codes)

    def test_centerline_intersection_without_crossing_is_warning(self) -> None:
        annotation = base_annotation()
        annotation.sperm_instances.extend(
            [
                SpermInstance(
                    instance_id="sperm_001",
                    tail_centerline=[[10, 50], [90, 50]],
                    instance_status="full",
                    tail_visibility="full",
                ),
                SpermInstance(
                    instance_id="sperm_002",
                    tail_centerline=[[50, 10], [50, 90]],
                    instance_status="full",
                    tail_visibility="full",
                ),
            ]
        )

        report = validate_annotation(annotation)
        codes = {issue.code for issue in report.warnings}

        self.assertIn("unannotated_centerline_intersection", codes)

        annotation.crossings.append(
            CrossingAnnotation(
                crossing_id="crossing_001",
                center=[50, 50],
                radius_pixels=10,
                involved_instance_ids=["sperm_001", "sperm_002"],
            )
        )
        report_with_crossing = validate_annotation(annotation)
        codes_with_crossing = {issue.code for issue in report_with_crossing.warnings}
        self.assertNotIn("unannotated_centerline_intersection", codes_with_crossing)

    def test_boundary_and_indeterminate_certainty_warnings(self) -> None:
        annotation = base_annotation()
        annotation.sperm_instances.append(
            SpermInstance(
                instance_id="sperm_001",
                tail_centerline=[[1, 1], [30, 30]],
                instance_status="full",
                tail_visibility="full",
                certainty="high",
                crossing_ids=["crossing_001"],
            )
        )
        annotation.crossings.append(
            CrossingAnnotation(
                crossing_id="crossing_001",
                center=[20, 20],
                radius_pixels=5,
                involved_instance_ids=["sperm_001", "sperm_002"],
                determinacy="indeterminate",
            )
        )

        report = validate_annotation(annotation)
        codes = {issue.code for issue in report.warnings}

        self.assertIn("full_sperm_reaches_boundary", codes)
        self.assertIn("high_certainty_indeterminate_crossing", codes)


if __name__ == "__main__":
    unittest.main()
