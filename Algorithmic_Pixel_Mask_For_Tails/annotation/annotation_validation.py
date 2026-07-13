#!/usr/bin/env python3
"""Validation rules for Ward gold sperm annotations."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Iterable

from .annotation_models import (
    ANNOTATION_STATUSES,
    BOUNDARY_SIDES,
    CERTAINTY_STATUSES,
    CROSSING_DETERMINACY,
    HEAD_TYPES,
    INSTANCE_STATUSES,
    SCHEMA_VERSION,
    TAIL_VISIBILITY_STATUSES,
    VISIBILITY_STATUSES,
    CrossingAnnotation,
    HeadAnnotation,
    ImageAnnotation,
    SpermInstance,
)


@dataclass
class ValidationThresholds:
    boundary_margin_px: float = 10.0
    endpoint_mismatch_px: float = 10.0
    neck_to_tail_start_px: float = 30.0
    neck_to_head_px: float = 40.0
    crossing_proximity_px: float = 25.0


@dataclass
class ValidationIssue:
    severity: str
    code: str
    message: str
    object_id: str = ""

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass
class ValidationReport:
    thresholds: ValidationThresholds = field(default_factory=ValidationThresholds)
    errors: list[ValidationIssue] = field(default_factory=list)
    warnings: list[ValidationIssue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def add_error(self, code: str, message: str, object_id: str = "") -> None:
        self.errors.append(ValidationIssue("error", code, message, object_id))

    def add_warning(self, code: str, message: str, object_id: str = "") -> None:
        self.warnings.append(ValidationIssue("warning", code, message, object_id))

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "thresholds": asdict(self.thresholds),
            "errors": [issue.to_dict() for issue in self.errors],
            "warnings": [issue.to_dict() for issue in self.warnings],
        }


def _distance(a: list[float] | None, b: list[float] | None) -> float | None:
    if a is None or b is None:
        return None
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def _point_in_bounds(point: list[float] | None, width: int, height: int) -> bool:
    if point is None:
        return True
    x, y = float(point[0]), float(point[1])
    return 0.0 <= x < width and 0.0 <= y < height


def _is_near_boundary(point: list[float], width: int, height: int, margin: float) -> bool:
    x, y = float(point[0]), float(point[1])
    return x <= margin or y <= margin or x >= width - 1 - margin or y >= height - 1 - margin


def _head_reference_point(head: HeadAnnotation) -> list[float] | None:
    if head.center is not None:
        return head.center
    if head.polygon:
        xs = [point[0] for point in head.polygon]
        ys = [point[1] for point in head.polygon]
        return [sum(xs) / len(xs), sum(ys) / len(ys)]
    return None


def _validate_point(
    report: ValidationReport,
    point: list[float] | None,
    width: int,
    height: int,
    code: str,
    label: str,
    object_id: str,
) -> None:
    if point is not None and not _point_in_bounds(point, width, height):
        report.add_error(code, f"{label} is outside image bounds", object_id)


def _validate_points(
    report: ValidationReport,
    points: Iterable[list[float]],
    width: int,
    height: int,
    code: str,
    label: str,
    object_id: str,
) -> None:
    for index, point in enumerate(points):
        if not _point_in_bounds(point, width, height):
            report.add_error(code, f"{label} point {index} is outside image bounds", object_id)


def _orientation(a: list[float], b: list[float], c: list[float]) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _on_segment(a: list[float], b: list[float], c: list[float]) -> bool:
    return (
        min(a[0], c[0]) - 1e-6 <= b[0] <= max(a[0], c[0]) + 1e-6
        and min(a[1], c[1]) - 1e-6 <= b[1] <= max(a[1], c[1]) + 1e-6
    )


def _segment_intersection(
    a1: list[float],
    a2: list[float],
    b1: list[float],
    b2: list[float],
) -> list[float] | None:
    o1 = _orientation(a1, a2, b1)
    o2 = _orientation(a1, a2, b2)
    o3 = _orientation(b1, b2, a1)
    o4 = _orientation(b1, b2, a2)
    if o1 * o2 > 0 or o3 * o4 > 0:
        return None

    if abs(o1) < 1e-6 and _on_segment(a1, b1, a2):
        return b1
    if abs(o2) < 1e-6 and _on_segment(a1, b2, a2):
        return b2
    if abs(o3) < 1e-6 and _on_segment(b1, a1, b2):
        return a1
    if abs(o4) < 1e-6 and _on_segment(b1, a2, b2):
        return a2

    denominator = (a1[0] - a2[0]) * (b1[1] - b2[1]) - (a1[1] - a2[1]) * (b1[0] - b2[0])
    if abs(denominator) < 1e-6:
        return None
    x_num = (
        (a1[0] * a2[1] - a1[1] * a2[0]) * (b1[0] - b2[0])
        - (a1[0] - a2[0]) * (b1[0] * b2[1] - b1[1] * b2[0])
    )
    y_num = (
        (a1[0] * a2[1] - a1[1] * a2[0]) * (b1[1] - b2[1])
        - (a1[1] - a2[1]) * (b1[0] * b2[1] - b1[1] * b2[0])
    )
    return [x_num / denominator, y_num / denominator]


def _crossing_near(point: list[float], crossings: list[CrossingAnnotation], threshold: float) -> bool:
    for crossing in crossings:
        if crossing.center is None:
            continue
        radius = max(float(crossing.radius_pixels), threshold)
        distance = _distance(point, crossing.center)
        if distance is not None and distance <= radius:
            return True
    return False


def _validate_enums(annotation: ImageAnnotation, report: ValidationReport) -> None:
    if annotation.schema_version != SCHEMA_VERSION:
        report.add_error("invalid_schema_version", f"Unsupported schema version: {annotation.schema_version}")
    if annotation.annotation_status not in ANNOTATION_STATUSES:
        report.add_error("invalid_annotation_status", annotation.annotation_status, annotation.image_id)

    for instance in annotation.sperm_instances:
        if instance.head.type not in HEAD_TYPES:
            report.add_error("invalid_head_type", instance.head.type, instance.instance_id)
        if instance.instance_status not in INSTANCE_STATUSES:
            report.add_error("invalid_instance_status", instance.instance_status, instance.instance_id)
        if instance.head_visibility not in VISIBILITY_STATUSES:
            report.add_error("invalid_head_visibility", instance.head_visibility, instance.instance_id)
        if instance.tail_visibility not in TAIL_VISIBILITY_STATUSES:
            report.add_error("invalid_tail_visibility", instance.tail_visibility, instance.instance_id)
        if instance.certainty not in CERTAINTY_STATUSES:
            report.add_error("invalid_certainty", instance.certainty, instance.instance_id)
        for side in instance.boundary_sides:
            if side not in BOUNDARY_SIDES:
                report.add_error("invalid_boundary_side", side, instance.instance_id)

    for crossing in annotation.crossings:
        if crossing.determinacy not in CROSSING_DETERMINACY:
            report.add_error("invalid_crossing_determinacy", crossing.determinacy, crossing.crossing_id)


def validate_annotation(
    annotation: ImageAnnotation,
    thresholds: ValidationThresholds | None = None,
) -> ValidationReport:
    thresholds = thresholds or ValidationThresholds()
    report = ValidationReport(thresholds=thresholds)
    width, height = int(annotation.width), int(annotation.height)

    if width <= 0 or height <= 0:
        report.add_error("invalid_image_size", "Image width and height must be positive", annotation.image_id)
    _validate_enums(annotation, report)

    instance_ids = [instance.instance_id for instance in annotation.sperm_instances]
    duplicate_instances = {value for value in instance_ids if instance_ids.count(value) > 1}
    for instance_id in sorted(duplicate_instances):
        report.add_error("duplicate_instance_id", f"Duplicate instance ID: {instance_id}", instance_id)

    crossing_ids = [crossing.crossing_id for crossing in annotation.crossings]
    duplicate_crossings = {value for value in crossing_ids if crossing_ids.count(value) > 1}
    for crossing_id in sorted(duplicate_crossings):
        report.add_error("duplicate_crossing_id", f"Duplicate crossing ID: {crossing_id}", crossing_id)

    instance_id_set = set(instance_ids)
    crossing_id_set = set(crossing_ids)

    for instance in annotation.sperm_instances:
        _validate_instance(annotation, instance, report, thresholds)
        for crossing_id in instance.crossing_ids:
            if crossing_id not in crossing_id_set:
                report.add_error("broken_instance_crossing_reference", crossing_id, instance.instance_id)

    for crossing in annotation.crossings:
        _validate_crossing(annotation, crossing, instance_id_set, report)

    _validate_unannotated_centerline_intersections(annotation, report, thresholds)
    return report


def _validate_instance(
    annotation: ImageAnnotation,
    instance: SpermInstance,
    report: ValidationReport,
    thresholds: ValidationThresholds,
) -> None:
    width, height = int(annotation.width), int(annotation.height)
    head = instance.head

    _validate_point(report, head.center, width, height, "coordinate_out_of_bounds", "head center", instance.instance_id)
    _validate_points(report, head.polygon, width, height, "coordinate_out_of_bounds", "head polygon", instance.instance_id)
    _validate_point(report, instance.neck_point, width, height, "coordinate_out_of_bounds", "neck point", instance.instance_id)
    _validate_points(report, instance.tail_centerline, width, height, "coordinate_out_of_bounds", "tail centerline", instance.instance_id)
    _validate_point(report, instance.distal_endpoint, width, height, "coordinate_out_of_bounds", "distal endpoint", instance.instance_id)

    if head.type == "ellipse" and (head.center is None or head.axes is None):
        report.add_error("incomplete_head_ellipse", "Ellipse head requires center and axes", instance.instance_id)
    if head.type == "polygon" and len(head.polygon) < 3:
        report.add_error("incomplete_head_polygon", "Polygon head requires at least three points", instance.instance_id)
    if head.type == "point_only" and head.center is None:
        report.add_error("incomplete_head_point", "Point-only head requires a center", instance.instance_id)

    if instance.head_visibility != "not_visible" and head.type == "not_visible":
        report.add_warning("missing_head", "Head is not annotated", instance.instance_id)
    if instance.neck_point is None:
        report.add_warning("missing_neck_point", "Neck point is not annotated", instance.instance_id)

    tail_declared_visible = instance.tail_visibility in {"full", "partial"} or instance.instance_status in {
        "full",
        "partial",
        "boundary_truncated",
    }
    if tail_declared_visible and len(instance.tail_centerline) < 2:
        report.add_error("tail_centerline_too_short", "Visible tail needs at least two centerline points", instance.instance_id)

    head_ref = _head_reference_point(head)
    head_neck_distance = _distance(head_ref, instance.neck_point)
    if head_neck_distance is not None and head_neck_distance > thresholds.neck_to_head_px:
        report.add_warning("neck_far_from_head", f"Neck is {head_neck_distance:.1f}px from head reference", instance.instance_id)

    if instance.neck_point is not None and instance.tail_centerline:
        distance = _distance(instance.neck_point, instance.tail_centerline[0])
        if distance is not None and distance > thresholds.neck_to_tail_start_px:
            report.add_warning("tail_start_far_from_neck", f"Tail start is {distance:.1f}px from neck", instance.instance_id)

    if instance.distal_endpoint is not None and instance.tail_centerline:
        distance = _distance(instance.distal_endpoint, instance.tail_centerline[-1])
        if distance is not None and distance > thresholds.endpoint_mismatch_px:
            report.add_warning("endpoint_mismatch", f"Endpoint is {distance:.1f}px from final centerline point", instance.instance_id)

    reaches_boundary = any(
        _is_near_boundary(point, width, height, thresholds.boundary_margin_px)
        for point in instance.tail_centerline
    )
    if instance.instance_status == "full" and reaches_boundary:
        report.add_warning("full_sperm_reaches_boundary", "Full sperm centerline approaches image boundary", instance.instance_id)
    if instance.instance_status == "boundary_truncated" and not reaches_boundary:
        report.add_warning("boundary_status_without_boundary", "Boundary-truncated sperm does not approach image boundary", instance.instance_id)

    if instance.certainty == "high":
        crossing_by_id = {crossing.crossing_id: crossing for crossing in annotation.crossings}
        for crossing_id in instance.crossing_ids:
            crossing = crossing_by_id.get(crossing_id)
            if crossing is not None and crossing.determinacy == "indeterminate":
                report.add_warning("high_certainty_indeterminate_crossing", crossing_id, instance.instance_id)


def _validate_crossing(
    annotation: ImageAnnotation,
    crossing: CrossingAnnotation,
    instance_id_set: set[str],
    report: ValidationReport,
) -> None:
    width, height = int(annotation.width), int(annotation.height)
    _validate_point(report, crossing.center, width, height, "coordinate_out_of_bounds", "crossing center", crossing.crossing_id)
    if crossing.radius_pixels < 0:
        report.add_error("invalid_crossing_radius", "Crossing radius cannot be negative", crossing.crossing_id)
    if len(crossing.involved_instance_ids) < 2:
        report.add_warning("crossing_too_few_instances", "Crossing involves fewer than two instances", crossing.crossing_id)
    for instance_id in crossing.involved_instance_ids:
        if instance_id not in instance_id_set:
            report.add_error("broken_crossing_instance_reference", instance_id, crossing.crossing_id)
    for continuation in crossing.continuations:
        if continuation.instance_id not in instance_id_set:
            report.add_error("broken_continuation_instance_reference", continuation.instance_id, crossing.crossing_id)
        _validate_point(report, continuation.incoming_point, width, height, "coordinate_out_of_bounds", "incoming continuation", crossing.crossing_id)
        _validate_point(report, continuation.outgoing_point, width, height, "coordinate_out_of_bounds", "outgoing continuation", crossing.crossing_id)


def _validate_unannotated_centerline_intersections(
    annotation: ImageAnnotation,
    report: ValidationReport,
    thresholds: ValidationThresholds,
) -> None:
    instances = [instance for instance in annotation.sperm_instances if len(instance.tail_centerline) >= 2]
    for left_index, left in enumerate(instances):
        for right in instances[left_index + 1 :]:
            for a1, a2 in zip(left.tail_centerline, left.tail_centerline[1:]):
                for b1, b2 in zip(right.tail_centerline, right.tail_centerline[1:]):
                    intersection = _segment_intersection(a1, a2, b1, b2)
                    if intersection is None:
                        continue
                    if not _crossing_near(intersection, annotation.crossings, thresholds.crossing_proximity_px):
                        report.add_warning(
                            "unannotated_centerline_intersection",
                            f"{left.instance_id} intersects {right.instance_id} without nearby crossing",
                            f"{left.instance_id},{right.instance_id}",
                        )
                        return
