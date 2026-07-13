#!/usr/bin/env python3
"""Schema-versioned data models for Ward gold sperm annotations."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


SCHEMA_VERSION = "1.0.0"
DEFAULT_DATASET_ID = "ward_gold_v1"

HEAD_TYPES = {"ellipse", "polygon", "point_only", "not_visible"}
ANNOTATION_STATUSES = {"not_started", "in_progress", "review_needed", "complete", "locked"}
INSTANCE_STATUSES = {"full", "partial", "boundary_truncated", "occluded", "uncertain"}
VISIBILITY_STATUSES = {"full", "partial", "not_visible", "uncertain"}
TAIL_VISIBILITY_STATUSES = {"full", "partial", "uncertain"}
CERTAINTY_STATUSES = {"high", "medium", "low", "indeterminate"}
CROSSING_DETERMINACY = {"determinable", "indeterminate", "uncertain"}
BOUNDARY_SIDES = {"left", "right", "top", "bottom"}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _point_or_none(value: Any) -> list[float] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    try:
        return [float(value[0]), float(value[1])]
    except (TypeError, ValueError):
        return None


def _point_list(value: Any) -> list[list[float]]:
    if not isinstance(value, list):
        return []
    points: list[list[float]] = []
    for item in value:
        point = _point_or_none(item)
        if point is not None:
            points.append(point)
    return points


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item)]


@dataclass
class HeadAnnotation:
    type: str = "not_visible"
    center: list[float] | None = None
    axes: list[float] | None = None
    angle_degrees: float = 0.0
    polygon: list[list[float]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "center": deepcopy(self.center),
            "axes": deepcopy(self.axes),
            "angle_degrees": float(self.angle_degrees),
            "polygon": deepcopy(self.polygon),
        }

    @classmethod
    def from_dict(cls, data: Any) -> "HeadAnnotation":
        if not isinstance(data, dict):
            return cls()
        try:
            angle = float(data.get("angle_degrees", 0.0))
        except (TypeError, ValueError):
            angle = 0.0
        return cls(
            type=str(data.get("type", "not_visible")),
            center=_point_or_none(data.get("center")),
            axes=_point_or_none(data.get("axes")),
            angle_degrees=angle,
            polygon=_point_list(data.get("polygon", [])),
        )


@dataclass
class CrossingContinuation:
    instance_id: str = ""
    incoming_point: list[float] | None = None
    outgoing_point: list[float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "incoming_point": deepcopy(self.incoming_point),
            "outgoing_point": deepcopy(self.outgoing_point),
        }

    @classmethod
    def from_dict(cls, data: Any) -> "CrossingContinuation":
        if not isinstance(data, dict):
            return cls()
        return cls(
            instance_id=str(data.get("instance_id", "")),
            incoming_point=_point_or_none(data.get("incoming_point")),
            outgoing_point=_point_or_none(data.get("outgoing_point")),
        )


@dataclass
class CrossingAnnotation:
    crossing_id: str
    center: list[float] | None = None
    radius_pixels: float = 0.0
    involved_instance_ids: list[str] = field(default_factory=list)
    determinacy: str = "uncertain"
    continuations: list[CrossingContinuation] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "crossing_id": self.crossing_id,
            "center": deepcopy(self.center),
            "radius_pixels": float(self.radius_pixels),
            "involved_instance_ids": list(self.involved_instance_ids),
            "determinacy": self.determinacy,
            "continuations": [row.to_dict() for row in self.continuations],
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Any) -> "CrossingAnnotation":
        if not isinstance(data, dict):
            data = {}
        try:
            radius = float(data.get("radius_pixels", 0.0))
        except (TypeError, ValueError):
            radius = 0.0
        return cls(
            crossing_id=str(data.get("crossing_id", "")),
            center=_point_or_none(data.get("center")),
            radius_pixels=radius,
            involved_instance_ids=_string_list(data.get("involved_instance_ids", [])),
            determinacy=str(data.get("determinacy", "uncertain")),
            continuations=[
                CrossingContinuation.from_dict(item)
                for item in data.get("continuations", [])
                if isinstance(item, dict)
            ],
            notes=str(data.get("notes", "")),
        )


@dataclass
class SpermInstance:
    instance_id: str
    head: HeadAnnotation = field(default_factory=HeadAnnotation)
    neck_point: list[float] | None = None
    tail_centerline: list[list[float]] = field(default_factory=list)
    distal_endpoint: list[float] | None = None
    distal_endpoint_visible: bool = True
    instance_status: str = "uncertain"
    head_visibility: str = "uncertain"
    tail_visibility: str = "uncertain"
    certainty: str = "low"
    crossing_ids: list[str] = field(default_factory=list)
    boundary_sides: list[str] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "head": self.head.to_dict(),
            "neck_point": deepcopy(self.neck_point),
            "tail_centerline": deepcopy(self.tail_centerline),
            "distal_endpoint": deepcopy(self.distal_endpoint),
            "distal_endpoint_visible": bool(self.distal_endpoint_visible),
            "instance_status": self.instance_status,
            "head_visibility": self.head_visibility,
            "tail_visibility": self.tail_visibility,
            "certainty": self.certainty,
            "crossing_ids": list(self.crossing_ids),
            "boundary_sides": list(self.boundary_sides),
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Any) -> "SpermInstance":
        if not isinstance(data, dict):
            data = {}
        return cls(
            instance_id=str(data.get("instance_id", "")),
            head=HeadAnnotation.from_dict(data.get("head", {})),
            neck_point=_point_or_none(data.get("neck_point")),
            tail_centerline=_point_list(data.get("tail_centerline", [])),
            distal_endpoint=_point_or_none(data.get("distal_endpoint")),
            distal_endpoint_visible=bool(data.get("distal_endpoint_visible", True)),
            instance_status=str(data.get("instance_status", "uncertain")),
            head_visibility=str(data.get("head_visibility", "uncertain")),
            tail_visibility=str(data.get("tail_visibility", "uncertain")),
            certainty=str(data.get("certainty", "low")),
            crossing_ids=_string_list(data.get("crossing_ids", [])),
            boundary_sides=_string_list(data.get("boundary_sides", [])),
            notes=str(data.get("notes", "")),
        )


@dataclass
class ImageAnnotation:
    schema_version: str
    dataset_id: str
    image_id: str
    source_path: str
    source_sha256: str
    width: int
    height: int
    magnification: str = "unknown"
    annotation_status: str = "not_started"
    annotator: str = ""
    reviewer: str = ""
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)
    notes: str = ""
    sperm_instances: list[SpermInstance] = field(default_factory=list)
    crossings: list[CrossingAnnotation] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "dataset_id": self.dataset_id,
            "image_id": self.image_id,
            "source_path": self.source_path,
            "source_sha256": self.source_sha256,
            "width": int(self.width),
            "height": int(self.height),
            "magnification": self.magnification,
            "annotation_status": self.annotation_status,
            "annotator": self.annotator,
            "reviewer": self.reviewer,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "notes": self.notes,
            "sperm_instances": [instance.to_dict() for instance in self.sperm_instances],
            "crossings": [crossing.to_dict() for crossing in self.crossings],
        }

    @classmethod
    def from_dict(cls, data: Any) -> "ImageAnnotation":
        if not isinstance(data, dict):
            raise ValueError("Annotation JSON must contain an object at the top level")
        return cls(
            schema_version=str(data.get("schema_version", "")),
            dataset_id=str(data.get("dataset_id", DEFAULT_DATASET_ID)),
            image_id=str(data.get("image_id", "")),
            source_path=str(data.get("source_path", "")),
            source_sha256=str(data.get("source_sha256", "")),
            width=int(data.get("width", 0) or 0),
            height=int(data.get("height", 0) or 0),
            magnification=str(data.get("magnification", "unknown")),
            annotation_status=str(data.get("annotation_status", "not_started")),
            annotator=str(data.get("annotator", "")),
            reviewer=str(data.get("reviewer", "")),
            created_at=str(data.get("created_at", utc_now_iso())),
            updated_at=str(data.get("updated_at", utc_now_iso())),
            notes=str(data.get("notes", "")),
            sperm_instances=[
                SpermInstance.from_dict(item)
                for item in data.get("sperm_instances", [])
                if isinstance(item, dict)
            ],
            crossings=[
                CrossingAnnotation.from_dict(item)
                for item in data.get("crossings", [])
                if isinstance(item, dict)
            ],
        )

    def clone(self) -> "ImageAnnotation":
        return ImageAnnotation.from_dict(self.to_dict())

    def next_instance_id(self) -> str:
        used = {instance.instance_id for instance in self.sperm_instances}
        index = 1
        while f"sperm_{index:03d}" in used:
            index += 1
        return f"sperm_{index:03d}"

    def next_crossing_id(self) -> str:
        used = {crossing.crossing_id for crossing in self.crossings}
        index = 1
        while f"crossing_{index:03d}" in used:
            index += 1
        return f"crossing_{index:03d}"

    def get_instance(self, instance_id: str) -> SpermInstance | None:
        for instance in self.sperm_instances:
            if instance.instance_id == instance_id:
                return instance
        return None

    def get_crossing(self, crossing_id: str) -> CrossingAnnotation | None:
        for crossing in self.crossings:
            if crossing.crossing_id == crossing_id:
                return crossing
        return None
