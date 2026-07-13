#!/usr/bin/env python3
"""Regenerate manifest, summaries, previews, and rasters from annotations."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from annotation.annotation_io import (  # type: ignore[no-redef]
        collect_source_images,
        load_annotation_for_image,
        resolve_scp_path,
    )
    from annotation.annotation_models import ImageAnnotation  # type: ignore[no-redef]
    from annotation.annotation_validation import validate_annotation  # type: ignore[no-redef]
else:
    from .annotation_io import collect_source_images, load_annotation_for_image, resolve_scp_path
    from .annotation_models import ImageAnnotation
    from .annotation_validation import validate_annotation


PALETTE = [
    (31, 119, 180),
    (255, 127, 14),
    (44, 160, 44),
    (214, 39, 40),
    (148, 103, 189),
    (23, 190, 207),
    (227, 119, 194),
    (188, 189, 34),
    (127, 127, 127),
    (140, 86, 75),
]


def palette_color(index: int) -> tuple[int, int, int]:
    return PALETTE[index % len(PALETTE)]


def get_font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _points(points: list[list[float]]) -> list[tuple[float, float]]:
    return [(float(x), float(y)) for x, y in points]


def render_preview(image_path: Path, annotation: ImageAnnotation, output_path: Path) -> None:
    with Image.open(image_path) as image:
        base = image.convert("RGB")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = get_font(18)

    for index, instance in enumerate(annotation.sperm_instances):
        color = palette_color(index)
        rgba = (*color, 230)
        head = instance.head
        if head.type == "ellipse" and head.center is not None and head.axes is not None:
            x, y = head.center
            rx, ry = max(1.0, head.axes[0]), max(1.0, head.axes[1])
            draw.ellipse((x - rx, y - ry, x + rx, y + ry), outline=rgba, width=3)
        elif head.type == "polygon" and len(head.polygon) >= 3:
            draw.line(_points(head.polygon + [head.polygon[0]]), fill=rgba, width=3)
        elif head.type == "point_only" and head.center is not None:
            x, y = head.center
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=rgba)

        if instance.tail_centerline:
            draw.line(_points(instance.tail_centerline), fill=rgba, width=3, joint="curve")
            x, y = instance.tail_centerline[0]
            draw.text((x + 6, y + 6), instance.instance_id, fill=rgba, font=font)
        if instance.neck_point is not None:
            x, y = instance.neck_point
            draw.rectangle((x - 4, y - 4, x + 4, y + 4), fill=(*color, 255))
        if instance.distal_endpoint is not None:
            x, y = instance.distal_endpoint
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), outline=(*color, 255), width=2)

    for crossing in annotation.crossings:
        if crossing.center is None:
            continue
        x, y = crossing.center
        radius = max(3.0, crossing.radius_pixels)
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=(255, 255, 0, 240), width=3)
        draw.text((x + radius + 4, y), crossing.crossing_id, fill=(255, 255, 0, 255), font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.alpha_composite(base.convert("RGBA"), overlay).convert("RGB").save(output_path)


def _blank_raster(annotation: ImageAnnotation) -> np.ndarray:
    return np.zeros((int(annotation.height), int(annotation.width)), dtype=np.uint16)


def _draw_point(mask: np.ndarray, point: list[float] | None, value: int, radius: int = 4) -> None:
    if point is None:
        return
    cv2.circle(mask, (int(round(point[0])), int(round(point[1]))), radius, int(value), thickness=-1)


def _draw_polyline(mask: np.ndarray, points: list[list[float]], value: int, thickness: int = 3) -> None:
    if len(points) < 2:
        return
    arr = np.array([[int(round(x)), int(round(y))] for x, y in points], dtype=np.int32)
    cv2.polylines(mask, [arr], isClosed=False, color=int(value), thickness=thickness)


def write_rasters(annotation: ImageAnnotation, output_root: Path) -> None:
    head = _blank_raster(annotation)
    centerline = _blank_raster(annotation)
    endpoints = _blank_raster(annotation)
    necks = _blank_raster(annotation)
    crossings = _blank_raster(annotation)
    cos2 = np.zeros_like(head, dtype=np.float32)
    sin2 = np.zeros_like(head, dtype=np.float32)
    orientation_ids = np.zeros_like(head, dtype=np.uint16)

    for index, instance in enumerate(annotation.sperm_instances, start=1):
        h = instance.head
        if h.type == "ellipse" and h.center is not None and h.axes is not None:
            cv2.ellipse(
                head,
                (int(round(h.center[0])), int(round(h.center[1]))),
                (max(1, int(round(h.axes[0]))), max(1, int(round(h.axes[1])))),
                float(h.angle_degrees),
                0,
                360,
                int(index),
                thickness=-1,
            )
        elif h.type == "polygon" and len(h.polygon) >= 3:
            arr = np.array([[int(round(x)), int(round(y))] for x, y in h.polygon], dtype=np.int32)
            cv2.fillPoly(head, [arr], color=int(index))
        elif h.type == "point_only":
            _draw_point(head, h.center, index, radius=5)

        _draw_polyline(centerline, instance.tail_centerline, index)
        _draw_point(endpoints, instance.distal_endpoint, index, radius=4)
        _draw_point(necks, instance.neck_point, index, radius=4)

        for a, b in zip(instance.tail_centerline, instance.tail_centerline[1:]):
            segment_mask = np.zeros_like(head, dtype=np.uint8)
            _draw_polyline(segment_mask, [a, b], 1, thickness=3)
            theta = math.atan2(float(b[1]) - float(a[1]), float(b[0]) - float(a[0]))
            rows = segment_mask > 0
            cos2[rows] = math.cos(2.0 * theta)
            sin2[rows] = math.sin(2.0 * theta)
            orientation_ids[rows] = index

    for index, crossing in enumerate(annotation.crossings, start=1):
        radius = max(1, int(round(crossing.radius_pixels)))
        _draw_point(crossings, crossing.center, index, radius=radius)

    raster_root = output_root / "rasters"
    for folder, mask in (
        ("head", head),
        ("centerline", centerline),
        ("endpoint", endpoints),
        ("neck", necks),
        ("crossing", crossings),
    ):
        out_dir = raster_root / folder
        out_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.clip(mask, 0, 255).astype(np.uint8)).save(out_dir / f"{annotation.image_id}_{folder}.png")

    orientation_dir = output_root / "orientation_maps"
    orientation_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        orientation_dir / f"{annotation.image_id}_orientation.npz",
        cos2theta=cos2,
        sin2theta=sin2,
        instance_id_map=orientation_ids,
    )


def build_summary_row(annotation: ImageAnnotation, exists: bool) -> dict[str, Any]:
    report = validate_annotation(annotation)
    return {
        "image_id": annotation.image_id,
        "source_path": annotation.source_path,
        "magnification": annotation.magnification,
        "annotation_exists": str(bool(exists)),
        "annotation_status": annotation.annotation_status,
        "sperm_instances": len(annotation.sperm_instances),
        "crossings": len(annotation.crossings),
        "validation_errors": len(report.errors),
        "validation_warnings": len(report.warnings),
        "source_sha256": annotation.source_sha256,
    }


def export_dataset(input_root: Path, annotation_root: Path, export_root: Path) -> dict[str, Any]:
    input_root = resolve_scp_path(input_root)
    annotation_root = resolve_scp_path(annotation_root)
    export_root = resolve_scp_path(export_root)
    export_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    manifest_images: list[dict[str, Any]] = []
    for image_path in collect_source_images(input_root):
        load_result = load_annotation_for_image(image_path, input_root, annotation_root)
        annotation = load_result.annotation
        exists = load_result.path.exists()
        rows.append(build_summary_row(annotation, exists=exists))
        manifest_images.append(
            {
                "image_id": annotation.image_id,
                "source_path": annotation.source_path,
                "source_sha256": annotation.source_sha256,
                "width": annotation.width,
                "height": annotation.height,
                "magnification": annotation.magnification,
                "annotation_path": str(load_result.path),
                "annotation_exists": exists,
            }
        )
        render_preview(image_path, annotation, export_root / "previews" / f"{annotation.image_id}_preview.png")
        write_rasters(annotation, export_root)

    summary_path = export_root / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "image_id",
            "source_path",
            "magnification",
            "annotation_exists",
            "annotation_status",
            "sperm_instances",
            "crossings",
            "validation_errors",
            "validation_warnings",
            "source_sha256",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    manifest = {
        "dataset_id": "ward_gold_v1",
        "schema_version": "1.0.0",
        "input_root": str(input_root),
        "annotation_root": str(annotation_root),
        "export_root": str(export_root),
        "image_count": len(manifest_images),
        "images": manifest_images,
        "summary_csv": str(summary_path),
    }
    manifest_path = export_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    annotation_root.mkdir(parents=True, exist_ok=True)
    (annotation_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Ward gold annotation derived artifacts.")
    parser.add_argument("--input-root", type=Path, default=Path("Raw_Ward_Data"))
    parser.add_argument("--annotation-root", type=Path, default=Path("annotations/ward_gold_v1"))
    parser.add_argument("--export-root", type=Path, default=Path("annotations/ward_gold_v1/exports"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = export_dataset(args.input_root, args.annotation_root, args.export_root)
    print(json.dumps({"image_count": manifest["image_count"], "export_root": manifest["export_root"]}, indent=2))


if __name__ == "__main__":
    main()
