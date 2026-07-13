#!/usr/bin/env python3
"""Algorithmic color-mask proof of concept for sperm tail detection.

This script is intentionally separate from the model-training pipeline. It
uses the brown/olive stain in the Ward RGB TIFF images to create high-recall
tail, head, and overlap masks for visual review and future tuning.
"""

from __future__ import annotations

import argparse
import csv
import heapq
import json
import shutil
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError


IMAGE_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
SCRIPT_DIR = Path(__file__).resolve().parent
ASSIGNMENT_PALETTE = [
    np.array([0, 255, 120], dtype=np.float32),
    np.array([255, 150, 0], dtype=np.float32),
    np.array([0, 150, 255], dtype=np.float32),
    np.array([220, 90, 255], dtype=np.float32),
    np.array([255, 230, 0], dtype=np.float32),
    np.array([0, 235, 235], dtype=np.float32),
    np.array([255, 90, 120], dtype=np.float32),
    np.array([120, 255, 0], dtype=np.float32),
]


@dataclass
class ComponentInfo:
    component_id: int
    kind: str
    area_px: int
    bbox_xywh: list[int]
    centroid_xy: list[float]
    aspect_ratio: float
    fill_ratio: float
    circularity: float | None
    skeleton_pixels: int | None = None
    endpoints_xy: list[list[int]] | None = None
    branchpoints_xy: list[list[int]] | None = None
    width_p95_px: float | None = None
    width_median_px: float | None = None
    has_overlap: bool | None = None
    overlap_reasons: list[str] | None = None
    head_matches: dict[str, Any] | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find sperm tails from Ward RGB images using algorithmic color masks."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("Raw_Ward_Data"),
        help="Root containing Ward images. Relative paths resolve inside this PoC folder.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/updated_outputs_path_v2"),
        help="Output folder. Relative paths resolve inside this PoC folder.",
    )
    parser.add_argument(
        "--path-v2",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use deterministic head-to-endpoint skeleton path scoring for final tail assignments.",
    )
    parser.add_argument(
        "--path-mode",
        choices=["legacy", "geometry-only", "color-only", "hybrid"],
        default="hybrid",
        help="Path assignment scoring mode. legacy preserves full-component/flood-split behavior.",
    )
    parser.add_argument(
        "--normalization",
        choices=["raw", "local-background", "white-balance"],
        default="raw",
        help="Color normalization used before Lab/HSV path continuity scoring.",
    )
    parser.add_argument(
        "--acceptance-profile",
        choices=["conservative", "balanced-review"],
        default="conservative",
        help="Candidate status policy. balanced-review accepts some good crops with mask/path risk flags.",
    )
    parser.add_argument(
        "--diagnostics",
        action="store_true",
        help="Write expanded candidate diagnostic overlays under diagnostics/.",
    )
    parser.add_argument(
        "--diagnostic-limit",
        type=int,
        default=None,
        help="Optional maximum number of candidate diagnostics to write per image.",
    )
    parser.add_argument(
        "--tail-debug-overlays",
        action="store_true",
        help="Write full-frame tail assignment debug overlays. Expensive on 40x batches.",
    )
    parser.add_argument(
        "--tail-debug-limit",
        type=int,
        default=None,
        help="Optional maximum number of full-frame tail debug overlays to write per image.",
    )
    parser.add_argument(
        "--record-ablation-scores",
        action="store_true",
        help="Record inactive geometry/color/hybrid path scores. Adds color-scoring cost in geometry-only runs.",
    )
    parser.add_argument(
        "--path-fast-isolated",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep legacy full-component assignment for clean isolated single-head tails.",
    )
    parser.add_argument(
        "--mode",
        choices=["high-recall", "balanced", "high-precision"],
        default="high-recall",
        help="Threshold preset for candidate masking and cleanup.",
    )
    parser.add_argument(
        "--save-intermediates",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save raw brown candidate masks and local-darkness masks.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete an existing output folder before writing.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of valid real image files to process.",
    )
    parser.add_argument(
        "--min-component-area",
        type=int,
        default=None,
        help="Override the preset minimum candidate component area in pixels.",
    )
    parser.add_argument(
        "--max-head-distance",
        type=float,
        default=70.0,
        help="Maximum endpoint-to-head distance, in pixels, before scale adjustment.",
    )
    parser.add_argument(
        "--head-contact-radius",
        type=float,
        default=8.0,
        help="Scaled radius used to decide whether a tail component touches a head.",
    )
    parser.add_argument(
        "--crop-padding",
        type=int,
        default=24,
        help="Scaled padding in pixels added around each individual sperm crop.",
    )
    return parser.parse_args()


def resolve_poc_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return SCRIPT_DIR / path


def is_real_image(path: Path) -> bool:
    if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
        return False
    if path.name.startswith("._"):
        return False
    return "__MACOSX" not in path.parts


def collect_images(input_root: Path, limit: int | None) -> list[Path]:
    images = sorted(path for path in input_root.rglob("*") if is_real_image(path))
    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit must be positive when provided")
        images = images[:limit]
    return images


def valid_image_count(input_root: Path) -> int:
    return sum(1 for path in input_root.rglob("*") if is_real_image(path))


def load_rgb(path: Path) -> np.ndarray:
    try:
        with Image.open(path) as img:
            return np.asarray(img.convert("RGB"))
    except (UnidentifiedImageError, OSError) as exc:
        raise RuntimeError(f"Could not read image: {path}") from exc


def mode_settings(mode: str, min_component_area: int | None) -> dict[str, float | int]:
    presets: dict[str, dict[str, float | int]] = {
        "high-recall": {
            "sat_min": 20,
            "value_max": 250,
            "local_dark_min": 4,
            "head_value_max": 130,
            "min_area": 12,
            "tail_min_area": 70,
            "tail_min_aspect": 2.0,
            "tail_max_fill": 0.55,
        },
        "balanced": {
            "sat_min": 28,
            "value_max": 245,
            "local_dark_min": 6,
            "head_value_max": 125,
            "min_area": 18,
            "tail_min_area": 100,
            "tail_min_aspect": 2.5,
            "tail_max_fill": 0.45,
        },
        "high-precision": {
            "sat_min": 38,
            "value_max": 235,
            "local_dark_min": 9,
            "head_value_max": 115,
            "min_area": 28,
            "tail_min_area": 150,
            "tail_min_aspect": 3.0,
            "tail_max_fill": 0.34,
        },
    }
    settings = presets[mode].copy()
    if min_component_area is not None:
        if min_component_area <= 0:
            raise ValueError("--min-component-area must be positive when provided")
        settings["min_area"] = min_component_area
    return settings


def scaled_kernel(size: int, scale: float) -> np.ndarray:
    scaled = max(3, int(round(size * scale)))
    if scaled % 2 == 0:
        scaled += 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (scaled, scaled))


def image_scale_factor(width: int, height: int) -> float:
    # Ward 100x images are 1600x1200. Larger 40x images scale thresholds gently.
    return max(1.0, max(width, height) / 1600.0)


def remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned = np.zeros(mask.shape, dtype=np.uint8)
    for label_idx in range(1, n_labels):
        area = int(stats[label_idx, cv2.CC_STAT_AREA])
        if area >= min_area:
            cleaned[labels == label_idx] = 255
    return cleaned


def build_candidate_masks(
    rgb: np.ndarray, settings: dict[str, float | int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = rgb.shape[:2]
    scale = image_scale_factor(width, height)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hue, sat, value = cv2.split(hsv)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    sigma = max(25.0, min(width, height) / 30.0)
    background = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma, sigmaY=sigma)
    local_dark = (background.astype(np.int16) - gray.astype(np.int16)).clip(0, 255)
    local_dark_mask = (local_dark >= int(settings["local_dark_min"])).astype(np.uint8) * 255

    # OpenCV hue is 0-179. Brown/olive stain typically lands from orange to yellow-green.
    hue_mask = ((hue >= 6) & (hue <= 82)).astype(np.uint8) * 255
    stain_mask = (
        (hue_mask > 0)
        & (sat >= int(settings["sat_min"]))
        & (value <= int(settings["value_max"]))
        & (local_dark_mask > 0)
    ).astype(np.uint8) * 255

    stain_mask = cv2.medianBlur(stain_mask, 3)
    stain_mask = cv2.morphologyEx(
        stain_mask, cv2.MORPH_CLOSE, scaled_kernel(3, scale), iterations=1
    )

    min_area = max(1, int(round(int(settings["min_area"]) * scale * scale)))
    cleaned = remove_small_components(stain_mask, min_area=min_area)
    return stain_mask, cleaned, local_dark_mask


def contour_metrics(component_mask: np.ndarray) -> tuple[float | None, float | None]:
    contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None
    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    perimeter = float(cv2.arcLength(contour, True))
    if perimeter <= 0:
        circularity = None
    else:
        circularity = float((4.0 * np.pi * area) / (perimeter * perimeter))
    return area, circularity


def detect_heads(
    cleaned_mask: np.ndarray,
    rgb: np.ndarray,
    settings: dict[str, float | int],
) -> tuple[np.ndarray, list[ComponentInfo]]:
    height, width = cleaned_mask.shape
    scale = image_scale_factor(width, height)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    _, sat, value = cv2.split(hsv)
    head_seed = (
        (cleaned_mask > 0)
        & (sat >= max(15, int(settings["sat_min"]) - 4))
        & (value <= int(settings["head_value_max"]))
    ).astype(np.uint8) * 255
    head_seed = cv2.morphologyEx(
        head_seed, cv2.MORPH_CLOSE, scaled_kernel(5, scale), iterations=1
    )
    head_seed = cv2.dilate(head_seed, scaled_kernel(3, scale), iterations=1)

    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        head_seed, connectivity=8
    )
    head_mask = np.zeros_like(cleaned_mask, dtype=np.uint8)
    heads: list[ComponentInfo] = []

    min_area = int(round(35 * scale * scale))
    max_area = int(round(7000 * scale * scale))
    for label_idx in range(1, n_labels):
        area = int(stats[label_idx, cv2.CC_STAT_AREA])
        if not (min_area <= area <= max_area):
            continue

        x = int(stats[label_idx, cv2.CC_STAT_LEFT])
        y = int(stats[label_idx, cv2.CC_STAT_TOP])
        w = int(stats[label_idx, cv2.CC_STAT_WIDTH])
        h = int(stats[label_idx, cv2.CC_STAT_HEIGHT])
        aspect = max(w, h) / max(1, min(w, h))
        fill = area / max(1, w * h)

        roi = np.zeros((h, w), dtype=np.uint8)
        roi[labels[y : y + h, x : x + w] == label_idx] = 255
        _, circularity = contour_metrics(roi)

        if aspect > 4.2 or fill < 0.14:
            continue
        if circularity is not None and circularity < 0.08:
            continue

        head_mask[labels == label_idx] = 255
        heads.append(
            ComponentInfo(
                component_id=len(heads) + 1,
                kind="head",
                area_px=area,
                bbox_xywh=[x, y, w, h],
                centroid_xy=[float(centroids[label_idx][0]), float(centroids[label_idx][1])],
                aspect_ratio=float(aspect),
                fill_ratio=float(fill),
                circularity=float(circularity) if circularity is not None else None,
            )
        )

    return head_mask, heads


def skeletonize_binary(mask: np.ndarray) -> np.ndarray:
    mask = (mask > 0).astype(np.uint8) * 255
    if not np.any(mask):
        return mask

    if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "thinning"):
        return cv2.ximgproc.thinning(mask)

    return zhang_suen_thinning(mask)


def zhang_suen_thinning(mask: np.ndarray) -> np.ndarray:
    img = (mask > 0).astype(np.uint8)
    if not np.any(img):
        return img * 255

    img = np.pad(img, 1, mode="constant")
    changed = True
    while changed:
        changed = False
        for step in (0, 1):
            p2 = img[:-2, 1:-1]
            p3 = img[:-2, 2:]
            p4 = img[1:-1, 2:]
            p5 = img[2:, 2:]
            p6 = img[2:, 1:-1]
            p7 = img[2:, :-2]
            p8 = img[1:-1, :-2]
            p9 = img[:-2, :-2]
            center = img[1:-1, 1:-1]

            neighbors = [p2, p3, p4, p5, p6, p7, p8, p9]
            neighbor_count = sum(neighbors)
            transitions = sum(
                (neighbors[idx] == 0) & (neighbors[(idx + 1) % 8] == 1)
                for idx in range(8)
            )

            if step == 0:
                cond_a = (p2 * p4 * p6) == 0
                cond_b = (p4 * p6 * p8) == 0
            else:
                cond_a = (p2 * p4 * p8) == 0
                cond_b = (p2 * p6 * p8) == 0

            remove = (
                (center == 1)
                & (neighbor_count >= 2)
                & (neighbor_count <= 6)
                & (transitions == 1)
                & cond_a
                & cond_b
            )
            if np.any(remove):
                center[remove] = 0
                changed = True

    return (img[1:-1, 1:-1] * 255).astype(np.uint8)


def prune_skeleton_endpoints(skeleton: np.ndarray, iterations: int) -> np.ndarray:
    pruned = (skeleton > 0).astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    for _ in range(max(0, iterations)):
        if not np.any(pruned):
            break
        neighbor_count = cv2.filter2D(pruned, cv2.CV_16S, kernel, borderType=cv2.BORDER_CONSTANT) - pruned
        endpoints = (pruned > 0) & (neighbor_count <= 1)
        if not np.any(endpoints):
            break
        pruned[endpoints] = 0
    return pruned.astype(np.uint8) * 255


def skeleton_points_by_degree(skeleton: np.ndarray, offset_x: int, offset_y: int) -> tuple[list[list[int]], list[list[int]]]:
    binary = (skeleton > 0).astype(np.uint8)
    if not np.any(binary):
        return [], []

    kernel = np.ones((3, 3), dtype=np.uint8)
    neighbor_count = cv2.filter2D(binary, cv2.CV_16S, kernel, borderType=cv2.BORDER_CONSTANT) - binary
    endpoints_yx = np.argwhere((binary > 0) & (neighbor_count == 1))
    branch_mask = ((binary > 0) & (neighbor_count >= 3)).astype(np.uint8) * 255

    endpoints = [[int(x + offset_x), int(y + offset_y)] for y, x in endpoints_yx]
    branchpoints: list[list[int]] = []
    n_labels, _, stats, centroids = cv2.connectedComponentsWithStats(branch_mask, connectivity=8)
    for label_idx in range(1, n_labels):
        if int(stats[label_idx, cv2.CC_STAT_AREA]) < 1:
            continue
        cx, cy = centroids[label_idx]
        branchpoints.append([int(round(cx + offset_x)), int(round(cy + offset_y))])
    return endpoints, branchpoints


def nearest_heads_for_tail(
    endpoints: list[list[int]],
    heads: list[ComponentInfo],
    max_distance: float,
) -> dict[str, Any]:
    if not endpoints or not heads:
        return {
            "proximal": None,
            "distal": None,
            "uncertain": True,
            "reason": "missing_endpoints_or_heads",
        }

    matches: list[dict[str, Any]] = []
    for endpoint in endpoints:
        ep = np.array(endpoint, dtype=np.float32)
        best: dict[str, Any] | None = None
        for head in heads:
            centroid = np.array(head.centroid_xy, dtype=np.float32)
            distance = float(np.linalg.norm(ep - centroid))
            if distance <= max_distance and (best is None or distance < best["distance_px"]):
                best = {
                    "endpoint_xy": endpoint,
                    "head_id": head.component_id,
                    "head_centroid_xy": head.centroid_xy,
                    "distance_px": distance,
                }
        if best is not None:
            matches.append(best)

    if not matches:
        return {
            "proximal": None,
            "distal": None,
            "uncertain": True,
            "reason": "no_head_within_distance",
        }

    matches.sort(key=lambda item: item["distance_px"])
    proximal = matches[0]
    distal = matches[1] if len(matches) > 1 else None
    return {
        "proximal": proximal,
        "distal": distal,
        "uncertain": distal is None or proximal["head_id"] == distal["head_id"],
        "reason": "matched_nearest_heads",
    }


def detect_tails(
    cleaned_mask: np.ndarray,
    head_mask: np.ndarray,
    heads: list[ComponentInfo],
    settings: dict[str, float | int],
    max_head_distance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[ComponentInfo]]:
    tail_seed = cleaned_mask.copy()
    tail_seed[head_mask > 0] = 0
    tail_seed = cv2.morphologyEx(
        tail_seed,
        cv2.MORPH_CLOSE,
        np.ones((3, 3), dtype=np.uint8),
        iterations=1,
    )
    tail_seed = remove_small_components(tail_seed, min_area=int(settings["min_area"]))

    height, width = tail_seed.shape
    scale = image_scale_factor(width, height)
    scaled_max_head_distance = max_head_distance * scale
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        tail_seed, connectivity=8
    )

    tail_mask = np.zeros_like(tail_seed, dtype=np.uint8)
    overlap_mask = np.zeros_like(tail_seed, dtype=np.uint8)
    tail_label_map = np.zeros_like(tail_seed, dtype=np.int32)
    tails: list[ComponentInfo] = []

    min_tail_area = int(round(int(settings["tail_min_area"]) * scale * scale))
    for label_idx in range(1, n_labels):
        area = int(stats[label_idx, cv2.CC_STAT_AREA])
        x = int(stats[label_idx, cv2.CC_STAT_LEFT])
        y = int(stats[label_idx, cv2.CC_STAT_TOP])
        w = int(stats[label_idx, cv2.CC_STAT_WIDTH])
        h = int(stats[label_idx, cv2.CC_STAT_HEIGHT])
        aspect = max(w, h) / max(1, min(w, h))
        fill = area / max(1, w * h)

        line_like = aspect >= float(settings["tail_min_aspect"]) and fill <= float(
            settings["tail_max_fill"]
        )
        sparse_branch_like = (
            area >= int(round(150 * scale * scale)) and aspect >= 1.2 and fill <= 0.22
        )
        large_irregular = area >= int(round(800 * scale * scale)) and fill <= 0.38
        if area < min_tail_area or not (line_like or sparse_branch_like or large_irregular):
            continue

        pad = 2
        y0 = max(0, y - pad)
        y1 = min(height, y + h + pad)
        x0 = max(0, x - pad)
        x1 = min(width, x + w + pad)
        roi_labels = labels[y0:y1, x0:x1]
        roi_mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
        roi_mask[roi_labels == label_idx] = 255

        skeleton = skeletonize_binary(roi_mask)
        pruned_skeleton = prune_skeleton_endpoints(skeleton, iterations=max(1, int(round(2 * scale))))
        skeleton_for_topology = pruned_skeleton if np.any(pruned_skeleton) else skeleton
        endpoints, branchpoints = skeleton_points_by_degree(skeleton_for_topology, x0, y0)

        distances = cv2.distanceTransform((roi_mask > 0).astype(np.uint8), cv2.DIST_L2, 5)
        widths = distances[skeleton > 0] * 2.0
        if widths.size:
            width_p95 = float(np.percentile(widths, 95))
            width_median = float(np.median(widths))
        else:
            width_p95 = 0.0
            width_median = 0.0

        width_spike_threshold = max(width_median * 2.8, width_median + 6.0 * scale)
        width_spike = width_median > 0 and width_p95 >= width_spike_threshold
        wide_branchpoints: list[list[int]] = []
        branch_width_threshold = max(width_median * 1.8, width_median + 3.0 * scale)
        for branch_x, branch_y in branchpoints:
            local_x = int(np.clip(branch_x - x0, 0, distances.shape[1] - 1))
            local_y = int(np.clip(branch_y - y0, 0, distances.shape[0] - 1))
            if float(distances[local_y, local_x] * 2.0) >= branch_width_threshold:
                wide_branchpoints.append([branch_x, branch_y])

        overlap_reasons: list[str] = []
        if wide_branchpoints:
            overlap_reasons.append("wide_branchpoints")
        if len(endpoints) > 2 and (wide_branchpoints or width_spike):
            overlap_reasons.append("multi_endpoint_wide_region")
        if width_spike:
            overlap_reasons.append("local_width_spike")
        if fill <= 0.22 and area >= int(round(150 * scale * scale)) and aspect < 2.0:
            overlap_reasons.append("wide_sparse_component")

        component_pixels = labels == label_idx
        component_id = len(tails) + 1
        tail_mask[component_pixels] = 255
        tail_label_map[component_pixels] = component_id
        if overlap_reasons:
            for branch_x, branch_y in wide_branchpoints:
                cv2.circle(
                    overlap_mask,
                    (branch_x, branch_y),
                    radius=max(4, int(round(5 * scale))),
                    color=255,
                    thickness=-1,
                )
            if width_spike:
                width_region = ((skeleton > 0) & ((distances * 2.0) >= width_spike_threshold)).astype(
                    np.uint8
                ) * 255
                width_region = cv2.dilate(
                    width_region,
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
                    iterations=1,
                )
                overlap_roi = overlap_mask[y0:y1, x0:x1]
                overlap_roi[width_region > 0] = 255
            if "wide_sparse_component" in overlap_reasons and not branchpoints and not width_spike:
                overlap_mask[component_pixels] = 255

        matches = nearest_heads_for_tail(endpoints, heads, max_distance=scaled_max_head_distance)
        tails.append(
            ComponentInfo(
                component_id=component_id,
                kind="tail",
                area_px=area,
                bbox_xywh=[x, y, w, h],
                centroid_xy=[float(centroids[label_idx][0]), float(centroids[label_idx][1])],
                aspect_ratio=float(aspect),
                fill_ratio=float(fill),
                circularity=None,
                skeleton_pixels=int(cv2.countNonZero(skeleton)),
                endpoints_xy=endpoints,
                branchpoints_xy=branchpoints,
                width_p95_px=width_p95,
                width_median_px=width_median,
                has_overlap=bool(overlap_reasons),
                overlap_reasons=overlap_reasons,
                head_matches=matches,
            )
        )

    return tail_mask, overlap_mask, tail_label_map, tails


def overlay_masks(rgb: np.ndarray, tail_mask: np.ndarray, head_mask: np.ndarray, overlap_mask: np.ndarray) -> np.ndarray:
    overlay = rgb.copy()
    colors = [
        (tail_mask > 0, np.array([0, 210, 255], dtype=np.float32), 0.60),
        (head_mask > 0, np.array([255, 0, 180], dtype=np.float32), 0.68),
        (overlap_mask > 0, np.array([255, 235, 0], dtype=np.float32), 0.68),
    ]
    for mask, color, alpha in colors:
        if np.any(mask):
            overlay[mask] = ((1.0 - alpha) * overlay[mask].astype(np.float32) + alpha * color).astype(
                np.uint8
            )
    return overlay


def safe_stem(path: Path, input_root: Path) -> str:
    rel = path.relative_to(input_root)
    parts = [part.replace(" ", "_").replace(".", "_") for part in rel.parts]
    return "__".join(parts)


def write_mask(path: Path, mask: np.ndarray) -> None:
    Image.fromarray((mask > 0).astype(np.uint8) * 255).save(path)


def bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask > 0)
    if xs.size == 0 or ys.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def pad_bbox(
    bbox: tuple[int, int, int, int],
    width: int,
    height: int,
    padding: int,
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = bbox
    return (
        max(0, x0 - padding),
        max(0, y0 - padding),
        min(width, x1 + padding),
        min(height, y1 + padding),
    )


def bbox_from_mask_with_padding(
    mask: np.ndarray,
    width: int,
    height: int,
    padding: int,
) -> tuple[int, int, int, int] | None:
    bbox = bbox_from_mask(mask)
    if bbox is None:
        return None
    return pad_bbox(bbox, width, height, padding)


def point_to_bbox_edge_distance(point_xy: list[int] | tuple[int, int] | None, bbox: tuple[int, int, int, int]) -> float | None:
    if point_xy is None:
        return None
    x, y = int(point_xy[0]), int(point_xy[1])
    x0, y0, x1, y1 = bbox
    return float(min(abs(x - x0), abs(y - y0), abs((x1 - 1) - x), abs((y1 - 1) - y)))


def crop_boundary_risk_from_distance(distance_px: float | None, scale: float) -> str:
    if distance_px is None:
        return "unknown"
    if distance_px <= max(4.0 * scale, 6.0):
        return "high"
    if distance_px <= max(10.0 * scale, 12.0):
        return "medium"
    return "low"


def skeleton_graph(skeleton: np.ndarray) -> tuple[np.ndarray, list[list[int]]]:
    binary = (skeleton > 0).astype(np.uint8)
    coords_yx = np.argwhere(binary > 0)
    coord_to_idx = {tuple(coord): idx for idx, coord in enumerate(coords_yx)}
    neighbors: list[list[int]] = [[] for _ in range(len(coords_yx))]
    for idx, (y, x) in enumerate(coords_yx):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                neighbor_idx = coord_to_idx.get((y + dy, x + dx))
                if neighbor_idx is not None:
                    neighbors[idx].append(neighbor_idx)
    return coords_yx, neighbors


def nearest_skeleton_anchor(
    coords_yx: np.ndarray,
    head_roi: np.ndarray,
) -> tuple[int | None, float | None, list[int] | None]:
    if coords_yx.size == 0 or not np.any(head_roi):
        return None, None, None

    # Distance from every ROI pixel to the candidate head. Sampling that at the
    # skeleton chooses the closest plausible neck/root point for graph growth.
    distances = cv2.distanceTransform((head_roi == 0).astype(np.uint8), cv2.DIST_L2, 5)
    skeleton_distances = distances[coords_yx[:, 0], coords_yx[:, 1]]
    if skeleton_distances.size == 0:
        return None, None, None

    anchor_idx = int(np.argmin(skeleton_distances))
    y, x = coords_yx[anchor_idx]
    return anchor_idx, float(skeleton_distances[anchor_idx]), [int(x), int(y)]


def multi_source_skeleton_owners(
    coords_yx: np.ndarray,
    neighbors: list[list[int]],
    anchors: dict[int, int],
    blocked_indices: set[int] | None = None,
) -> np.ndarray:
    owners = np.zeros(len(coords_yx), dtype=np.int32)
    distances = np.full(len(coords_yx), np.inf, dtype=np.float64)
    heap: list[tuple[float, int, int]] = []
    blocked = set(blocked_indices or set()) - {int(idx) for idx in anchors.values()}

    for head_id, anchor_idx in anchors.items():
        owners[anchor_idx] = int(head_id)
        distances[anchor_idx] = 0.0
        heapq.heappush(heap, (0.0, int(head_id), int(anchor_idx)))

    while heap:
        distance, head_id, current_idx = heapq.heappop(heap)
        if distance != distances[current_idx] or head_id != owners[current_idx]:
            continue

        y, x = coords_yx[current_idx]
        for neighbor_idx in neighbors[current_idx]:
            if neighbor_idx in blocked:
                continue
            ny, nx = coords_yx[neighbor_idx]
            step = 1.41421356237 if ny != y and nx != x else 1.0
            new_distance = distance + step
            if new_distance + 1e-6 < distances[neighbor_idx]:
                distances[neighbor_idx] = new_distance
                owners[neighbor_idx] = head_id
                heapq.heappush(heap, (new_distance, head_id, neighbor_idx))

    return owners


def flood_tail_assignments(
    tail_roi: np.ndarray,
    coords_yx: np.ndarray,
    skeleton_owners: np.ndarray,
    barrier_mask: np.ndarray | None = None,
) -> np.ndarray:
    tail_binary = tail_roi > 0
    if barrier_mask is not None:
        tail_binary = tail_binary & ~(barrier_mask > 0)
    assignment = np.zeros(tail_binary.shape, dtype=np.int32)
    queue: deque[tuple[int, int, int]] = deque()

    for idx, (y, x) in enumerate(coords_yx):
        head_id = int(skeleton_owners[idx])
        if head_id <= 0 or not tail_binary[y, x]:
            continue
        assignment[y, x] = head_id
        queue.append((int(y), int(x), head_id))

    # Expand skeleton ownership through the full-width tail mask. This keeps the
    # crop mask useful for visual review instead of saving only a one-pixel line.
    while queue:
        y, x, head_id = queue.popleft()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                yy = y + dy
                xx = x + dx
                if (
                    0 <= yy < assignment.shape[0]
                    and 0 <= xx < assignment.shape[1]
                    and tail_binary[yy, xx]
                    and assignment[yy, xx] == 0
                ):
                    assignment[yy, xx] = head_id
                    queue.append((yy, xx, head_id))

    return assignment


def shortest_path_tree(
    coords_yx: np.ndarray,
    neighbors: list[list[int]],
    start_idx: int,
) -> tuple[list[int], np.ndarray]:
    parents = [-1] * len(coords_yx)
    distances = np.full(len(coords_yx), np.inf, dtype=np.float64)
    distances[start_idx] = 0.0
    heap: list[tuple[float, int]] = [(0.0, start_idx)]

    while heap:
        distance, current_idx = heapq.heappop(heap)
        if distance != distances[current_idx]:
            continue

        y, x = coords_yx[current_idx]
        for neighbor_idx in neighbors[current_idx]:
            ny, nx = coords_yx[neighbor_idx]
            step = 1.41421356237 if ny != y and nx != x else 1.0
            new_distance = distance + step
            if new_distance + 1e-6 < distances[neighbor_idx]:
                distances[neighbor_idx] = new_distance
                parents[neighbor_idx] = current_idx
                heapq.heappush(heap, (new_distance, neighbor_idx))

    return parents, distances


def reconstruct_path(parents: list[int], start_idx: int, end_idx: int) -> list[int]:
    path: list[int] = []
    current = end_idx
    while current != -1:
        path.append(current)
        if current == start_idx:
            break
        current = parents[current]
    if not path or path[-1] != start_idx:
        return []
    path.reverse()
    return path


def measure_perpendicular_width(
    tail_roi: np.ndarray,
    y: int,
    x: int,
    tangent_dy: float,
    tangent_dx: float,
    max_radius: int,
) -> int:
    norm = float(np.hypot(tangent_dy, tangent_dx))
    if norm <= 0:
        return 0

    perp_y = -tangent_dx / norm
    perp_x = tangent_dy / norm
    height, width = tail_roi.shape
    sampled: set[tuple[int, int]] = {(int(y), int(x))}

    for sign in (-1.0, 1.0):
        for step in range(1, max_radius + 1):
            yy = int(round(y + sign * perp_y * step))
            xx = int(round(x + sign * perp_x * step))
            if yy < 0 or yy >= height or xx < 0 or xx >= width or tail_roi[yy, xx] == 0:
                break
            sampled.add((yy, xx))

    return len(sampled)


def directional_width_spike_analysis(
    tail_roi: np.ndarray,
    coords_yx: np.ndarray,
    neighbors: list[list[int]],
    anchors: dict[int, int],
    scale: float,
    average_tail_width_px: float | None = None,
) -> dict[str, Any]:
    endpoint_indices = [idx for idx, items in enumerate(neighbors) if len(items) <= 1]
    empty_mask = np.zeros(tail_roi.shape, dtype=np.uint8)
    if not endpoint_indices or not anchors:
        return {
            "normal_width_px": average_tail_width_px,
            "threshold_width_px": None,
            "sampled_skeleton_points": 0,
            "spike_skeleton_points": 0,
            "barrier_pixels": 0,
            "barrier_indices": [],
            "barrier_mask": empty_mask,
        }

    width_by_idx: dict[int, list[int]] = {}
    lookaround = max(2, int(round(4 * scale)))
    edge_skip = max(3, int(round(5 * scale)))
    fallback_width = average_tail_width_px if average_tail_width_px and average_tail_width_px > 0 else 4.0 * scale
    max_radius = max(int(round(18 * scale)), int(round(fallback_width * 7.0)))

    for anchor_idx in anchors.values():
        parents, distances = shortest_path_tree(coords_yx, neighbors, int(anchor_idx))
        for endpoint_idx in endpoint_indices:
            if not np.isfinite(distances[endpoint_idx]):
                continue
            path = reconstruct_path(parents, int(anchor_idx), int(endpoint_idx))
            if len(path) <= (2 * edge_skip + 2):
                continue
            for path_pos in range(edge_skip, len(path) - edge_skip):
                current_idx = path[path_pos]
                prev_idx = path[max(0, path_pos - lookaround)]
                next_idx = path[min(len(path) - 1, path_pos + lookaround)]
                prev_y, prev_x = coords_yx[prev_idx]
                next_y, next_x = coords_yx[next_idx]
                tangent_dy = float(next_y - prev_y)
                tangent_dx = float(next_x - prev_x)
                if tangent_dy == 0.0 and tangent_dx == 0.0:
                    continue
                y, x = coords_yx[current_idx]
                measured_width = measure_perpendicular_width(
                    tail_roi,
                    int(y),
                    int(x),
                    tangent_dy,
                    tangent_dx,
                    max_radius=max_radius,
                )
                if measured_width > 0:
                    width_by_idx.setdefault(int(current_idx), []).append(measured_width)

    if not width_by_idx:
        return {
            "normal_width_px": average_tail_width_px,
            "threshold_width_px": None,
            "sampled_skeleton_points": 0,
            "spike_skeleton_points": 0,
            "barrier_pixels": 0,
            "barrier_indices": [],
            "barrier_mask": empty_mask,
        }

    per_point_widths = {
        idx: float(max(widths)) for idx, widths in width_by_idx.items() if widths
    }
    sampled_widths = np.array(list(per_point_widths.values()), dtype=np.float32)
    width_percentiles = {
        "p50": float(np.percentile(sampled_widths, 50)),
        "p75": float(np.percentile(sampled_widths, 75)),
        "p90": float(np.percentile(sampled_widths, 90)),
        "p95": float(np.percentile(sampled_widths, 95)),
    }
    if average_tail_width_px is not None and average_tail_width_px > 0:
        normal_width = max(float(average_tail_width_px), width_percentiles["p50"])
    else:
        # The median perpendicular profile is the local normal caliber for this
        # skeleton. Width spikes must rise well above this value to count.
        normal_width = width_percentiles["p50"]

    # The perpendicular sampler counts discrete mask pixels and is intentionally
    # more conservative than the distance-transform diameter used in detect_tails.
    # Requiring a larger jump avoids treating ordinary tail thickness/diagonal
    # aliasing as an intersection.
    threshold_width = max(normal_width * 2.5, normal_width + 8.0 * scale)
    spike_seed = np.zeros(tail_roi.shape, dtype=np.uint8)
    for idx, measured_width in per_point_widths.items():
        if measured_width >= threshold_width:
            y, x = coords_yx[idx]
            spike_seed[int(y), int(x)] = 255

    if not np.any(spike_seed):
        return {
            "normal_width_px": normal_width,
            "threshold_width_px": threshold_width,
            "width_percentiles": width_percentiles,
            "sampled_skeleton_points": len(per_point_widths),
            "spike_skeleton_points": 0,
            "barrier_pixels": 0,
            "barrier_indices": [],
            "barrier_mask": empty_mask,
        }

    min_cluster_pixels = max(2, int(round(2 * scale)))
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(spike_seed, connectivity=8)
    sustained_spike_mask = np.zeros(tail_roi.shape, dtype=np.uint8)
    for label_idx in range(1, n_labels):
        if int(stats[label_idx, cv2.CC_STAT_AREA]) >= min_cluster_pixels:
            sustained_spike_mask[labels == label_idx] = 255

    if not np.any(sustained_spike_mask):
        return {
            "normal_width_px": normal_width,
            "threshold_width_px": threshold_width,
            "width_percentiles": width_percentiles,
            "sampled_skeleton_points": len(per_point_widths),
            "spike_skeleton_points": 0,
            "barrier_pixels": 0,
            "barrier_indices": [],
            "barrier_mask": empty_mask,
        }

    coord_to_idx = {tuple(coord): idx for idx, coord in enumerate(coords_yx)}
    barrier_indices: list[int] = []
    for y, x in np.argwhere(sustained_spike_mask > 0):
        idx = coord_to_idx.get((int(y), int(x)))
        if idx is not None:
            barrier_indices.append(int(idx))

    barrier_radius = max(1, int(round(normal_width / 2.0)))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * barrier_radius + 1, 2 * barrier_radius + 1),
    )
    barrier_mask = cv2.dilate(sustained_spike_mask, kernel, iterations=1)
    barrier_mask[(tail_roi == 0)] = 0

    return {
        "normal_width_px": normal_width,
        "threshold_width_px": threshold_width,
        "width_percentiles": width_percentiles,
        "sampled_skeleton_points": len(per_point_widths),
        "spike_skeleton_points": len(barrier_indices),
        "barrier_pixels": int(np.count_nonzero(barrier_mask)),
        "barrier_indices": sorted(set(barrier_indices)),
        "barrier_mask": barrier_mask,
    }


def head_area_threshold_for_shared_tail(
    touching_head_ids: list[int],
    head_labels: np.ndarray,
    scale: float,
    head_areas: dict[int, int] | None = None,
) -> tuple[dict[int, int], int]:
    areas = {
        int(head_id): int(
            head_areas[head_id]
            if head_areas is not None and head_id in head_areas
            else np.count_nonzero(head_labels == head_id)
        )
        for head_id in touching_head_ids
    }
    if not areas:
        return areas, int(round(120 * scale * scale))

    p90_area = float(np.percentile(list(areas.values()), 90))
    threshold = max(int(round(120 * scale * scale)), int(round(0.25 * p90_area)))
    return areas, threshold


def estimate_average_tail_width_px(tails: list[ComponentInfo], scale: float) -> float:
    high_confidence_widths = [
        float(tail.width_median_px)
        for tail in tails
        if tail.width_median_px is not None
        and tail.width_median_px > 0
        and not tail.has_overlap
        and len(tail.endpoints_xy or []) <= 2
        and len(tail.branchpoints_xy or []) <= 2
    ]
    if high_confidence_widths:
        return float(np.median(high_confidence_widths))

    fallback_widths = [
        float(tail.width_median_px)
        for tail in tails
        if tail.width_median_px is not None and tail.width_median_px > 0
    ]
    if fallback_widths:
        return float(np.median(fallback_widths))

    return max(3.0, 4.0 * scale)


def full_size_roi_mask(
    roi_mask: np.ndarray,
    shape: tuple[int, int],
    x0: int,
    y0: int,
) -> np.ndarray:
    full_mask = np.zeros(shape, dtype=bool)
    full_mask[y0 : y0 + roi_mask.shape[0], x0 : x0 + roi_mask.shape[1]] = roi_mask > 0
    return full_mask


def graph_path_length_px(coords_yx: np.ndarray, path_indices: list[int]) -> float:
    if len(path_indices) < 2:
        return 0.0
    points = coords_yx[path_indices].astype(np.float32)
    deltas = np.diff(points, axis=0)
    return float(np.sum(np.hypot(deltas[:, 0], deltas[:, 1])))


def graph_path_curvature(coords_yx: np.ndarray, path_indices: list[int]) -> float:
    if len(path_indices) < 5:
        return 0.0

    points = coords_yx[path_indices].astype(np.float32)
    total = 0.0
    for idx in range(2, len(points) - 2):
        before = points[idx] - points[idx - 2]
        after = points[idx + 2] - points[idx]
        before_norm = float(np.linalg.norm(before))
        after_norm = float(np.linalg.norm(after))
        if before_norm <= 0.0 or after_norm <= 0.0:
            continue
        cos_angle = float(np.dot(before, after) / (before_norm * after_norm))
        cos_angle = max(-1.0, min(1.0, cos_angle))
        total += float(np.arccos(cos_angle))
    return total


def make_head_collar_mask(
    head_roi: np.ndarray,
    scale: float,
    contact_radius: int,
    average_tail_width_px: float | None,
) -> np.ndarray:
    base_radius = max(3.0 * scale, 0.75 * float(contact_radius))
    if average_tail_width_px is not None and average_tail_width_px > 0:
        base_radius = max(base_radius, 1.6 * float(average_tail_width_px))
    radius = max(2, int(round(base_radius)))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * radius + 1, 2 * radius + 1),
    )
    return cv2.dilate((head_roi > 0).astype(np.uint8) * 255, kernel, iterations=1)


def summarize_path_exit(
    coords_yx: np.ndarray,
    path_indices: list[int],
    head_collar_mask: np.ndarray,
) -> dict[str, Any]:
    if not path_indices:
        return {
            "head_exit_count": 0,
            "head_exit_path_length_px": 0.0,
            "head_exit_xy": None,
            "path_skeleton_inside_head_collar": 0,
            "path_skeleton_outside_head_collar": 0,
        }

    inside_flags = [
        bool(head_collar_mask[int(coords_yx[idx][0]), int(coords_yx[idx][1])] > 0)
        for idx in path_indices
    ]
    outside_positions = [idx for idx, inside in enumerate(inside_flags) if not inside]
    exit_count = sum(
        1
        for prev_inside, next_inside in zip(inside_flags, inside_flags[1:])
        if prev_inside and not next_inside
    )
    if outside_positions:
        first_outside = outside_positions[0]
        exit_path_length = graph_path_length_px(coords_yx, path_indices[first_outside:])
        exit_y, exit_x = coords_yx[path_indices[first_outside]]
        head_exit_xy = [int(exit_x), int(exit_y)]
    else:
        exit_path_length = 0.0
        head_exit_xy = None

    return {
        "head_exit_count": int(exit_count),
        "head_exit_path_length_px": float(exit_path_length),
        "head_exit_xy": head_exit_xy,
        "path_skeleton_inside_head_collar": int(sum(inside_flags)),
        "path_skeleton_outside_head_collar": int(len(inside_flags) - sum(inside_flags)),
    }


def normalize_rgb_for_color_scoring(
    rgb_roi: np.ndarray | None,
    mode: str,
    scale: float,
) -> np.ndarray | None:
    if rgb_roi is None:
        return None
    if mode == "raw":
        return rgb_roi

    rgb_float = rgb_roi.astype(np.float32)
    if mode == "white-balance":
        channel_medians = np.median(rgb_float.reshape(-1, 3), axis=0)
        target = float(np.mean(channel_medians))
        gains = target / np.maximum(channel_medians, 1.0)
        return np.clip(rgb_float * gains, 0, 255).astype(np.uint8)

    if mode == "local-background":
        sigma = max(8.0 * scale, min(rgb_roi.shape[:2]) / 8.0)
        background = cv2.GaussianBlur(rgb_float, (0, 0), sigmaX=sigma, sigmaY=sigma)
        background_center = np.median(background.reshape(-1, 3), axis=0)
        normalized = rgb_float - background + background_center
        return np.clip(normalized, 0, 255).astype(np.uint8)

    raise ValueError(f"Unknown color normalization mode: {mode}")


def make_color_scoring_context(
    rgb_roi: np.ndarray | None,
    normalization: str,
    scale: float,
) -> dict[str, Any]:
    if rgb_roi is None:
        return {
            "raw_lab": None,
            "normalized_lab": None,
            "normalization": normalization,
        }

    raw_lab = cv2.cvtColor(rgb_roi, cv2.COLOR_RGB2LAB).astype(np.float32)
    if normalization == "raw":
        normalized_lab = raw_lab
    else:
        normalized_rgb = normalize_rgb_for_color_scoring(rgb_roi, normalization, scale)
        normalized_lab = (
            None
            if normalized_rgb is None
            else cv2.cvtColor(normalized_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        )
    return {
        "raw_lab": raw_lab,
        "normalized_lab": normalized_lab,
        "normalization": normalization,
    }


def _path_color_continuity_for_lab(
    lab_roi: np.ndarray | None,
    coords_yx: np.ndarray,
    path_indices: list[int],
    head_collar_mask: np.ndarray,
    scale: float,
) -> dict[str, Any]:
    if lab_roi is None or len(path_indices) < 3:
        return {
            "score": None,
            "delta_mean": None,
            "delta_p75": None,
            "reference_points": 0,
        }

    outside_indices = [
        idx
        for idx in path_indices
        if head_collar_mask[int(coords_yx[idx][0]), int(coords_yx[idx][1])] == 0
    ]
    if len(outside_indices) < 3:
        return {
            "score": 0.0,
            "delta_mean": None,
            "delta_p75": None,
            "reference_points": 0,
        }

    reference_count = max(3, min(len(outside_indices), int(round(10 * scale))))
    reference_points = outside_indices[:reference_count]
    reference_samples = np.array(
        [lab_roi[int(coords_yx[idx][0]), int(coords_yx[idx][1])] for idx in reference_points],
        dtype=np.float32,
    )
    reference_color = np.median(reference_samples, axis=0)
    samples = np.array(
        [lab_roi[int(coords_yx[idx][0]), int(coords_yx[idx][1])] for idx in outside_indices],
        dtype=np.float32,
    )
    deltas = np.linalg.norm(samples - reference_color, axis=1)
    mean_delta = float(np.mean(deltas)) if deltas.size else 0.0
    p75_delta = float(np.percentile(deltas, 75)) if deltas.size else 0.0
    continuity = max(0.0, min(1.0, 1.0 - (p75_delta / 55.0)))
    return {
        "score": float(continuity),
        "delta_mean": mean_delta,
        "delta_p75": p75_delta,
        "reference_points": int(reference_count),
    }


def path_color_continuity(
    rgb_roi: np.ndarray | None,
    coords_yx: np.ndarray,
    path_indices: list[int],
    head_collar_mask: np.ndarray,
    scale: float,
    normalization: str = "raw",
    color_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if color_context is None:
        color_context = make_color_scoring_context(rgb_roi, normalization, scale)

    raw = _path_color_continuity_for_lab(
        lab_roi=color_context.get("raw_lab"),
        coords_yx=coords_yx,
        path_indices=path_indices,
        head_collar_mask=head_collar_mask,
        scale=scale,
    )
    normalized = (
        raw
        if normalization == "raw"
        else _path_color_continuity_for_lab(
            lab_roi=color_context.get("normalized_lab"),
            coords_yx=coords_yx,
            path_indices=path_indices,
            head_collar_mask=head_collar_mask,
            scale=scale,
        )
    )
    selected = normalized if normalization != "raw" else raw
    return {
        "path_color_continuity_score": selected["score"],
        "path_color_delta_mean": selected["delta_mean"],
        "path_color_delta_p75": selected["delta_p75"],
        "path_color_reference_points": selected["reference_points"],
        "color_score_raw": raw["score"],
        "color_score_normalized": normalized["score"],
        "color_delta_p75_raw": raw["delta_p75"],
        "color_delta_p75_normalized": normalized["delta_p75"],
        "path_color_normalization": normalization,
    }


def empty_color_continuity(normalization: str) -> dict[str, Any]:
    return {
        "path_color_continuity_score": None,
        "path_color_delta_mean": None,
        "path_color_delta_p75": None,
        "path_color_reference_points": 0,
        "color_score_raw": None,
        "color_score_normalized": None,
        "color_delta_p75_raw": None,
        "color_delta_p75_normalized": None,
        "path_color_normalization": normalization,
    }


def reconstruct_path_tail_mask(
    tail_roi: np.ndarray,
    coords_yx: np.ndarray,
    path_indices: list[int],
    scale: float,
    average_tail_width_px: float | None,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    path_skeleton = np.zeros(tail_roi.shape, dtype=np.uint8)
    for idx in path_indices:
        y, x = coords_yx[idx]
        path_skeleton[int(y), int(x)] = 255

    distances = cv2.distanceTransform((tail_roi > 0).astype(np.uint8), cv2.DIST_L2, 5)
    path_widths = distances[path_skeleton > 0] * 2.0
    if path_widths.size:
        median_width = float(np.median(path_widths))
        p95_width = float(np.percentile(path_widths, 95))
    else:
        median_width = float(average_tail_width_px or max(3.0, 4.0 * scale))
        p95_width = median_width
    if average_tail_width_px is not None and average_tail_width_px > 0:
        normal_width = max(float(average_tail_width_px), median_width)
    else:
        normal_width = max(3.0 * scale, median_width)

    radius = max(1, int(round(normal_width * 0.7)))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * radius + 1, 2 * radius + 1),
    )
    path_mask = cv2.dilate(path_skeleton, kernel, iterations=1)
    path_mask[(tail_roi == 0)] = 0
    width_spike_ratio = float(p95_width / max(normal_width, 1.0))
    return path_mask, path_skeleton, normal_width, width_spike_ratio


def score_head_tail_paths(
    tail_roi: np.ndarray,
    coords_yx: np.ndarray,
    neighbors: list[list[int]],
    anchor_idx: int,
    head_roi: np.ndarray,
    scale: float,
    contact_radius: int,
    average_tail_width_px: float | None,
    rgb_roi: np.ndarray | None = None,
    path_mode: str = "hybrid",
    normalization: str = "raw",
    color_context: dict[str, Any] | None = None,
    record_ablation_scores: bool = False,
) -> dict[str, Any]:
    endpoint_indices = [idx for idx, items in enumerate(neighbors) if len(items) <= 1]
    if not endpoint_indices:
        _, distances = shortest_path_tree(coords_yx, neighbors, int(anchor_idx))
        finite_indices = [idx for idx, distance in enumerate(distances) if np.isfinite(distance)]
        endpoint_indices = sorted(finite_indices, key=lambda idx: distances[idx], reverse=True)[:6]

    parents, distances = shortest_path_tree(coords_yx, neighbors, int(anchor_idx))
    max_endpoint_candidates = 40
    if len(endpoint_indices) > max_endpoint_candidates:
        endpoint_indices = sorted(
            endpoint_indices,
            key=lambda idx: distances[idx] if np.isfinite(distances[idx]) else -1.0,
            reverse=True,
        )[:max_endpoint_candidates]
    head_collar_mask = make_head_collar_mask(
        head_roi=head_roi,
        scale=scale,
        contact_radius=contact_radius,
        average_tail_width_px=average_tail_width_px,
    )
    min_exit_path_length = max(12.0 * scale, 2.2 * float(average_tail_width_px or 4.0 * scale))
    min_total_path_length = max(18.0 * scale, 4.0 * float(average_tail_width_px or 4.0 * scale))

    candidates: list[dict[str, Any]] = []
    for endpoint_idx in endpoint_indices:
        if int(endpoint_idx) == int(anchor_idx) or not np.isfinite(distances[endpoint_idx]):
            continue
        path_indices = reconstruct_path(parents, int(anchor_idx), int(endpoint_idx))
        if len(path_indices) < 2:
            continue

        path_length = graph_path_length_px(coords_yx, path_indices)
        path_mask, path_skeleton, normal_width, width_spike_ratio = reconstruct_path_tail_mask(
            tail_roi=tail_roi,
            coords_yx=coords_yx,
            path_indices=path_indices,
            scale=scale,
            average_tail_width_px=average_tail_width_px,
        )
        path_component_trim_fraction = 1.0 - (
            float(np.count_nonzero(path_mask)) / float(max(1, np.count_nonzero(tail_roi)))
        )
        exit_summary = summarize_path_exit(coords_yx, path_indices, head_collar_mask)
        tail_pixels_inside = int(np.count_nonzero((path_mask > 0) & (head_collar_mask > 0)))
        tail_pixels_outside = int(np.count_nonzero((path_mask > 0) & (head_collar_mask == 0)))
        outside_fraction = (
            float(tail_pixels_outside) / float(max(1, tail_pixels_inside + tail_pixels_outside))
        )
        needs_color_score = path_mode in {"color-only", "hybrid"} or record_ablation_scores
        color_summary = (
            path_color_continuity(
                rgb_roi=rgb_roi,
                coords_yx=coords_yx,
                path_indices=path_indices,
                head_collar_mask=head_collar_mask,
                scale=scale,
                normalization=normalization,
                color_context=color_context,
            )
            if needs_color_score
            else empty_color_continuity(normalization)
        )
        color_score = color_summary["path_color_continuity_score"]
        color_component = 0.72 if color_score is None else float(color_score)

        curvature_total = graph_path_curvature(coords_yx, path_indices)
        branchpoint_count = int(sum(1 for idx in path_indices if len(neighbors[idx]) >= 3))
        length_score = min(1.0, path_length / max(min_total_path_length * 2.5, 1.0))
        exit_score = min(1.0, float(exit_summary["head_exit_path_length_px"]) / max(min_exit_path_length * 2.0, 1.0))
        curvature_penalty = min(0.22, curvature_total / (8.0 * np.pi))
        branch_penalty = min(0.24, 0.035 * branchpoint_count)
        width_penalty = min(0.18, max(0.0, width_spike_ratio - 2.1) * 0.12)
        simplicity_score = max(0.0, 1.0 - curvature_penalty - branch_penalty - width_penalty)
        geometry_score = (
            0.46 * length_score
            + 0.34 * exit_score
            + 0.20 * simplicity_score
        )
        color_only_score = (
            0.70 * color_component
            + 0.15 * length_score
            + 0.10 * exit_score
            + 0.05 * simplicity_score
            if needs_color_score
            else None
        )
        hybrid_score = (
            0.24 * length_score
            + 0.18 * exit_score
            + 0.42 * color_component
            + 0.16 * simplicity_score
            if needs_color_score
            else None
        )
        score_by_mode = {
            "geometry-only": geometry_score,
            "color-only": color_only_score,
            "hybrid": hybrid_score,
        }
        selected_score = score_by_mode.get(path_mode)
        score = geometry_score if selected_score is None else selected_score

        valid_tail_continuation = (
            path_length >= min_total_path_length
            and float(exit_summary["head_exit_path_length_px"]) >= min_exit_path_length
            and tail_pixels_outside >= max(12, int(round(18 * scale * scale)))
            and outside_fraction >= 0.25
        )
        path_rejection_reason: str | None = None
        if not valid_tail_continuation:
            if outside_fraction < 0.25 or int(exit_summary["head_exit_count"]) == 0:
                path_rejection_reason = "head_outline_false_tail"
            else:
                path_rejection_reason = "tail_path_too_short"
            score -= 0.35
            geometry_score -= 0.35
            if color_only_score is not None:
                color_only_score -= 0.35
            if hybrid_score is not None:
                hybrid_score -= 0.35

        endpoint_y, endpoint_x = coords_yx[endpoint_idx]
        candidates.append(
            {
                "endpoint_idx": int(endpoint_idx),
                "endpoint_xy": [int(endpoint_x), int(endpoint_y)],
                "path_indices": path_indices,
                "path_length_px": float(path_length),
                "path_score": float(max(0.0, min(1.0, score))),
                "path_score_geometry": float(max(0.0, min(1.0, geometry_score))),
                "path_score_color": (
                    None
                    if color_only_score is None
                    else float(max(0.0, min(1.0, color_only_score)))
                ),
                "path_score_hybrid": (
                    None
                    if hybrid_score is None
                    else float(max(0.0, min(1.0, hybrid_score)))
                ),
                "path_mode": path_mode,
                "path_mask": path_mask,
                "path_skeleton_mask": path_skeleton,
                "head_collar_mask": head_collar_mask,
                "normal_width_px": float(normal_width),
                "path_width_spike_ratio": float(width_spike_ratio),
                "path_component_trim_fraction": float(max(0.0, path_component_trim_fraction)),
                "path_curvature_total": float(curvature_total),
                "path_branchpoint_count": branchpoint_count,
                "tail_pixels_inside_head_collar": tail_pixels_inside,
                "tail_pixels_outside_head_collar": tail_pixels_outside,
                "valid_tail_continuation": bool(valid_tail_continuation),
                "path_rejection_reason": path_rejection_reason,
                **exit_summary,
                **color_summary,
            }
        )

    if not candidates:
        return {
            "best": None,
            "candidates": [],
            "reason": "missing_endpoint_paths",
        }

    valid_candidates = [row for row in candidates if row["valid_tail_continuation"]]
    ranked = sorted(
        valid_candidates if valid_candidates else candidates,
        key=lambda row: row["path_score"],
        reverse=True,
    )
    best = dict(ranked[0])
    second_score = float(ranked[1]["path_score"]) if len(ranked) > 1 else None
    score_margin = 1.0 if second_score is None else float(best["path_score"] - second_score)
    low_margin = bool(second_score is not None and score_margin < 0.10)
    best["path_score_margin"] = score_margin
    best["path_low_margin"] = low_margin
    best["competing_path_count"] = max(0, len(ranked) - 1)
    best["competing_path_skeleton_mask"] = np.zeros(tail_roi.shape, dtype=np.uint8)
    for competitor in ranked[1:]:
        best["competing_path_skeleton_mask"][competitor["path_skeleton_mask"] > 0] = 255

    mode_fields = {"geometry": "path_score_geometry"}
    if path_mode == "color-only" or record_ablation_scores:
        mode_fields["color"] = "path_score_color"
    if path_mode == "hybrid" or record_ablation_scores:
        mode_fields["hybrid"] = "path_score_hybrid"
    ranking_pool = valid_candidates if valid_candidates else candidates
    for label, field in mode_fields.items():
        mode_candidates = [row for row in ranking_pool if row.get(field) is not None]
        if not mode_candidates:
            best[f"path_rank_{label}"] = None
            best[f"path_score_margin_{label}"] = None
            continue
        mode_ranked = sorted(mode_candidates, key=lambda row: row[field], reverse=True)
        selected_rank = next(
            (
                idx + 1
                for idx, row in enumerate(mode_ranked)
                if int(row["endpoint_idx"]) == int(best["endpoint_idx"])
            ),
            None,
        )
        second_for_selected = next(
            (
                row
                for row in mode_ranked
                if int(row["endpoint_idx"]) != int(best["endpoint_idx"])
            ),
            None,
        )
        selected_score = float(best[field]) if best.get(field) is not None else None
        competitor_score = float(second_for_selected[field]) if second_for_selected else None
        best[f"path_rank_{label}"] = selected_rank
        best[f"path_score_margin_{label}"] = (
            None
            if selected_score is None
            else 1.0
            if competitor_score is None
            else selected_score - competitor_score
        )

    for label in ("geometry", "color", "hybrid"):
        best.setdefault(f"path_rank_{label}", None)
        best.setdefault(f"path_score_margin_{label}", None)

    path_ambiguity_reason: str | None = None
    if best["valid_tail_continuation"]:
        color_score = best.get("path_color_continuity_score")
        has_branch_competition = (
            int(best.get("path_branchpoint_count") or 0) > 0
            or len(endpoint_indices) > 3
        )
        clean_crossing_like_competition = (
            low_margin
            and has_branch_competition
            and int(best.get("path_branchpoint_count") or 0) <= 20
            and float(best.get("path_component_trim_fraction") or 0.0) >= 0.25
        )
        if path_mode != "geometry-only" and color_score is not None and float(color_score) < 0.35:
            path_ambiguity_reason = "low_color_continuity"
        elif path_mode != "geometry-only" and clean_crossing_like_competition:
            path_ambiguity_reason = "ambiguous_path_color_margin"
    best["path_ambiguity_reason"] = path_ambiguity_reason
    return {
        "best": best,
        "candidates": candidates,
        "reason": "scored_endpoint_paths",
    }


def mark_path_conflicts(assignments: list[dict[str, Any]]) -> None:
    if len(assignments) < 2:
        return

    for idx, assignment in enumerate(assignments):
        path_mask = assignment.get("path_skeleton_mask")
        if path_mask is None:
            continue
        path_pixels = int(np.count_nonzero(path_mask))
        if path_pixels <= 0:
            continue
        for other in assignments[idx + 1 :]:
            other_path_mask = other.get("path_skeleton_mask")
            if other_path_mask is None:
                continue
            other_pixels = int(np.count_nonzero(other_path_mask))
            if other_pixels <= 0:
                continue
            overlap = int(np.count_nonzero((path_mask > 0) & (other_path_mask > 0)))
            conflict_fraction = float(overlap) / float(max(1, min(path_pixels, other_pixels)))
            if conflict_fraction >= 0.35:
                assignment["path_ambiguity_reason"] = "ambiguous_path_conflict"
                other["path_ambiguity_reason"] = "ambiguous_path_conflict"
                assignment["path_conflict_fraction"] = max(
                    float(assignment.get("path_conflict_fraction") or 0.0),
                    conflict_fraction,
                )
                other["path_conflict_fraction"] = max(
                    float(other.get("path_conflict_fraction") or 0.0),
                    conflict_fraction,
                )


def score_to_percent(value: object) -> float | None:
    if value is None:
        return None
    return round(100.0 * float(value), 3)


def build_path_scored_tail_assignment_plan(
    tail_component: np.ndarray,
    head_labels: np.ndarray,
    touching_head_ids: list[int],
    scale: float,
    contact_radius: int,
    head_areas: dict[int, int] | None = None,
    average_tail_width_px: float | None = None,
    rgb: np.ndarray | None = None,
    path_mode: str = "hybrid",
    normalization: str = "raw",
    record_ablation_scores: bool = False,
) -> dict[str, Any]:
    touching_head_ids = sorted({int(head_id) for head_id in touching_head_ids})
    shared_tail_pixels = int(np.count_nonzero(tail_component))
    base_plan: dict[str, Any] = {
        "mode": "unassigned",
        "assignments": [],
        "rejected_heads": [],
        "ambiguous": None,
        "touching_head_ids": touching_head_ids,
        "touching_head_count": len(touching_head_ids),
        "shared_tail_pixels": shared_tail_pixels,
        "path_v2": True,
        "path_mode": path_mode,
        "normalization": normalization,
    }

    if not touching_head_ids or shared_tail_pixels == 0:
        base_plan["mode"] = "ambiguous"
        base_plan["ambiguous"] = {
            "reason": "missing_tail_or_heads",
            "touching_head_ids": touching_head_ids,
            "shared_tail_pixels": shared_tail_pixels,
        }
        return base_plan

    height, width = tail_component.shape
    roi_padding = max(2, int(round(contact_radius + 10 * scale)))
    bbox = bbox_from_mask_with_padding(tail_component, width, height, roi_padding)
    if bbox is None:
        base_plan["mode"] = "ambiguous"
        base_plan["ambiguous"] = {
            "reason": "missing_tail_bbox",
            "touching_head_ids": touching_head_ids,
            "shared_tail_pixels": shared_tail_pixels,
        }
        return base_plan

    x0, y0, x1, y1 = bbox
    tail_roi = tail_component[y0:y1, x0:x1].astype(np.uint8) * 255
    rgb_roi = rgb[y0:y1, x0:x1] if rgb is not None else None
    needs_color_score = path_mode in {"color-only", "hybrid"} or record_ablation_scores
    color_context = (
        make_color_scoring_context(rgb_roi, normalization, scale)
        if needs_color_score
        else None
    )
    skeleton = skeletonize_binary(tail_roi)
    coords_yx, neighbors = skeleton_graph(skeleton)
    if len(coords_yx) == 0:
        base_plan["mode"] = "ambiguous"
        base_plan["ambiguous"] = {
            "reason": "missing_skeleton",
            "touching_head_ids": touching_head_ids,
            "shared_tail_pixels": shared_tail_pixels,
            "roi_xyxy": [x0, y0, x1, y1],
        }
        return base_plan

    candidate_head_areas, min_head_area = head_area_threshold_for_shared_tail(
        touching_head_ids, head_labels, scale, head_areas=head_areas
    )
    anchors: dict[int, int] = {}
    anchor_metadata: dict[int, dict[str, Any]] = {}
    max_anchor_distance = max(float(contact_radius) + 10.0 * scale, 18.0 * scale)
    for head_id in touching_head_ids:
        head_area = candidate_head_areas.get(head_id, 0)
        if head_area < min_head_area:
            base_plan["rejected_heads"].append(
                {
                    "head_id": head_id,
                    "reason": "small_head_candidate",
                    "head_area_px": head_area,
                    "min_head_area_px": min_head_area,
                }
            )
            continue

        head_roi = head_labels[y0:y1, x0:x1] == head_id
        anchor_idx, anchor_distance, anchor_xy = nearest_skeleton_anchor(coords_yx, head_roi)
        if anchor_idx is None or anchor_distance is None or anchor_xy is None:
            base_plan["rejected_heads"].append(
                {
                    "head_id": head_id,
                    "reason": "missing_head_anchor",
                    "head_area_px": head_area,
                }
            )
            continue
        if anchor_distance > max_anchor_distance:
            base_plan["rejected_heads"].append(
                {
                    "head_id": head_id,
                    "reason": "head_anchor_too_far",
                    "head_area_px": head_area,
                    "anchor_distance_px": anchor_distance,
                    "max_anchor_distance_px": max_anchor_distance,
                }
            )
            continue
        anchors[head_id] = int(anchor_idx)
        anchor_metadata[head_id] = {
            "anchor_xy": [int(anchor_xy[0] + x0), int(anchor_xy[1] + y0)],
            "anchor_distance_px": anchor_distance,
            "head_area_px": head_area,
        }

    if not anchors:
        base_plan["mode"] = "ambiguous"
        base_plan["ambiguous"] = {
            "reason": "no_valid_head_candidates",
            "touching_head_ids": touching_head_ids,
            "rejected_heads": base_plan["rejected_heads"],
            "shared_tail_pixels": shared_tail_pixels,
            "roi_xyxy": [x0, y0, x1, y1],
        }
        return base_plan

    assignments: list[dict[str, Any]] = []
    path_debug: dict[int, Any] = {}
    for head_id, anchor_idx in sorted(anchors.items()):
        head_roi = head_labels[y0:y1, x0:x1] == head_id
        scored_paths = score_head_tail_paths(
            tail_roi=tail_roi,
            coords_yx=coords_yx,
            neighbors=neighbors,
            anchor_idx=anchor_idx,
            head_roi=head_roi,
            scale=scale,
            contact_radius=contact_radius,
            average_tail_width_px=average_tail_width_px,
            rgb_roi=rgb_roi,
            path_mode=path_mode,
            normalization=normalization,
            color_context=color_context,
            record_ablation_scores=record_ablation_scores,
        )
        best = scored_paths["best"]
        path_debug[head_id] = {
            "reason": scored_paths["reason"],
            "candidate_count": len(scored_paths["candidates"]),
        }
        if best is None:
            base_plan["rejected_heads"].append(
                {
                    "head_id": head_id,
                    "reason": scored_paths["reason"],
                    "head_area_px": anchor_metadata[head_id]["head_area_px"],
                }
            )
            continue

        local_tail_mask = best["path_mask"] > 0
        local_skeleton_mask = best["path_skeleton_mask"] > 0
        assigned_tail_pixels = int(np.count_nonzero(local_tail_mask))
        assigned_skeleton_pixels = int(np.count_nonzero(local_skeleton_mask))
        trimmed_pixels = max(0, shared_tail_pixels - assigned_tail_pixels)
        assignment = {
            "head_id": head_id,
            "mask": full_size_roi_mask(local_tail_mask, tail_component.shape, x0, y0),
            "path_skeleton_mask": full_size_roi_mask(local_skeleton_mask, tail_component.shape, x0, y0),
            "head_collar_mask": full_size_roi_mask(best["head_collar_mask"] > 0, tail_component.shape, x0, y0),
            "competing_path_skeleton_mask": full_size_roi_mask(
                best["competing_path_skeleton_mask"] > 0,
                tail_component.shape,
                x0,
                y0,
            ),
            "assigned_tail_pixels": assigned_tail_pixels,
            "assigned_skeleton_pixels": assigned_skeleton_pixels,
            "shared_tail_pixels": shared_tail_pixels,
            "touching_head_count": len(touching_head_ids),
            "anchor_xy": anchor_metadata[head_id]["anchor_xy"],
            "anchor_distance_px": anchor_metadata[head_id]["anchor_distance_px"],
            "head_anchor_xy": anchor_metadata[head_id]["anchor_xy"],
            "head_anchor_distance_px": anchor_metadata[head_id]["anchor_distance_px"],
            "head_area_px": anchor_metadata[head_id]["head_area_px"],
            "width_spike_split": False,
            "path_v2": True,
            "path_mode": path_mode,
            "path_endpoint_xy": [int(best["endpoint_xy"][0] + x0), int(best["endpoint_xy"][1] + y0)],
            "path_length_px": round(float(best["path_length_px"]), 3),
            "path_confidence_score": round(100.0 * float(best["path_score"]), 3),
            "path_score_geometry": score_to_percent(best.get("path_score_geometry")),
            "path_score_color": score_to_percent(best.get("path_score_color")),
            "path_score_hybrid": score_to_percent(best.get("path_score_hybrid")),
            "path_score_margin": round(100.0 * float(best["path_score_margin"]), 3),
            "path_score_margin_geometry": score_to_percent(best.get("path_score_margin_geometry")),
            "path_score_margin_color": score_to_percent(best.get("path_score_margin_color")),
            "path_score_margin_hybrid": score_to_percent(best.get("path_score_margin_hybrid")),
            "path_rank_geometry": best.get("path_rank_geometry"),
            "path_rank_color": best.get("path_rank_color"),
            "path_rank_hybrid": best.get("path_rank_hybrid"),
            "path_low_margin": bool(best.get("path_low_margin", False)),
            "competing_path_count": int(best["competing_path_count"]),
            "path_color_continuity_score": (
                None
                if best.get("path_color_continuity_score") is None
                else round(100.0 * float(best["path_color_continuity_score"]), 3)
            ),
            "path_color_delta_mean": (
                None
                if best.get("path_color_delta_mean") is None
                else round(float(best["path_color_delta_mean"]), 3)
            ),
            "path_color_delta_p75": (
                None
                if best.get("path_color_delta_p75") is None
                else round(float(best["path_color_delta_p75"]), 3)
            ),
            "path_color_reference_points": int(best.get("path_color_reference_points") or 0),
            "path_color_normalization": best.get("path_color_normalization"),
            "color_score_raw": (
                None
                if best.get("color_score_raw") is None
                else round(100.0 * float(best["color_score_raw"]), 3)
            ),
            "color_score_normalized": (
                None
                if best.get("color_score_normalized") is None
                else round(100.0 * float(best["color_score_normalized"]), 3)
            ),
            "color_delta_p75_raw": (
                None
                if best.get("color_delta_p75_raw") is None
                else round(float(best["color_delta_p75_raw"]), 3)
            ),
            "color_delta_p75_normalized": (
                None
                if best.get("color_delta_p75_normalized") is None
                else round(float(best["color_delta_p75_normalized"]), 3)
            ),
            "path_curvature_total": round(float(best["path_curvature_total"]), 3),
            "path_branchpoint_count": int(best["path_branchpoint_count"]),
            "path_width_spike_ratio": round(float(best["path_width_spike_ratio"]), 3),
            "path_trimmed_branch_pixels": int(trimmed_pixels),
            "path_trimmed_branch_fraction": round(float(trimmed_pixels) / float(max(1, shared_tail_pixels)), 3),
            "tail_pixels_inside_head_collar": int(best["tail_pixels_inside_head_collar"]),
            "tail_pixels_outside_head_collar": int(best["tail_pixels_outside_head_collar"]),
            "head_exit_count": int(best["head_exit_count"]),
            "head_exit_path_length_px": round(float(best["head_exit_path_length_px"]), 3),
            "head_exit_xy": (
                [int(best["head_exit_xy"][0] + x0), int(best["head_exit_xy"][1] + y0)]
                if best.get("head_exit_xy") is not None
                else None
            ),
            "valid_tail_continuation": bool(best["valid_tail_continuation"]),
            "path_rejection_reason": best.get("path_rejection_reason"),
            "path_ambiguity_reason": best.get("path_ambiguity_reason"),
        }
        assignments.append(assignment)

    if not assignments:
        base_plan["mode"] = "ambiguous"
        base_plan["ambiguous"] = {
            "reason": "missing_valid_path_assignments",
            "touching_head_ids": touching_head_ids,
            "rejected_heads": base_plan["rejected_heads"],
            "shared_tail_pixels": shared_tail_pixels,
            "roi_xyxy": [x0, y0, x1, y1],
        }
        return base_plan

    mark_path_conflicts(assignments)
    base_plan["mode"] = "single_head_component" if len(touching_head_ids) == 1 else "split_shared_tail"
    base_plan["assignments"] = assignments
    base_plan["roi_xyxy"] = [x0, y0, x1, y1]
    base_plan["path_debug"] = path_debug
    return base_plan


def build_tail_assignment_plan(
    tail_component: np.ndarray,
    head_labels: np.ndarray,
    touching_head_ids: list[int],
    scale: float,
    contact_radius: int,
    head_areas: dict[int, int] | None = None,
    average_tail_width_px: float | None = None,
    rgb: np.ndarray | None = None,
    use_path_scoring: bool = False,
    path_mode: str = "hybrid",
    normalization: str = "raw",
    record_ablation_scores: bool = False,
) -> dict[str, Any]:
    if use_path_scoring:
        return build_path_scored_tail_assignment_plan(
            tail_component=tail_component,
            head_labels=head_labels,
            touching_head_ids=touching_head_ids,
            scale=scale,
            contact_radius=contact_radius,
            head_areas=head_areas,
            average_tail_width_px=average_tail_width_px,
            rgb=rgb,
            path_mode=path_mode,
            normalization=normalization,
            record_ablation_scores=record_ablation_scores,
        )

    touching_head_ids = sorted({int(head_id) for head_id in touching_head_ids})
    shared_tail_pixels = int(np.count_nonzero(tail_component))
    base_plan: dict[str, Any] = {
        "mode": "unassigned",
        "assignments": [],
        "rejected_heads": [],
        "ambiguous": None,
        "touching_head_ids": touching_head_ids,
        "touching_head_count": len(touching_head_ids),
        "shared_tail_pixels": shared_tail_pixels,
    }

    if not touching_head_ids or shared_tail_pixels == 0:
        base_plan["mode"] = "ambiguous"
        base_plan["ambiguous"] = {
            "reason": "missing_tail_or_heads",
            "touching_head_ids": touching_head_ids,
            "shared_tail_pixels": shared_tail_pixels,
        }
        return base_plan

    if len(touching_head_ids) == 1:
        head_id = touching_head_ids[0]
        base_plan["mode"] = "single_head_component"
        base_plan["assignments"].append(
            {
                "head_id": head_id,
                "mask": tail_component.astype(bool).copy(),
                "assigned_tail_pixels": shared_tail_pixels,
                "assigned_skeleton_pixels": None,
                "shared_tail_pixels": shared_tail_pixels,
                "touching_head_count": 1,
                "anchor_xy": None,
                "anchor_distance_px": None,
            }
        )
        return base_plan

    height, width = tail_component.shape
    roi_padding = max(2, int(round(contact_radius + 8 * scale)))
    bbox = bbox_from_mask_with_padding(tail_component, width, height, roi_padding)
    if bbox is None:
        base_plan["mode"] = "ambiguous"
        base_plan["ambiguous"] = {
            "reason": "missing_tail_bbox",
            "touching_head_ids": touching_head_ids,
            "shared_tail_pixels": shared_tail_pixels,
        }
        return base_plan

    x0, y0, x1, y1 = bbox
    tail_roi = tail_component[y0:y1, x0:x1].astype(np.uint8) * 255
    skeleton = skeletonize_binary(tail_roi)
    coords_yx, neighbors = skeleton_graph(skeleton)
    if len(coords_yx) == 0:
        base_plan["mode"] = "ambiguous"
        base_plan["ambiguous"] = {
            "reason": "missing_skeleton",
            "touching_head_ids": touching_head_ids,
            "shared_tail_pixels": shared_tail_pixels,
            "roi_xyxy": [x0, y0, x1, y1],
        }
        return base_plan

    candidate_head_areas, min_head_area = head_area_threshold_for_shared_tail(
        touching_head_ids, head_labels, scale, head_areas=head_areas
    )
    valid_head_ids: list[int] = []
    for head_id in touching_head_ids:
        head_area = candidate_head_areas.get(head_id, 0)
        if head_area < min_head_area:
            base_plan["rejected_heads"].append(
                {
                    "head_id": head_id,
                    "reason": "small_head_candidate",
                    "head_area_px": head_area,
                    "min_head_area_px": min_head_area,
                }
            )
            continue
        valid_head_ids.append(head_id)

    if not valid_head_ids:
        base_plan["mode"] = "ambiguous"
        base_plan["ambiguous"] = {
            "reason": "no_valid_head_candidates",
            "touching_head_ids": touching_head_ids,
            "rejected_heads": base_plan["rejected_heads"],
            "shared_tail_pixels": shared_tail_pixels,
            "roi_xyxy": [x0, y0, x1, y1],
        }
        return base_plan

    if len(valid_head_ids) == 1:
        head_id = valid_head_ids[0]
        base_plan["mode"] = "filtered_single_shared_tail"
        base_plan["assignments"].append(
            {
                "head_id": head_id,
                "mask": tail_component.astype(bool).copy(),
                "assigned_tail_pixels": shared_tail_pixels,
                "assigned_skeleton_pixels": int(cv2.countNonZero(skeleton)),
                "shared_tail_pixels": shared_tail_pixels,
                "touching_head_count": len(touching_head_ids),
                "anchor_xy": None,
                "anchor_distance_px": None,
            }
        )
        return base_plan

    max_anchor_distance = max(float(contact_radius) + 10.0 * scale, 18.0 * scale)
    anchors: dict[int, int] = {}
    anchor_metadata: dict[int, dict[str, Any]] = {}
    for head_id in valid_head_ids:
        head_roi = head_labels[y0:y1, x0:x1] == head_id
        anchor_idx, anchor_distance, anchor_xy = nearest_skeleton_anchor(coords_yx, head_roi)
        if anchor_idx is None or anchor_distance is None or anchor_xy is None:
            base_plan["rejected_heads"].append(
                {
                    "head_id": head_id,
                    "reason": "missing_head_anchor",
                    "head_area_px": candidate_head_areas.get(head_id, 0),
                }
            )
            continue
        if anchor_distance > max_anchor_distance:
            base_plan["rejected_heads"].append(
                {
                    "head_id": head_id,
                    "reason": "head_anchor_too_far",
                    "head_area_px": candidate_head_areas.get(head_id, 0),
                    "anchor_distance_px": anchor_distance,
                    "max_anchor_distance_px": max_anchor_distance,
                }
            )
            continue

        anchors[head_id] = anchor_idx
        anchor_metadata[head_id] = {
            "anchor_xy": [int(anchor_xy[0] + x0), int(anchor_xy[1] + y0)],
            "anchor_distance_px": anchor_distance,
            "head_area_px": candidate_head_areas.get(head_id, 0),
        }

    if len(anchors) < 2:
        base_plan["mode"] = "ambiguous"
        base_plan["ambiguous"] = {
            "reason": "fewer_than_two_valid_anchors",
            "touching_head_ids": touching_head_ids,
            "valid_head_ids": valid_head_ids,
            "rejected_heads": base_plan["rejected_heads"],
            "shared_tail_pixels": shared_tail_pixels,
            "roi_xyxy": [x0, y0, x1, y1],
        }
        return base_plan

    width_spike_analysis = directional_width_spike_analysis(
        tail_roi=tail_roi,
        coords_yx=coords_yx,
        neighbors=neighbors,
        anchors=anchors,
        scale=scale,
        average_tail_width_px=average_tail_width_px,
    )
    width_barrier_fraction = (
        float(width_spike_analysis["barrier_pixels"]) / float(shared_tail_pixels)
        if shared_tail_pixels > 0
        else 0.0
    )
    use_width_barrier = (
        width_spike_analysis["barrier_pixels"] > 0
        and width_barrier_fraction <= 0.12
    )
    blocked_indices = set(width_spike_analysis["barrier_indices"]) if use_width_barrier else set()
    skeleton_owners = multi_source_skeleton_owners(
        coords_yx,
        neighbors,
        anchors,
        blocked_indices=blocked_indices,
    )
    assignment_roi = flood_tail_assignments(
        tail_roi,
        coords_yx,
        skeleton_owners,
        barrier_mask=width_spike_analysis["barrier_mask"] if use_width_barrier else None,
    )
    directional_width_summary = {
        "normal_width_px": width_spike_analysis["normal_width_px"],
        "threshold_width_px": width_spike_analysis["threshold_width_px"],
        "width_percentiles": width_spike_analysis.get("width_percentiles"),
        "sampled_skeleton_points": width_spike_analysis["sampled_skeleton_points"],
        "spike_skeleton_points": width_spike_analysis["spike_skeleton_points"],
        "barrier_pixels": width_spike_analysis["barrier_pixels"],
        "barrier_fraction": width_barrier_fraction,
        "used_as_cut_barrier": use_width_barrier,
    }
    width_spike_mask = full_size_roi_mask(
        width_spike_analysis["barrier_mask"] > 0,
        tail_component.shape,
        x0,
        y0,
    )
    min_tail_pixels = max(int(round(24 * scale * scale)), int(round(0.003 * shared_tail_pixels)))
    min_skeleton_pixels = max(int(round(6 * scale)), int(round(0.003 * len(coords_yx))))

    accepted: list[dict[str, Any]] = []
    for head_id in sorted(anchors):
        local_tail_mask = assignment_roi == head_id
        assigned_tail_pixels = int(np.count_nonzero(local_tail_mask))
        assigned_skeleton_pixels = int(np.count_nonzero(skeleton_owners == head_id))
        if assigned_tail_pixels < min_tail_pixels or assigned_skeleton_pixels < min_skeleton_pixels:
            base_plan["rejected_heads"].append(
                {
                    "head_id": head_id,
                    "reason": "weak_assigned_tail_segment",
                    "assigned_tail_pixels": assigned_tail_pixels,
                    "assigned_skeleton_pixels": assigned_skeleton_pixels,
                    "min_tail_pixels": min_tail_pixels,
                    "min_skeleton_pixels": min_skeleton_pixels,
                }
            )
            continue

        accepted.append(
            {
                "head_id": head_id,
                "mask": full_size_roi_mask(local_tail_mask, tail_component.shape, x0, y0),
                "assigned_tail_pixels": assigned_tail_pixels,
                "assigned_skeleton_pixels": assigned_skeleton_pixels,
                "shared_tail_pixels": shared_tail_pixels,
                "touching_head_count": len(touching_head_ids),
                "anchor_xy": anchor_metadata[head_id]["anchor_xy"],
                "anchor_distance_px": anchor_metadata[head_id]["anchor_distance_px"],
                "head_area_px": anchor_metadata[head_id]["head_area_px"],
                "width_spike_split": bool(use_width_barrier),
            }
        )

    if len(accepted) < 2:
        base_plan["directional_width"] = directional_width_summary
        base_plan["width_spike_mask"] = width_spike_mask
        base_plan["mode"] = "ambiguous"
        base_plan["ambiguous"] = {
            "reason": "insufficient_accepted_segments",
            "touching_head_ids": touching_head_ids,
            "valid_anchor_head_ids": sorted(anchors),
            "rejected_heads": base_plan["rejected_heads"],
            "shared_tail_pixels": shared_tail_pixels,
            "roi_xyxy": [x0, y0, x1, y1],
        }
        return base_plan

    base_plan["mode"] = "split_shared_tail"
    base_plan["assignments"] = accepted
    base_plan["roi_xyxy"] = [x0, y0, x1, y1]
    base_plan["min_tail_pixels"] = min_tail_pixels
    base_plan["min_skeleton_pixels"] = min_skeleton_pixels
    base_plan["directional_width"] = directional_width_summary
    base_plan["width_spike_mask"] = width_spike_mask
    return base_plan


def assignment_plan_for_json(plan: dict[str, Any]) -> dict[str, Any]:
    return {
        "mode": plan["mode"],
        "path_v2": bool(plan.get("path_v2", False)),
        "path_mode": plan.get("path_mode"),
        "normalization": plan.get("normalization"),
        "touching_head_ids": plan.get("touching_head_ids", []),
        "touching_head_count": plan.get("touching_head_count", 0),
        "shared_tail_pixels": plan.get("shared_tail_pixels", 0),
        "accepted_head_ids": [row["head_id"] for row in plan.get("assignments", [])],
        "rejected_heads": plan.get("rejected_heads", []),
        "directional_width": plan.get("directional_width"),
        "path_debug": plan.get("path_debug"),
        "ambiguous": plan.get("ambiguous"),
    }


def tail_assignment_debug_overlay(
    rgb: np.ndarray,
    tail_component: np.ndarray,
    head_labels: np.ndarray,
    touching_head_ids: list[int],
    plan: dict[str, Any],
    tail: ComponentInfo,
) -> np.ndarray:
    debug = rgb.copy()
    base_mask = tail_component > 0
    if np.any(base_mask):
        color = np.array([0, 210, 255], dtype=np.float32)
        debug[base_mask] = (0.45 * debug[base_mask].astype(np.float32) + 0.55 * color).astype(
            np.uint8
        )

    accepted_head_ids = {int(row["head_id"]) for row in plan.get("assignments", [])}
    rejected_head_ids = {int(row["head_id"]) for row in plan.get("rejected_heads", [])}
    for assignment in plan.get("assignments", []):
        competing_path_mask = assignment.get("competing_path_skeleton_mask")
        if isinstance(competing_path_mask, np.ndarray) and np.any(competing_path_mask):
            line = cv2.dilate(
                competing_path_mask.astype(np.uint8) * 255,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
                iterations=1,
            )
            mask = line > 0
            color = np.array([165, 165, 165], dtype=np.float32)
            debug[mask] = (0.35 * debug[mask].astype(np.float32) + 0.65 * color).astype(
                np.uint8
            )

    for idx, assignment in enumerate(plan.get("assignments", [])):
        mask = assignment["mask"] > 0
        if not np.any(mask):
            continue
        color = ASSIGNMENT_PALETTE[idx % len(ASSIGNMENT_PALETTE)]
        debug[mask] = (0.25 * debug[mask].astype(np.float32) + 0.75 * color).astype(np.uint8)

        collar_mask = assignment.get("head_collar_mask")
        if isinstance(collar_mask, np.ndarray) and np.any(collar_mask):
            contours, _ = cv2.findContours(
                collar_mask.astype(np.uint8) * 255,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            cv2.drawContours(debug, contours, -1, (255, 255, 255), 1)

        path_mask = assignment.get("path_skeleton_mask")
        if isinstance(path_mask, np.ndarray) and np.any(path_mask):
            line = cv2.dilate(
                path_mask.astype(np.uint8) * 255,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
                iterations=1,
            )
            path_pixels = line > 0
            if np.any(path_pixels):
                path_color = np.array([0, 255, 70], dtype=np.float32)
                debug[path_pixels] = (
                    0.15 * debug[path_pixels].astype(np.float32) + 0.85 * path_color
                ).astype(np.uint8)

    width_spike_mask = plan.get("width_spike_mask")
    if width_spike_mask is not None and np.any(width_spike_mask):
        mask = width_spike_mask > 0
        color = np.array([255, 35, 35], dtype=np.float32)
        debug[mask] = (0.15 * debug[mask].astype(np.float32) + 0.85 * color).astype(np.uint8)

    for head_id in touching_head_ids:
        head_mask = (head_labels == head_id).astype(np.uint8) * 255
        contours, _ = cv2.findContours(head_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if head_id in accepted_head_ids:
            contour_color = (0, 255, 120)
        elif head_id in rejected_head_ids:
            contour_color = (255, 60, 60)
        else:
            contour_color = (255, 255, 255)
        cv2.drawContours(debug, contours, -1, contour_color, 2)

    for endpoint_x, endpoint_y in tail.endpoints_xy or []:
        cv2.circle(debug, (int(endpoint_x), int(endpoint_y)), 4, (0, 80, 255), thickness=-1)
    for branch_x, branch_y in tail.branchpoints_xy or []:
        cv2.circle(debug, (int(branch_x), int(branch_y)), 4, (255, 235, 0), thickness=-1)

    bbox = bbox_from_mask(tail_component)
    if bbox is not None:
        x0, y0, x1, y1 = bbox
        border_color = (255, 60, 60) if plan.get("mode") == "ambiguous" else (255, 255, 255)
        cv2.rectangle(debug, (x0, y0), (x1 - 1, y1 - 1), border_color, 2)

    return debug


def skeleton_longest_path(skeleton: np.ndarray) -> np.ndarray:
    binary = (skeleton > 0).astype(np.uint8)
    coords_yx = np.argwhere(binary > 0)
    if len(coords_yx) == 0:
        return np.zeros_like(skeleton, dtype=np.uint8)

    coord_to_idx = {tuple(coord): idx for idx, coord in enumerate(coords_yx)}
    neighbors: list[list[int]] = [[] for _ in range(len(coords_yx))]
    for idx, (y, x) in enumerate(coords_yx):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                neighbor_idx = coord_to_idx.get((y + dy, x + dx))
                if neighbor_idx is not None:
                    neighbors[idx].append(neighbor_idx)

    endpoint_indices = [idx for idx, items in enumerate(neighbors) if len(items) <= 1]
    if len(endpoint_indices) < 2:
        return binary * 255

    def farthest_from(start_idx: int) -> tuple[int, list[int]]:
        parent = [-1] * len(coords_yx)
        distance = [-1] * len(coords_yx)
        queue = [start_idx]
        distance[start_idx] = 0
        for current in queue:
            for neighbor_idx in neighbors[current]:
                if distance[neighbor_idx] != -1:
                    continue
                parent[neighbor_idx] = current
                distance[neighbor_idx] = distance[current] + 1
                queue.append(neighbor_idx)

        reachable_endpoints = [idx for idx in endpoint_indices if distance[idx] >= 0]
        if not reachable_endpoints:
            reachable_endpoints = [idx for idx, value in enumerate(distance) if value >= 0]
        farthest = max(reachable_endpoints, key=lambda idx: distance[idx])
        path: list[int] = []
        current = farthest
        while current != -1:
            path.append(current)
            current = parent[current]
        return farthest, path

    first, _ = farthest_from(endpoint_indices[0])
    _, path = farthest_from(first)
    line = np.zeros_like(skeleton, dtype=np.uint8)
    for idx in path:
        y, x = coords_yx[idx]
        line[y, x] = 255
    return line


def overlay_skeleton(rgb_crop: np.ndarray, skeleton: np.ndarray) -> np.ndarray:
    overlay = rgb_crop.copy()
    line = cv2.dilate(
        (skeleton > 0).astype(np.uint8) * 255,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    mask = line > 0
    if np.any(mask):
        color = np.array([0, 255, 90], dtype=np.float32)
        overlay[mask] = (0.25 * overlay[mask].astype(np.float32) + 0.75 * color).astype(np.uint8)
    return overlay


def highlighted_crop_overlay(
    rgb_crop: np.ndarray,
    tail_mask: np.ndarray,
    head_mask: np.ndarray,
    overlap_mask: np.ndarray,
) -> np.ndarray:
    highlighted = overlay_masks(rgb_crop, tail_mask, head_mask, overlap_mask)
    height, width = highlighted.shape[:2]
    if height > 2 and width > 2:
        cv2.rectangle(
            highlighted,
            (0, 0),
            (width - 1, height - 1),
            color=(255, 255, 255),
            thickness=2,
        )
        cv2.rectangle(
            highlighted,
            (2, 2),
            (width - 3, height - 3),
            color=(0, 0, 0),
            thickness=1,
        )
    return highlighted


def crop_candidate_thresholds(candidates: list[dict[str, Any]], scale: float) -> dict[str, Any]:
    tail_pixels = np.array(
        [int(row.get("assigned_tail_pixels", 0)) for row in candidates if int(row.get("assigned_tail_pixels", 0)) > 0],
        dtype=np.float32,
    )
    skeleton_pixels = np.array(
        [
            int(row.get("assigned_skeleton_pixels") or 0)
            for row in candidates
            if int(row.get("assigned_skeleton_pixels") or 0) > 0
        ],
        dtype=np.float32,
    )
    median_tail = float(np.median(tail_pixels)) if tail_pixels.size else 0.0
    median_skeleton = float(np.median(skeleton_pixels)) if skeleton_pixels.size else 0.0
    min_tail_pixels = max(
        int(round(300 * scale * scale)),
        int(round(0.05 * median_tail)) if median_tail > 0 else 0,
    )
    min_skeleton_pixels = max(
        int(round(18 * scale)),
        int(round(0.05 * median_skeleton)) if median_skeleton > 0 else 0,
    )
    return {
        "median_tail_pixels": median_tail,
        "median_skeleton_pixels": median_skeleton,
        "min_tail_pixels": min_tail_pixels,
        "min_skeleton_pixels": min_skeleton_pixels,
    }


def infer_tail_cutoff_stage(candidate: dict[str, Any]) -> str:
    if candidate.get("source_bbox_touches_image_border") or candidate.get("crop_bbox_touches_image_border"):
        return "image_or_crop_boundary"
    if candidate.get("crop_boundary_risk") == "high":
        return "crop_boundary"
    if candidate.get("path_rejection_reason") == "head_outline_false_tail":
        return "head_anchor_or_outline"
    if candidate.get("path_rejection_reason") == "tail_path_too_short":
        return "path_endpoint_selection"
    if candidate.get("valid_tail_continuation") is False:
        return "path_endpoint_selection"
    if candidate.get("path_ambiguity_reason") in {
        "ambiguous_path_color_margin",
        "ambiguous_path_conflict",
        "low_color_continuity",
    }:
        return "branch_intersection_or_color"
    if float(candidate.get("path_trimmed_branch_fraction") or 0.0) > 0.65:
        return "path_trimmed_component"
    if int(candidate.get("assigned_tail_pixels", 0)) < 300:
        return "component_or_mask_filter"
    return "not_flagged"


def score_crop_candidate(
    candidate: dict[str, Any],
    thresholds: dict[str, Any],
    scale: float,
    acceptance_profile: str = "conservative",
) -> None:
    tail_pixels = int(candidate.get("assigned_tail_pixels", 0))
    skeleton_pixels = int(candidate.get("assigned_skeleton_pixels") or 0)
    foreign_content_score = float(candidate.get("foreign_content_score", 0.0))
    touching_head_count = int(candidate.get("touching_head_count", 1))
    branchpoint_count = int(candidate.get("tail_branchpoint_count", 0))
    endpoint_count = int(candidate.get("tail_endpoint_count", 0))
    tail_overlap_reasons = list(candidate.get("tail_overlap_reasons", []))
    has_overlap = bool(candidate.get("tail_has_overlap", False))
    width_spike_split = bool(candidate.get("width_spike_split", False))
    path_v2 = bool(candidate.get("path_v2", False))
    path_rejection_reason = candidate.get("path_rejection_reason")
    path_ambiguity_reason = candidate.get("path_ambiguity_reason")
    valid_tail_continuation = candidate.get("valid_tail_continuation")
    path_confidence_score = (
        float(candidate.get("path_confidence_score"))
        if candidate.get("path_confidence_score") not in (None, "")
        else (100.0 if not path_v2 else 0.0)
    )
    path_color_continuity_score = (
        float(candidate.get("path_color_continuity_score"))
        if candidate.get("path_color_continuity_score") not in (None, "")
        else None
    )
    path_trimmed_branch_fraction = float(candidate.get("path_trimmed_branch_fraction") or 0.0)
    path_low_margin = bool(candidate.get("path_low_margin", False))

    median_tail = max(1.0, float(thresholds.get("median_tail_pixels", 0.0)))
    median_skeleton = max(1.0, float(thresholds.get("median_skeleton_pixels", 0.0)))
    tail_score = min(28.0, 28.0 * tail_pixels / max(tail_pixels, median_tail))
    skeleton_score = min(18.0, 18.0 * skeleton_pixels / max(skeleton_pixels, median_skeleton))
    topology_penalty = max(0, branchpoint_count - 2) * 2.5 + max(0, endpoint_count - 3) * 1.5
    score = 45.0 + tail_score + skeleton_score
    if candidate.get("tail_assignment_mode") == "single_head_component":
        score += 8.0
    if width_spike_split:
        score += 6.0
    if has_overlap:
        score -= 8.0
    if touching_head_count > 1:
        score -= min(18.0, 4.0 * (touching_head_count - 1))
    score -= min(36.0, foreign_content_score * 30.0)
    score -= min(20.0, topology_penalty)
    if candidate.get("source_bbox_touches_image_border"):
        score -= 8.0
    if candidate.get("crop_bbox_touches_image_border"):
        score -= 4.0
    if path_v2:
        score += max(-20.0, min(8.0, (path_confidence_score - 60.0) / 5.0))

    crop_quality_score = 100.0 - min(70.0, foreign_content_score * 85.0)
    if candidate.get("source_bbox_touches_image_border"):
        crop_quality_score -= 10.0
    if candidate.get("crop_bbox_touches_image_border"):
        crop_quality_score -= 5.0
    crop_quality_score = max(0.0, min(100.0, crop_quality_score))

    mask_quality_score = 100.0
    mask_quality_score -= min(35.0, topology_penalty * 3.0)
    if has_overlap:
        mask_quality_score -= 12.0
    if touching_head_count > 1:
        mask_quality_score -= min(20.0, 4.0 * (touching_head_count - 1))
    if path_trimmed_branch_fraction > 0.15:
        mask_quality_score -= min(24.0, path_trimmed_branch_fraction * 45.0)
    if valid_tail_continuation is False:
        mask_quality_score -= 45.0
    if path_color_continuity_score is not None and path_color_continuity_score < 55.0:
        mask_quality_score -= min(25.0, (55.0 - path_color_continuity_score) * 0.6)
    if path_v2:
        mask_quality_score = 0.65 * mask_quality_score + 0.35 * path_confidence_score
    mask_quality_score = max(0.0, min(100.0, mask_quality_score))

    risk_flags: set[str] = set()
    if candidate.get("source_bbox_touches_image_border"):
        risk_flags.add("source_bbox_touches_image_border")
    if candidate.get("crop_bbox_touches_image_border"):
        risk_flags.add("crop_bbox_touches_image_border")
    if candidate.get("crop_boundary_risk") in {"high", "medium"}:
        risk_flags.add(f"crop_boundary_{candidate['crop_boundary_risk']}_risk")
    if foreign_content_score >= 0.35:
        risk_flags.add("foreign_content_in_crop")
    if has_overlap:
        risk_flags.add("tail_overlap_component")
    if touching_head_count > 1:
        risk_flags.add("shared_tail_component")
    if path_v2 and path_trimmed_branch_fraction > 0.15:
        risk_flags.add("path_trimmed_extra_branch")
        if path_trimmed_branch_fraction > 0.65 and mask_quality_score < 50.0:
            risk_flags.add("mask_branch_extra_tail")
    elif branchpoint_count >= 3 or path_trimmed_branch_fraction > 0.15:
        risk_flags.add("mask_branch_extra_tail")
    if path_low_margin:
        risk_flags.add("path_low_margin")
    if valid_tail_continuation is False:
        risk_flags.add(str(path_rejection_reason or "mask_tail_partial"))
    if path_color_continuity_score is not None and path_color_continuity_score < 55.0:
        risk_flags.add("low_color_continuity")
    if path_ambiguity_reason:
        risk_flags.add(str(path_ambiguity_reason))

    status = "accepted"
    rejection_reason: str | None = None
    ambiguity_reason: str | None = None
    min_tail_pixels = int(thresholds["min_tail_pixels"])
    min_skeleton_pixels = int(thresholds["min_skeleton_pixels"])
    strong_clean_path = (
        path_v2
        and path_confidence_score >= 80.0
        and mask_quality_score >= 55.0
        and (
            path_color_continuity_score is None
            or path_color_continuity_score >= 70.0
        )
        and path_ambiguity_reason not in {
            "ambiguous_path_conflict",
            "low_color_continuity",
            "ambiguous_path_color_margin",
        }
    )
    balanced_review_accept = (
        acceptance_profile == "balanced-review"
        and crop_quality_score >= 60.0
        and path_confidence_score >= 75.0
        and valid_tail_continuation is not False
        and path_ambiguity_reason not in {
            "ambiguous_path_conflict",
            "low_color_continuity",
        }
    )
    if path_rejection_reason:
        status = "rejected"
        rejection_reason = str(path_rejection_reason)
        score -= 60.0
    elif tail_pixels < min_tail_pixels or skeleton_pixels < min_skeleton_pixels:
        status = "rejected"
        rejection_reason = "tiny_tail_fragment"
        score -= 60.0
    elif path_ambiguity_reason:
        if balanced_review_accept and str(path_ambiguity_reason) == "ambiguous_path_color_margin":
            risk_flags.add("accepted_with_path_margin_review")
        else:
            status = "ambiguous"
            ambiguity_reason = str(path_ambiguity_reason)
            score -= 18.0
    elif (
        path_v2
        and path_color_continuity_score is not None
        and path_color_continuity_score < 55.0
    ):
        status = "ambiguous"
        ambiguity_reason = "low_color_continuity"
        score -= 18.0
    elif path_v2 and path_confidence_score < 40.0:
        status = "ambiguous"
        ambiguity_reason = "low_path_confidence"
        score -= 15.0
    elif (
        path_v2
        and path_low_margin
        and (
            "mask_branch_extra_tail" in risk_flags
            or (
                path_trimmed_branch_fraction > 0.15
                and (has_overlap or touching_head_count > 1)
                and mask_quality_score < 55.0
            )
        )
    ):
        if balanced_review_accept:
            risk_flags.add("accepted_with_mask_review")
        else:
            status = "ambiguous"
            ambiguity_reason = "mask_branch_extra_tail"
            score -= 15.0
    elif (
        path_v2
        and (candidate.get("source_bbox_touches_image_border") or candidate.get("crop_bbox_touches_image_border"))
        and (has_overlap or path_low_margin)
    ):
        status = "ambiguous"
        ambiguity_reason = "mask_tail_partial"
        score -= 12.0
    elif crop_quality_score < 35.0 and not strong_clean_path:
        status = "ambiguous"
        ambiguity_reason = "foreign_content_in_crop"
        score -= 18.0
    elif mask_quality_score < 45.0:
        if balanced_review_accept and "mask_branch_extra_tail" in risk_flags:
            risk_flags.add("accepted_with_mask_review")
        else:
            status = "ambiguous"
            if "mask_branch_extra_tail" in risk_flags:
                ambiguity_reason = "mask_branch_extra_tail"
            else:
                ambiguity_reason = "mask_tail_partial"
            score -= 15.0
    elif foreign_content_score >= 0.85:
        status = "ambiguous"
        ambiguity_reason = "foreign_content_in_crop"
        score -= 18.0
    elif has_overlap and touching_head_count > 1:
        unresolved_width_spike = (
            "local_width_spike" in tail_overlap_reasons
            and not width_spike_split
        )
        broad_shared_tail = (
            foreign_content_score >= 0.35
            or "wide_sparse_component" in tail_overlap_reasons
            or branchpoint_count >= max(5, touching_head_count + 2)
            or endpoint_count >= max(5, touching_head_count + 2)
        )
        if unresolved_width_spike or broad_shared_tail:
            status = "ambiguous"
            ambiguity_reason = "unresolved_shared_tail_overlap"
            score -= 15.0

    candidate["candidate_status"] = status
    candidate["score"] = round(float(score), 3)
    candidate["crop_quality_score"] = round(float(crop_quality_score), 3)
    candidate["mask_quality_score"] = round(float(mask_quality_score), 3)
    candidate["path_confidence_score"] = round(float(path_confidence_score), 3)
    candidate["candidate_risk_flags"] = sorted(risk_flags)
    candidate["rejection_reason"] = rejection_reason
    candidate["ambiguity_reason"] = ambiguity_reason
    candidate["acceptance_profile"] = acceptance_profile
    candidate["tail_cutoff_stage"] = infer_tail_cutoff_stage(candidate)
    candidate["score_details"] = {
        "min_tail_pixels": min_tail_pixels,
        "min_skeleton_pixels": min_skeleton_pixels,
        "tail_score": round(float(tail_score), 3),
        "skeleton_score": round(float(skeleton_score), 3),
        "foreign_content_penalty": round(float(min(36.0, foreign_content_score * 30.0)), 3),
        "topology_penalty": round(float(min(20.0, topology_penalty)), 3),
        "crop_quality_score": round(float(crop_quality_score), 3),
        "mask_quality_score": round(float(mask_quality_score), 3),
        "path_confidence_score": round(float(path_confidence_score), 3),
        "candidate_risk_flags": sorted(risk_flags),
    }


def candidate_reuse_mask(candidate: dict[str, Any]) -> np.ndarray | None:
    path_mask = candidate.get("tail_path_skeleton_mask")
    if isinstance(path_mask, np.ndarray) and np.any(path_mask):
        return path_mask > 0
    tail_mask = candidate.get("tail_mask")
    if isinstance(tail_mask, np.ndarray) and np.any(tail_mask):
        return tail_mask > 0
    return None


def shared_mask_fraction(candidate_a: dict[str, Any], candidate_b: dict[str, Any]) -> float:
    mask_a = candidate_reuse_mask(candidate_a)
    mask_b = candidate_reuse_mask(candidate_b)
    if mask_a is None or mask_b is None:
        return 0.0
    pixels_a = int(np.count_nonzero(mask_a))
    pixels_b = int(np.count_nonzero(mask_b))
    if pixels_a <= 0 or pixels_b <= 0:
        return 0.0
    overlap = int(np.count_nonzero(mask_a & mask_b))
    return float(overlap) / float(max(1, min(pixels_a, pixels_b)))


def annotate_reuse_metrics(candidates: list[dict[str, Any]]) -> None:
    by_head: dict[int, list[dict[str, Any]]] = {}
    by_tail: dict[int, list[dict[str, Any]]] = {}
    for candidate in candidates:
        image_stem = candidate.get("image_stem", "image")
        head_id = int(candidate["head_label_id"])
        tail_id = int(candidate["tail_id"])
        candidate["head_reuse_group_id"] = f"{image_stem}__head_{head_id:03d}"
        candidate["tail_reuse_group_id"] = f"{image_stem}__tail_{tail_id:03d}"
        candidate["shared_path_fraction"] = 0.0
        by_head.setdefault(head_id, []).append(candidate)
        by_tail.setdefault(tail_id, []).append(candidate)

    for group in by_head.values():
        accepted_count = sum(1 for row in group if row.get("candidate_status") == "accepted")
        for candidate in group:
            candidate["head_reuse_candidate_count"] = len(group)
            candidate["head_reuse_accepted_count"] = accepted_count

    for group in by_tail.values():
        accepted_count = sum(1 for row in group if row.get("candidate_status") == "accepted")
        for candidate in group:
            candidate["tail_reuse_candidate_count"] = len(group)
            candidate["tail_reuse_accepted_count"] = accepted_count
        for idx, candidate in enumerate(group):
            max_shared = 0.0
            for other in group[:idx] + group[idx + 1 :]:
                max_shared = max(max_shared, shared_mask_fraction(candidate, other))
            candidate["shared_path_fraction"] = round(float(max_shared), 3)


def arbitrate_crop_candidates(
    candidates: list[dict[str, Any]],
    scale: float,
    acceptance_profile: str = "conservative",
) -> dict[str, Any]:
    thresholds = crop_candidate_thresholds(candidates, scale)
    for index, candidate in enumerate(candidates):
        candidate["candidate_index"] = index
        candidate["duplicate_group_id"] = f"{candidate.get('image_stem', 'image')}__head_{int(candidate['head_label_id']):03d}"
        candidate["duplicate_of_crop_id"] = None
        score_crop_candidate(candidate, thresholds, scale, acceptance_profile=acceptance_profile)
    annotate_reuse_metrics(candidates)

    by_head: dict[int, list[dict[str, Any]]] = {}
    for candidate in candidates:
        by_head.setdefault(int(candidate["head_label_id"]), []).append(candidate)

    duplicate_rejections = 0
    for head_candidates in by_head.values():
        accepted = [
            candidate
            for candidate in head_candidates
            if candidate.get("candidate_status") == "accepted"
        ]
        if len(accepted) <= 1:
            continue
        accepted.sort(
            key=lambda row: (
                float(row.get("score", 0.0)),
                int(row.get("assigned_tail_pixels", 0)),
                -float(row.get("foreign_content_score", 0.0)),
            ),
            reverse=True,
        )
        winner = accepted[0]
        for duplicate in accepted[1:]:
            duplicate["candidate_status"] = "rejected"
            duplicate["rejection_reason"] = "duplicate_lower_scoring_candidate"
            duplicate["duplicate_of_crop_id"] = winner["crop_id"]
            duplicate_rejections += 1

    tail_duplicate_ambiguous = 0
    by_tail: dict[int, list[dict[str, Any]]] = {}
    for candidate in candidates:
        by_tail.setdefault(int(candidate["tail_id"]), []).append(candidate)
    for tail_candidates in by_tail.values():
        accepted = [
            candidate
            for candidate in tail_candidates
            if candidate.get("candidate_status") == "accepted"
        ]
        if len(accepted) <= 1:
            continue
        accepted.sort(
            key=lambda row: (
                float(row.get("score", 0.0)),
                int(row.get("assigned_tail_pixels", 0)),
                -float(row.get("foreign_content_score", 0.0)),
            ),
            reverse=True,
        )
        winner = accepted[0]
        for duplicate in accepted[1:]:
            shared_fraction = shared_mask_fraction(winner, duplicate)
            duplicate["shared_path_fraction"] = max(
                float(duplicate.get("shared_path_fraction") or 0.0),
                round(shared_fraction, 3),
            )
            if shared_fraction < 0.65:
                continue
            duplicate["candidate_status"] = "ambiguous"
            duplicate["ambiguity_reason"] = "duplicate_shared_tail_path"
            duplicate["duplicate_of_crop_id"] = winner["crop_id"]
            flags = set(duplicate.get("candidate_risk_flags") or [])
            flags.add("duplicate_shared_tail_path")
            duplicate["candidate_risk_flags"] = sorted(flags)
            tail_duplicate_ambiguous += 1

    annotate_reuse_metrics(candidates)
    accepted_head_reuse_count = sum(
        max(0, len([row for row in group if row.get("candidate_status") == "accepted"]) - 1)
        for group in by_head.values()
    )
    accepted_tail_reuse_count = sum(
        max(0, len([row for row in group if row.get("candidate_status") == "accepted"]) - 1)
        for group in by_tail.values()
    )

    return {
        "thresholds": thresholds,
        "duplicate_rejections": duplicate_rejections,
        "tail_duplicate_ambiguous": tail_duplicate_ambiguous,
        "accepted_head_reuse_count": accepted_head_reuse_count,
        "accepted_tail_reuse_count": accepted_tail_reuse_count,
    }


def crop_candidate_for_json(candidate: dict[str, Any]) -> dict[str, Any]:
    omitted = {
        "tail_mask",
        "head_mask",
        "tail_path_skeleton_mask",
        "head_collar_mask",
        "competing_path_skeleton_mask",
    }
    return {
        key: value
        for key, value in candidate.items()
        if key not in omitted and not isinstance(value, np.ndarray)
    }


def write_crop_candidate_artifacts(
    candidate: dict[str, Any],
    rgb: np.ndarray,
    overlap_mask: np.ndarray,
    output_root: Path,
) -> dict[str, str]:
    crop_id = str(candidate["crop_id"])
    status = str(candidate["candidate_status"])
    x0, y0, x1, y1 = [int(value) for value in candidate["bbox_xyxy"]]
    crop_rgb = rgb[y0:y1, x0:x1]
    tail_crop_mask = candidate["tail_mask"][y0:y1, x0:x1].astype(np.uint8) * 255
    head_crop_mask = candidate["head_mask"][y0:y1, x0:x1].astype(np.uint8) * 255

    if status == "accepted":
        crop_path = output_root / "sperm_crops" / f"{crop_id}.png"
        crop_mask_path = output_root / "sperm_crops" / f"{crop_id}_mask.png"
        highlighted_crop_path = output_root / "highlighted_sperm_crops" / f"{crop_id}_highlighted.png"
        skeleton_mask_path = output_root / "tail_skeletons" / f"{crop_id}_skeleton_mask.png"
        skeleton_overlay_path = output_root / "tail_skeletons" / f"{crop_id}_skeleton_overlay.png"
    else:
        review_dir = output_root / "candidate_review" / status
        crop_path = review_dir / f"{crop_id}.png"
        crop_mask_path = review_dir / f"{crop_id}_mask.png"
        highlighted_crop_path = review_dir / f"{crop_id}_highlighted.png"
        skeleton_mask_path = review_dir / f"{crop_id}_skeleton_mask.png"
        skeleton_overlay_path = review_dir / f"{crop_id}_skeleton_overlay.png"

    Image.fromarray(crop_rgb).save(crop_path)
    write_mask(crop_mask_path, (tail_crop_mask > 0) | (head_crop_mask > 0))
    overlap_crop_mask = np.zeros_like(tail_crop_mask, dtype=np.uint8)
    overlap_crop_mask[(overlap_mask[y0:y1, x0:x1] > 0) & (tail_crop_mask > 0)] = 255
    Image.fromarray(
        highlighted_crop_overlay(
            crop_rgb,
            tail_crop_mask,
            head_crop_mask,
            overlap_crop_mask,
        )
    ).save(highlighted_crop_path)

    path_skeleton_mask = candidate.get("tail_path_skeleton_mask")
    if isinstance(path_skeleton_mask, np.ndarray) and np.any(path_skeleton_mask):
        single_line = path_skeleton_mask[y0:y1, x0:x1].astype(np.uint8) * 255
    else:
        skeleton = skeletonize_binary(tail_crop_mask)
        single_line = skeleton_longest_path(skeleton)
    write_mask(skeleton_mask_path, single_line)
    Image.fromarray(overlay_skeleton(crop_rgb, single_line)).save(skeleton_overlay_path)

    return {
        "crop": str(crop_path),
        "crop_mask": str(crop_mask_path),
        "highlighted_crop": str(highlighted_crop_path),
        "skeleton_mask": str(skeleton_mask_path),
        "skeleton_overlay": str(skeleton_overlay_path),
    }


def put_diagnostic_text(image: np.ndarray, lines: list[str]) -> np.ndarray:
    canvas = image.copy()
    y = 18
    for line in lines:
        cv2.putText(
            canvas,
            line[:110],
            (6, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            line[:110],
            (6, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        y += 17
    return canvas


def write_candidate_diagnostics(
    candidate: dict[str, Any],
    rgb: np.ndarray,
    overlap_mask: np.ndarray,
    output_root: Path,
) -> dict[str, str]:
    crop_id = str(candidate["crop_id"])
    x0, y0, x1, y1 = [int(value) for value in candidate["bbox_xyxy"]]
    crop_rgb = rgb[y0:y1, x0:x1]
    tail_crop_mask = candidate["tail_mask"][y0:y1, x0:x1].astype(np.uint8) * 255
    head_crop_mask = candidate["head_mask"][y0:y1, x0:x1].astype(np.uint8) * 255
    overlap_crop_mask = np.zeros_like(tail_crop_mask, dtype=np.uint8)
    overlap_crop_mask[(overlap_mask[y0:y1, x0:x1] > 0) & (tail_crop_mask > 0)] = 255
    base = highlighted_crop_overlay(crop_rgb, tail_crop_mask, head_crop_mask, overlap_crop_mask)

    path_mask = candidate.get("tail_path_skeleton_mask")
    if isinstance(path_mask, np.ndarray) and np.any(path_mask):
        line = cv2.dilate(
            path_mask[y0:y1, x0:x1].astype(np.uint8) * 255,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            iterations=1,
        )
        base[line > 0] = np.array([0, 255, 70], dtype=np.uint8)

    competing = candidate.get("competing_path_skeleton_mask")
    if isinstance(competing, np.ndarray) and np.any(competing):
        line = cv2.dilate(
            competing[y0:y1, x0:x1].astype(np.uint8) * 255,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            iterations=1,
        )
        base[line > 0] = np.array([165, 165, 165], dtype=np.uint8)

    anchor_xy = candidate.get("head_anchor_xy") or candidate.get("anchor_xy")
    endpoint_xy = candidate.get("path_endpoint_xy")
    exit_xy = candidate.get("head_exit_xy")
    for point, color in (
        (anchor_xy, (255, 255, 255)),
        (exit_xy, (255, 160, 0)),
        (endpoint_xy, (0, 0, 255)),
    ):
        if point is None:
            continue
        px = int(point[0]) - x0
        py = int(point[1]) - y0
        if 0 <= px < base.shape[1] and 0 <= py < base.shape[0]:
            cv2.circle(base, (px, py), 5, color, thickness=-1)

    sx0, sy0, sx1, sy1 = [int(value) for value in candidate["source_bbox_xyxy"]]
    boundary = base.copy()
    cv2.rectangle(boundary, (sx0 - x0, sy0 - y0), (sx1 - x0 - 1, sy1 - y0 - 1), (255, 255, 255), 1)
    cv2.rectangle(boundary, (0, 0), (boundary.shape[1] - 1, boundary.shape[0] - 1), (255, 80, 80), 2)

    head_anchor_lines = [
        f"status={candidate.get('candidate_status')} head={candidate.get('head_label_id')} tail={candidate.get('tail_id')}",
        f"anchor={anchor_xy} dist={candidate.get('head_anchor_distance_px')}",
        f"exit={exit_xy} endpoint={endpoint_xy}",
    ]
    path_lines = [
        f"mode={candidate.get('path_mode')} norm={candidate.get('path_color_normalization')}",
        f"geom={candidate.get('path_score_geometry')} color={candidate.get('path_score_color')} hybrid={candidate.get('path_score_hybrid')}",
        f"raw={candidate.get('color_score_raw')} norm={candidate.get('color_score_normalized')} margin={candidate.get('path_score_margin')}",
    ]
    boundary_lines = [
        f"boundary_risk={candidate.get('crop_boundary_risk')} adaptive_pad={candidate.get('crop_adaptive_padding_px')}",
        f"endpoint_crop_edge={candidate.get('endpoint_to_crop_edge_px')} endpoint_source_edge={candidate.get('endpoint_to_source_bbox_edge_px')}",
        f"cutoff_stage={candidate.get('tail_cutoff_stage')}",
    ]
    duplicate_lines = [
        f"head_group={candidate.get('head_reuse_group_id')} n={candidate.get('head_reuse_candidate_count')} accepted={candidate.get('head_reuse_accepted_count')}",
        f"tail_group={candidate.get('tail_reuse_group_id')} n={candidate.get('tail_reuse_candidate_count')} accepted={candidate.get('tail_reuse_accepted_count')}",
        f"shared_path_fraction={candidate.get('shared_path_fraction')} duplicate_of={candidate.get('duplicate_of_crop_id')}",
    ]

    paths = {
        "diagnostic_head_anchor": output_root / "diagnostics/head_anchors" / f"{crop_id}_head_anchor.png",
        "diagnostic_path_ablation": output_root / "diagnostics/path_ablation" / f"{crop_id}_path_ablation.png",
        "diagnostic_crop_boundary": output_root / "diagnostics/crop_boundaries" / f"{crop_id}_crop_boundary.png",
        "diagnostic_duplicate_reuse": output_root / "diagnostics/duplicate_reuse" / f"{crop_id}_duplicate_reuse.png",
        "diagnostic_cutoff_stage": output_root / "diagnostics/cutoff_stage" / f"{crop_id}_cutoff_stage.png",
    }
    Image.fromarray(put_diagnostic_text(base, head_anchor_lines)).save(paths["diagnostic_head_anchor"])
    Image.fromarray(put_diagnostic_text(base, path_lines)).save(paths["diagnostic_path_ablation"])
    Image.fromarray(put_diagnostic_text(boundary, boundary_lines)).save(paths["diagnostic_crop_boundary"])
    Image.fromarray(put_diagnostic_text(base, duplicate_lines)).save(paths["diagnostic_duplicate_reuse"])
    Image.fromarray(put_diagnostic_text(base, boundary_lines + path_lines[:1])).save(paths["diagnostic_cutoff_stage"])
    return {key: str(value) for key, value in paths.items()}


def should_use_path_scoring_for_tail(
    tail: ComponentInfo,
    touching_head_ids: list[int],
    use_path_scoring: bool,
    fast_isolated: bool,
) -> bool:
    if not use_path_scoring:
        return False
    if not fast_isolated:
        return True
    if len(touching_head_ids) != 1:
        return True
    if bool(tail.has_overlap):
        return True

    endpoint_count = len(tail.endpoints_xy or [])
    branchpoint_count = len(tail.branchpoints_xy or [])
    simple_topology = endpoint_count <= 3 and branchpoint_count <= 1
    return not simple_topology


def build_head_connected_outputs(
    rgb: np.ndarray,
    stem: str,
    output_root: Path,
    tail_label_map: np.ndarray,
    head_mask: np.ndarray,
    overlap_mask: np.ndarray,
    tails: list[ComponentInfo],
    args: argparse.Namespace,
) -> dict[str, Any]:
    height, width = head_mask.shape
    scale = image_scale_factor(width, height)
    contact_radius = max(1, int(round(args.head_contact_radius * scale)))
    crop_padding = max(0, int(round(args.crop_padding * scale)))
    path_mode = str(getattr(args, "path_mode", "hybrid"))
    use_path_scoring = bool(getattr(args, "path_v2", False)) and path_mode != "legacy"
    normalization = str(getattr(args, "normalization", "raw"))
    record_ablation_scores = bool(getattr(args, "record_ablation_scores", False))
    write_tail_debug = bool(getattr(args, "tail_debug_overlays", False))
    tail_debug_limit = getattr(args, "tail_debug_limit", None)
    fast_isolated = bool(getattr(args, "path_fast_isolated", True))
    contact_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * contact_radius + 1, 2 * contact_radius + 1),
    )

    n_heads, head_labels, head_stats, _ = cv2.connectedComponentsWithStats(head_mask, connectivity=8)
    head_areas = {
        int(label_idx): int(head_stats[label_idx, cv2.CC_STAT_AREA])
        for label_idx in range(1, n_heads)
    }
    tail_width_baseline_px = estimate_average_tail_width_px(tails, scale)
    connected_tail_mask = np.zeros_like(head_mask, dtype=np.uint8)
    connected_head_mask = np.zeros_like(head_mask, dtype=np.uint8)
    crop_rows: list[dict[str, Any]] = []
    crop_candidate_rows: list[dict[str, Any]] = []
    crop_candidates: list[dict[str, Any]] = []
    ambiguous_tails: list[dict[str, Any]] = []
    split_tail_components: list[dict[str, Any]] = []
    tail_assignment_debug_overlays: list[str] = []
    rejected_tail_head_assignments = 0

    for tail in tails:
        tail_component = tail_label_map == tail.component_id
        if not np.any(tail_component):
            continue

        dilated_tail = cv2.dilate(
            tail_component.astype(np.uint8) * 255,
            contact_kernel,
            iterations=1,
        )
        touching_head_ids = sorted(
            int(label)
            for label in np.unique(head_labels[(dilated_tail > 0) & (head_labels > 0)])
        )
        if not touching_head_ids:
            continue

        connected_tail_mask[tail_component] = 255
        connected_head_mask[np.isin(head_labels, touching_head_ids)] = 255
        use_tail_path_scoring = should_use_path_scoring_for_tail(
            tail=tail,
            touching_head_ids=touching_head_ids,
            use_path_scoring=use_path_scoring,
            fast_isolated=fast_isolated,
        )

        assignment_plan = build_tail_assignment_plan(
            tail_component=tail_component,
            head_labels=head_labels,
            touching_head_ids=touching_head_ids,
            scale=scale,
            contact_radius=contact_radius,
            head_areas=head_areas,
            average_tail_width_px=tail_width_baseline_px,
            rgb=rgb,
            use_path_scoring=use_tail_path_scoring,
            path_mode=path_mode,
            normalization=normalization,
            record_ablation_scores=record_ablation_scores,
        )
        rejected_tail_head_assignments += len(assignment_plan.get("rejected_heads", []))

        debug_overlay_path: str | None = None
        if write_tail_debug and (
            tail_debug_limit is None or len(tail_assignment_debug_overlays) < int(tail_debug_limit)
        ):
            debug_path = output_root / "tail_assignment_debug" / (
                f"{stem}__tail_{tail.component_id:03d}_assignment.png"
            )
            Image.fromarray(
                tail_assignment_debug_overlay(
                    rgb=rgb,
                    tail_component=tail_component,
                    head_labels=head_labels,
                    touching_head_ids=touching_head_ids,
                    plan=assignment_plan,
                    tail=tail,
                )
            ).save(debug_path)
            debug_overlay_path = str(debug_path)
            tail_assignment_debug_overlays.append(debug_overlay_path)

        plan_json = assignment_plan_for_json(assignment_plan)
        if assignment_plan["mode"] == "split_shared_tail":
            split_tail_components.append(
                {
                    "tail_id": tail.component_id,
                    "assignment": plan_json,
                    "debug_overlay": debug_overlay_path,
                }
            )

        if assignment_plan["mode"] == "ambiguous":
            ambiguous_tails.append(
                {
                    "tail_id": tail.component_id,
                    "assignment": plan_json,
                    "debug_overlay": debug_overlay_path,
                }
            )
            continue

        for assignment in assignment_plan["assignments"]:
            head_id = int(assignment["head_id"])
            assigned_tail_component = assignment["mask"] > 0
            single_head_mask = head_labels == head_id
            sperm_mask = assigned_tail_component | single_head_mask
            source_bbox = bbox_from_mask(sperm_mask)
            if source_bbox is None:
                continue
            sx0, sy0, sx1, sy1 = source_bbox
            path_endpoint_xy = assignment.get("path_endpoint_xy")
            endpoint_to_source_bbox_edge_px = point_to_bbox_edge_distance(path_endpoint_xy, source_bbox)
            adaptive_padding = 0
            if (
                endpoint_to_source_bbox_edge_px is not None
                and endpoint_to_source_bbox_edge_px <= max(3.0 * scale, 4.0)
                and crop_padding > 0
            ):
                adaptive_padding = max(2, int(round(0.5 * crop_padding)))
            x0, y0, x1, y1 = pad_bbox(source_bbox, width, height, crop_padding + adaptive_padding)
            endpoint_to_crop_edge_px = point_to_bbox_edge_distance(path_endpoint_xy, (x0, y0, x1, y1))
            crop_boundary_risk = crop_boundary_risk_from_distance(endpoint_to_crop_edge_px, scale)
            crop_id = f"{stem}__tail_{tail.component_id:03d}__head_{head_id:03d}"

            tail_bbox = bbox_from_mask(assigned_tail_component)
            if tail_bbox is not None:
                tx0, ty0, tx1, ty1 = tail_bbox
                tail_roi = assigned_tail_component[ty0:ty1, tx0:tx1].astype(np.uint8) * 255
                computed_skeleton_pixels = int(cv2.countNonZero(skeletonize_binary(tail_roi)))
            else:
                computed_skeleton_pixels = 0
            assigned_skeleton_pixels = assignment["assigned_skeleton_pixels"]
            if assigned_skeleton_pixels is None:
                assigned_skeleton_pixels = computed_skeleton_pixels

            crop_head_labels = head_labels[y0:y1, x0:x1]
            crop_tail_labels = tail_label_map[y0:y1, x0:x1]
            crop_assigned_tail = assigned_tail_component[y0:y1, x0:x1]
            crop_single_head = single_head_mask[y0:y1, x0:x1]
            foreign_head_pixels = int(
                np.count_nonzero((crop_head_labels > 0) & (crop_head_labels != head_id))
            )
            foreign_tail_pixels = int(
                np.count_nonzero((crop_tail_labels > 0) & (~crop_assigned_tail))
            )
            candidate_pixels = int(
                np.count_nonzero(crop_assigned_tail) + np.count_nonzero(crop_single_head)
            )
            foreign_content_score = (
                float(foreign_head_pixels + foreign_tail_pixels) / float(candidate_pixels)
                if candidate_pixels > 0
                else 0.0
            )
            crop_area_pixels = max(1, int((x1 - x0) * (y1 - y0)))
            foreign_content_by_crop_area = float(foreign_head_pixels + foreign_tail_pixels) / float(crop_area_pixels)
            directional_width = assignment_plan.get("directional_width") or {}
            path_field_names = [
                "path_v2",
                "path_mode",
                "path_endpoint_xy",
                "path_length_px",
                "path_confidence_score",
                "path_score_geometry",
                "path_score_color",
                "path_score_hybrid",
                "path_score_margin",
                "path_score_margin_geometry",
                "path_score_margin_color",
                "path_score_margin_hybrid",
                "path_rank_geometry",
                "path_rank_color",
                "path_rank_hybrid",
                "path_low_margin",
                "competing_path_count",
                "path_color_continuity_score",
                "path_color_delta_mean",
                "path_color_delta_p75",
                "path_color_reference_points",
                "path_color_normalization",
                "color_score_raw",
                "color_score_normalized",
                "color_delta_p75_raw",
                "color_delta_p75_normalized",
                "path_curvature_total",
                "path_branchpoint_count",
                "path_width_spike_ratio",
                "path_trimmed_branch_pixels",
                "path_trimmed_branch_fraction",
                "tail_pixels_inside_head_collar",
                "tail_pixels_outside_head_collar",
                "head_exit_count",
                "head_exit_path_length_px",
                "head_exit_xy",
                "valid_tail_continuation",
                "path_rejection_reason",
                "path_ambiguity_reason",
                "path_conflict_fraction",
            ]
            path_fields = {
                name: assignment.get(name)
                for name in path_field_names
                if name in assignment
            }

            crop_candidates.append(
                {
                    "image_stem": stem,
                    "crop_id": crop_id,
                    "tail_id": tail.component_id,
                    "head_label_id": head_id,
                    "bbox_xyxy": [x0, y0, x1, y1],
                    "source_bbox_xyxy": [sx0, sy0, sx1, sy1],
                    "tail_mask": assigned_tail_component.copy(),
                    "head_mask": single_head_mask.copy(),
                    "tail_has_overlap": bool(tail.has_overlap),
                    "tail_overlap_reasons": tail.overlap_reasons or [],
                    "tail_endpoint_count": len(tail.endpoints_xy or []),
                    "tail_branchpoint_count": len(tail.branchpoints_xy or []),
                    "tail_width_p95_px": tail.width_p95_px,
                    "tail_width_median_px": tail.width_median_px,
                    "tail_assignment_mode": assignment_plan["mode"],
                    "assigned_tail_pixels": int(assignment["assigned_tail_pixels"]),
                    "assigned_skeleton_pixels": int(assigned_skeleton_pixels),
                    "shared_tail_pixels": int(assignment["shared_tail_pixels"]),
                    "touching_head_count": int(assignment["touching_head_count"]),
                    "touching_head_ids": touching_head_ids,
                    "anchor_xy": assignment["anchor_xy"],
                    "anchor_distance_px": assignment["anchor_distance_px"],
                    "head_anchor_xy": assignment.get("head_anchor_xy", assignment["anchor_xy"]),
                    "head_anchor_distance_px": assignment.get(
                        "head_anchor_distance_px", assignment["anchor_distance_px"]
                    ),
                    "width_spike_split": bool(assignment.get("width_spike_split", False)),
                    "width_barrier_used": bool(
                        directional_width.get("used_as_cut_barrier", False)
                    ),
                    "width_barrier_fraction": directional_width.get("barrier_fraction"),
                    "tail_path_skeleton_mask": assignment.get("path_skeleton_mask"),
                    "head_collar_mask": assignment.get("head_collar_mask"),
                    "competing_path_skeleton_mask": assignment.get(
                        "competing_path_skeleton_mask"
                    ),
                    **path_fields,
                    "foreign_head_pixels": foreign_head_pixels,
                    "foreign_tail_pixels": foreign_tail_pixels,
                    "foreign_content_score": foreign_content_score,
                    "foreign_content_by_crop_area": foreign_content_by_crop_area,
                    "endpoint_to_source_bbox_edge_px": endpoint_to_source_bbox_edge_px,
                    "endpoint_to_crop_edge_px": endpoint_to_crop_edge_px,
                    "crop_boundary_risk": crop_boundary_risk,
                    "crop_adaptive_padding_px": adaptive_padding,
                    "source_bbox_touches_image_border": (
                        sx0 <= 0 or sy0 <= 0 or sx1 >= width or sy1 >= height
                    ),
                    "crop_bbox_touches_image_border": (
                        x0 <= 0 or y0 <= 0 or x1 >= width or y1 >= height
                    ),
                }
            )

    arbitration = arbitrate_crop_candidates(
        crop_candidates,
        scale,
        acceptance_profile=str(getattr(args, "acceptance_profile", "conservative")),
    )
    diagnostics_written = 0
    diagnostic_limit = getattr(args, "diagnostic_limit", None)
    for candidate in crop_candidates:
        artifact_paths = write_crop_candidate_artifacts(candidate, rgb, overlap_mask, output_root)
        candidate.update(artifact_paths)
        if bool(getattr(args, "diagnostics", False)) and (
            diagnostic_limit is None or diagnostics_written < int(diagnostic_limit)
        ):
            candidate.update(write_candidate_diagnostics(candidate, rgb, overlap_mask, output_root))
            diagnostics_written += 1
        public_row = crop_candidate_for_json(candidate)
        crop_candidate_rows.append(public_row)
        if candidate["candidate_status"] == "accepted":
            crop_rows.append(public_row)

    connected_overlap_mask = np.zeros_like(overlap_mask, dtype=np.uint8)
    connected_overlap_mask[(overlap_mask > 0) & (connected_tail_mask > 0)] = 255
    head_connected_overlay_path = output_root / "head_connected_overlays" / f"{stem}_head_connected_overlay.png"
    Image.fromarray(
        overlay_masks(rgb, connected_tail_mask, connected_head_mask, connected_overlap_mask)
    ).save(head_connected_overlay_path)

    return {
        "head_connected_overlay": str(head_connected_overlay_path),
        "head_connected_tail_pixels": int(np.count_nonzero(connected_tail_mask)),
        "head_connected_head_pixels": int(np.count_nonzero(connected_head_mask)),
        "head_connected_overlap_pixels": int(np.count_nonzero(connected_overlap_mask)),
        "head_connected_tail_components": int(len(set(row["tail_id"] for row in crop_rows))),
        "individual_sperm_crops": len(crop_rows),
        "crop_candidates": len(crop_candidate_rows),
        "accepted_crop_candidates": sum(
            1 for row in crop_candidate_rows if row["candidate_status"] == "accepted"
        ),
        "rejected_crop_candidates": sum(
            1 for row in crop_candidate_rows if row["candidate_status"] == "rejected"
        ),
        "ambiguous_crop_candidates": sum(
            1 for row in crop_candidate_rows if row["candidate_status"] == "ambiguous"
        ),
        "duplicate_rejected_crop_candidates": arbitration["duplicate_rejections"],
        "tail_duplicate_ambiguous_candidates": arbitration["tail_duplicate_ambiguous"],
        "accepted_head_reuse_count": arbitration["accepted_head_reuse_count"],
        "accepted_tail_reuse_count": arbitration["accepted_tail_reuse_count"],
        "crop_candidate_thresholds": arbitration["thresholds"],
        "candidate_heads_available": int(n_heads - 1),
        "split_tail_components_count": len(split_tail_components),
        "ambiguous_tail_components": len(ambiguous_tails),
        "rejected_tail_head_assignments": rejected_tail_head_assignments,
        "candidate_diagnostics_written": diagnostics_written,
        "tail_width_baseline_px": tail_width_baseline_px,
        "split_tail_components": split_tail_components,
        "ambiguous_tails": ambiguous_tails,
        "tail_assignment_debug_overlays": tail_assignment_debug_overlays,
        "crops": crop_rows,
        "crop_candidates_detail": crop_candidate_rows,
    }


def prepare_output(output_root: Path, overwrite: bool) -> None:
    if output_root.exists():
        if overwrite:
            shutil.rmtree(output_root)
        elif any(output_root.iterdir()):
            raise RuntimeError(f"{output_root} already exists and is not empty. Use --overwrite.")
    for child in (
        "overlays",
        "masks",
        "json",
        "intermediates",
        "head_connected_overlays",
        "sperm_crops",
        "highlighted_sperm_crops",
        "tail_skeletons",
        "tail_assignment_debug",
        "candidate_review/rejected",
        "candidate_review/ambiguous",
        "diagnostics/head_anchors",
        "diagnostics/path_ablation",
        "diagnostics/crop_boundaries",
        "diagnostics/duplicate_reuse",
        "diagnostics/cutoff_stage",
    ):
        (output_root / child).mkdir(parents=True, exist_ok=True)


def process_image(
    image_path: Path,
    input_root: Path,
    output_root: Path,
    settings: dict[str, float | int],
    args: argparse.Namespace,
) -> dict[str, Any]:
    rgb = load_rgb(image_path)
    height, width = rgb.shape[:2]
    raw_brown_mask, brown_mask, local_dark_mask = build_candidate_masks(rgb, settings)
    head_mask, heads = detect_heads(brown_mask, rgb, settings)
    tail_mask, overlap_mask, tail_label_map, tails = detect_tails(
        brown_mask,
        head_mask,
        heads,
        settings,
        max_head_distance=args.max_head_distance,
    )

    stem = safe_stem(image_path, input_root)
    mask_dir = output_root / "masks"
    overlay_path = output_root / "overlays" / f"{stem}_overlay.png"
    json_path = output_root / "json" / f"{stem}.json"

    brown_path = mask_dir / f"{stem}_brown_mask.png"
    tail_path = mask_dir / f"{stem}_tail_mask.png"
    head_path = mask_dir / f"{stem}_head_mask.png"
    overlap_path = mask_dir / f"{stem}_overlap_mask.png"

    write_mask(brown_path, brown_mask)
    write_mask(tail_path, tail_mask)
    write_mask(head_path, head_mask)
    write_mask(overlap_path, overlap_mask)
    Image.fromarray(overlay_masks(rgb, tail_mask, head_mask, overlap_mask)).save(overlay_path)

    intermediate_paths: dict[str, str] = {}
    if args.save_intermediates:
        raw_path = output_root / "intermediates" / f"{stem}_raw_brown_candidate_mask.png"
        local_path = output_root / "intermediates" / f"{stem}_local_dark_mask.png"
        write_mask(raw_path, raw_brown_mask)
        write_mask(local_path, local_dark_mask)
        intermediate_paths = {
            "raw_brown_candidate_mask": str(raw_path),
            "local_dark_mask": str(local_path),
        }

    downstream_outputs = build_head_connected_outputs(
        rgb=rgb,
        stem=stem,
        output_root=output_root,
        tail_label_map=tail_label_map,
        head_mask=head_mask,
        overlap_mask=overlap_mask,
        tails=tails,
        args=args,
    )

    image_summary = {
        "image": str(image_path),
        "relative_image": str(image_path.relative_to(input_root)),
        "width": width,
        "height": height,
        "mode": args.mode,
        "parameters": {
            "settings": settings,
            "max_head_distance": args.max_head_distance,
            "path_v2": bool(getattr(args, "path_v2", False)),
            "path_mode": str(getattr(args, "path_mode", "hybrid")),
            "normalization": str(getattr(args, "normalization", "raw")),
            "acceptance_profile": str(getattr(args, "acceptance_profile", "conservative")),
            "diagnostics": bool(getattr(args, "diagnostics", False)),
        },
        "counts": {
            "brown_pixels": int(np.count_nonzero(brown_mask)),
            "tail_pixels": int(np.count_nonzero(tail_mask)),
            "head_pixels": int(np.count_nonzero(head_mask)),
            "overlap_pixels": int(np.count_nonzero(overlap_mask)),
            "tail_components": len(tails),
            "head_components": len(heads),
            "overlap_components": sum(1 for tail in tails if tail.has_overlap),
            "tail_components_with_head_match": sum(
                1
                for tail in tails
                if tail.head_matches
                and tail.head_matches.get("proximal") is not None
            ),
            "head_connected_tail_components": downstream_outputs[
                "head_connected_tail_components"
            ],
            "individual_sperm_crops": downstream_outputs["individual_sperm_crops"],
            "crop_candidates": downstream_outputs["crop_candidates"],
            "accepted_crop_candidates": downstream_outputs["accepted_crop_candidates"],
            "rejected_crop_candidates": downstream_outputs["rejected_crop_candidates"],
            "ambiguous_crop_candidates": downstream_outputs["ambiguous_crop_candidates"],
            "duplicate_rejected_crop_candidates": downstream_outputs[
                "duplicate_rejected_crop_candidates"
            ],
            "tail_duplicate_ambiguous_candidates": downstream_outputs[
                "tail_duplicate_ambiguous_candidates"
            ],
            "accepted_head_reuse_count": downstream_outputs["accepted_head_reuse_count"],
            "accepted_tail_reuse_count": downstream_outputs["accepted_tail_reuse_count"],
            "split_tail_components": downstream_outputs["split_tail_components_count"],
            "ambiguous_tail_components": downstream_outputs["ambiguous_tail_components"],
            "rejected_tail_head_assignments": downstream_outputs[
                "rejected_tail_head_assignments"
            ],
            "candidate_diagnostics_written": downstream_outputs["candidate_diagnostics_written"],
        },
        "outputs": {
            "overlay": str(overlay_path),
            "brown_mask": str(brown_path),
            "tail_mask": str(tail_path),
            "head_mask": str(head_path),
            "overlap_mask": str(overlap_path),
            "intermediates": intermediate_paths,
            "head_connected": {
                "overlay": downstream_outputs["head_connected_overlay"],
                "sperm_crops": [row["crop"] for row in downstream_outputs["crops"]],
                "highlighted_sperm_crops": [
                    row["highlighted_crop"] for row in downstream_outputs["crops"]
                ],
                "tail_skeleton_masks": [
                    row["skeleton_mask"] for row in downstream_outputs["crops"]
                ],
                "tail_skeleton_overlays": [
                    row["skeleton_overlay"] for row in downstream_outputs["crops"]
                ],
                "rejected_candidate_crops": [
                    row["highlighted_crop"]
                    for row in downstream_outputs["crop_candidates_detail"]
                    if row["candidate_status"] == "rejected"
                ],
                "ambiguous_candidate_crops": [
                    row["highlighted_crop"]
                    for row in downstream_outputs["crop_candidates_detail"]
                    if row["candidate_status"] == "ambiguous"
                ],
                "tail_assignment_debug_overlays": downstream_outputs[
                    "tail_assignment_debug_overlays"
                ],
            },
        },
        "head_connected_postprocessing": downstream_outputs,
        "heads": [asdict(head) for head in heads],
        "tails": [asdict(tail) for tail in tails],
    }

    json_path.write_text(json.dumps(image_summary, indent=2), encoding="utf-8")
    return {
        "image": str(image_path.relative_to(input_root)),
        "width": width,
        "height": height,
        "brown_pixels": image_summary["counts"]["brown_pixels"],
        "tail_pixels": image_summary["counts"]["tail_pixels"],
        "head_pixels": image_summary["counts"]["head_pixels"],
        "overlap_pixels": image_summary["counts"]["overlap_pixels"],
        "tail_components": image_summary["counts"]["tail_components"],
        "head_components": image_summary["counts"]["head_components"],
        "overlap_components": image_summary["counts"]["overlap_components"],
        "tail_components_with_head_match": image_summary["counts"]["tail_components_with_head_match"],
        "head_connected_tail_components": image_summary["counts"][
            "head_connected_tail_components"
        ],
        "individual_sperm_crops": image_summary["counts"]["individual_sperm_crops"],
        "crop_candidates": image_summary["counts"]["crop_candidates"],
        "accepted_crop_candidates": image_summary["counts"]["accepted_crop_candidates"],
        "rejected_crop_candidates": image_summary["counts"]["rejected_crop_candidates"],
        "ambiguous_crop_candidates": image_summary["counts"]["ambiguous_crop_candidates"],
        "duplicate_rejected_crop_candidates": image_summary["counts"][
            "duplicate_rejected_crop_candidates"
        ],
        "tail_duplicate_ambiguous_candidates": image_summary["counts"][
            "tail_duplicate_ambiguous_candidates"
        ],
        "accepted_head_reuse_count": image_summary["counts"]["accepted_head_reuse_count"],
        "accepted_tail_reuse_count": image_summary["counts"]["accepted_tail_reuse_count"],
        "split_tail_components": image_summary["counts"]["split_tail_components"],
        "ambiguous_tail_components": image_summary["counts"]["ambiguous_tail_components"],
        "rejected_tail_head_assignments": image_summary["counts"][
            "rejected_tail_head_assignments"
        ],
        "candidate_diagnostics_written": image_summary["counts"]["candidate_diagnostics_written"],
        "overlay": str(overlay_path),
        "head_connected_overlay": downstream_outputs["head_connected_overlay"],
        "json": str(json_path),
    }


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "image",
        "width",
        "height",
        "brown_pixels",
        "tail_pixels",
        "head_pixels",
        "overlap_pixels",
        "tail_components",
        "head_components",
        "overlap_components",
        "tail_components_with_head_match",
        "head_connected_tail_components",
        "individual_sperm_crops",
        "crop_candidates",
        "accepted_crop_candidates",
        "rejected_crop_candidates",
        "ambiguous_crop_candidates",
        "duplicate_rejected_crop_candidates",
        "tail_duplicate_ambiguous_candidates",
        "accepted_head_reuse_count",
        "accepted_tail_reuse_count",
        "split_tail_components",
        "ambiguous_tail_components",
        "rejected_tail_head_assignments",
        "candidate_diagnostics_written",
        "overlay",
        "head_connected_overlay",
        "json",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.path_mode == "legacy":
        args.path_v2 = False
    if args.diagnostic_limit is not None and args.diagnostic_limit < 0:
        raise ValueError("--diagnostic-limit must be non-negative when provided")
    input_root = resolve_poc_path(args.input_root)
    output_root = resolve_poc_path(args.output_root)
    if not input_root.exists():
        raise FileNotFoundError(f"Input root not found: {input_root}")

    settings = mode_settings(args.mode, args.min_component_area)
    images = collect_images(input_root, args.limit)
    if not images:
        raise RuntimeError(f"No real image files found under {input_root}")

    total_real_images = valid_image_count(input_root)
    prepare_output(output_root, args.overwrite)

    rows: list[dict[str, Any]] = []
    for index, image_path in enumerate(images, start=1):
        print(f"[{index}/{len(images)}] {image_path.relative_to(input_root)}", flush=True)
        rows.append(process_image(image_path, input_root, output_root, settings, args))

    summary_csv = output_root / "summary.csv"
    write_summary_csv(summary_csv, rows)
    summary = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "mode": args.mode,
        "total_real_images_available": total_real_images,
        "processed_images": len(rows),
        "skipped_sidecars_and_macosx": True,
        "settings": settings,
        "summary_csv": str(summary_csv),
        "totals": {
            "tail_components": sum(int(row["tail_components"]) for row in rows),
            "head_components": sum(int(row["head_components"]) for row in rows),
            "overlap_components": sum(int(row["overlap_components"]) for row in rows),
            "tail_components_with_head_match": sum(
                int(row["tail_components_with_head_match"]) for row in rows
            ),
            "head_connected_tail_components": sum(
                int(row["head_connected_tail_components"]) for row in rows
            ),
            "individual_sperm_crops": sum(int(row["individual_sperm_crops"]) for row in rows),
            "crop_candidates": sum(int(row["crop_candidates"]) for row in rows),
            "accepted_crop_candidates": sum(
                int(row["accepted_crop_candidates"]) for row in rows
            ),
            "rejected_crop_candidates": sum(
                int(row["rejected_crop_candidates"]) for row in rows
            ),
            "ambiguous_crop_candidates": sum(
                int(row["ambiguous_crop_candidates"]) for row in rows
            ),
            "duplicate_rejected_crop_candidates": sum(
                int(row["duplicate_rejected_crop_candidates"]) for row in rows
            ),
            "tail_duplicate_ambiguous_candidates": sum(
                int(row["tail_duplicate_ambiguous_candidates"]) for row in rows
            ),
            "accepted_head_reuse_count": sum(
                int(row["accepted_head_reuse_count"]) for row in rows
            ),
            "accepted_tail_reuse_count": sum(
                int(row["accepted_tail_reuse_count"]) for row in rows
            ),
            "split_tail_components": sum(int(row["split_tail_components"]) for row in rows),
            "ambiguous_tail_components": sum(
                int(row["ambiguous_tail_components"]) for row in rows
            ),
            "rejected_tail_head_assignments": sum(
                int(row["rejected_tail_head_assignments"]) for row in rows
            ),
            "candidate_diagnostics_written": sum(
                int(row["candidate_diagnostics_written"]) for row in rows
            ),
        },
    }
    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
