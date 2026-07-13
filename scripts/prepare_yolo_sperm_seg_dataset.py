#!/usr/bin/env python3
"""Prepare a YOLOv8 segmentation dataset from cropped sperm positives.

The human positives do not include hand-authored masks. This builder creates
one weak pseudo-mask per crop by extracting the dominant foreground contour.
Those masks are intended for baseline instance-segmentation experiments, not
as final ground truth.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError


IMAGE_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}


@dataclass(frozen=True)
class Sample:
    source: str
    path: Path
    category: str


@dataclass(frozen=True)
class PseudoMask:
    polygon: np.ndarray
    mask_area: int
    method: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a YOLOv8 segmentation dataset from sperm crop positives"
    )
    parser.add_argument(
        "--include",
        choices=["human"],
        default="human",
        help="Input component to include. Only human positives are supported for this baseline.",
    )
    parser.add_argument(
        "--human-root",
        type=Path,
        default=Path("Training_Data/Human_Positives/unpacked_25621500/extracted"),
        help="Root containing extracted human positive cropped sperm images",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("datasets/yolo_sperm_seg_human_pseudo"),
        help="Output YOLO segmentation dataset root",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of valid samples to use, for smoke tests",
    )
    parser.add_argument(
        "--min-mask-area",
        type=int,
        default=350,
        help="Minimum connected-component area to accept before using fallback polygon",
    )
    parser.add_argument(
        "--foreground-percentile",
        type=float,
        default=90.0,
        help="Local-contrast percentile used to seed the foreground pseudo-mask",
    )
    parser.add_argument(
        "--polygon-epsilon",
        type=float,
        default=0.003,
        help="Contour approximation epsilon as a fraction of contour perimeter",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=180,
        help="Maximum polygon points written to each YOLO segmentation label",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete an existing output folder before writing",
    )
    return parser.parse_args()


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTS


def is_valid_image(path: Path) -> bool:
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except (UnidentifiedImageError, OSError):
        return False


def collect_human_samples(root: Path, limit: int | None) -> list[Sample]:
    samples: list[Sample] = []
    for path in sorted(root.rglob("*")):
        if not is_image_file(path) or not is_valid_image(path):
            continue
        category = str(path.relative_to(root).parent)
        samples.append(Sample(source="human", path=path, category=category))
        if limit is not None and len(samples) >= limit:
            break
    return samples


def split_samples(
    samples: list[Sample],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> dict[str, list[Sample]]:
    if not samples:
        raise ValueError("No samples were provided for splitting")

    rng = random.Random(seed)
    shuffled = samples.copy()
    rng.shuffle(shuffled)

    n_total = len(shuffled)
    if n_total < 3:
        return {"train": shuffled, "val": [], "test": []}

    n_train = max(1, int(n_total * train_ratio))
    n_val = max(1, int(n_total * val_ratio))
    if n_train + n_val >= n_total:
        n_train = max(1, n_total - 2)
        n_val = 1

    train = shuffled[:n_train]
    val = shuffled[n_train : n_train + n_val]
    test = shuffled[n_train + n_val :]
    return {"train": train, "val": val, "test": test}


def ensure_output_dirs(root: Path, overwrite: bool) -> None:
    if root.exists():
        if overwrite:
            shutil.rmtree(root)
        elif any(root.iterdir()):
            raise RuntimeError(f"{root} already exists and is not empty. Use --overwrite.")

    for split in ("train", "val", "test"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)


def normalize_to_u8(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

    if arr.dtype == np.uint8:
        return arr

    arr_float = arr.astype(np.float32)
    lo, hi = np.percentile(arr_float, [1, 99])
    if hi <= lo:
        lo = float(np.min(arr_float))
        hi = float(np.max(arr_float))
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8)

    scaled = (arr_float - lo) * (255.0 / (hi - lo))
    return np.clip(scaled, 0, 255).astype(np.uint8)


def load_normalized_gray(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        return normalize_to_u8(np.asarray(img))


def bbox_polygon(width: int, height: int, inset: float = 0.02) -> np.ndarray:
    x0 = max(0.0, inset * width)
    y0 = max(0.0, inset * height)
    x1 = min(float(width - 1), (1.0 - inset) * width)
    y1 = min(float(height - 1), (1.0 - inset) * height)
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)


def largest_foreground_component(mask: np.ndarray, min_area: int) -> tuple[np.ndarray | None, int]:
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n_labels <= 1:
        return None, 0

    height, width = mask.shape
    center = np.array([width / 2.0, height / 2.0], dtype=np.float32)
    max_dist = max(1.0, float(np.hypot(width, height)) / 2.0)

    best_label = None
    best_score = -1.0
    best_area = 0
    for label_idx in range(1, n_labels):
        area = int(stats[label_idx, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        comp_width = int(stats[label_idx, cv2.CC_STAT_WIDTH])
        comp_height = int(stats[label_idx, cv2.CC_STAT_HEIGHT])
        aspect = max(comp_width, comp_height) / max(1, min(comp_width, comp_height))
        centroid = np.array(centroids[label_idx], dtype=np.float32)
        center_bonus = max(0.0, 1.0 - float(np.linalg.norm(centroid - center)) / max_dist)
        elongation_bonus = min(aspect, 12.0) / 12.0
        score = area * (1.0 + 0.35 * center_bonus + 0.75 * elongation_bonus)
        if score > best_score:
            best_score = score
            best_label = label_idx
            best_area = area

    if best_label is None:
        return None, 0

    component = np.zeros_like(mask, dtype=np.uint8)
    component[labels == best_label] = 255
    return component, best_area


def contour_to_polygon(
    component: np.ndarray,
    width: int,
    height: int,
    epsilon_fraction: float,
    max_points: int,
) -> np.ndarray | None:
    contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) <= 0:
        return None

    perimeter = cv2.arcLength(contour, closed=True)
    epsilon = max(1.0, epsilon_fraction * perimeter)
    approx = cv2.approxPolyDP(contour, epsilon, closed=True)

    for _ in range(12):
        if len(approx) <= max_points:
            break
        epsilon *= 1.35
        approx = cv2.approxPolyDP(contour, epsilon, closed=True)

    points = approx.reshape(-1, 2).astype(np.float32)
    if len(points) < 3:
        return None

    points[:, 0] = np.clip(points[:, 0], 0, width - 1)
    points[:, 1] = np.clip(points[:, 1], 0, height - 1)
    return points


def build_pseudo_mask(
    gray: np.ndarray,
    min_area: int,
    foreground_percentile: float,
    epsilon_fraction: float,
    max_points: int,
) -> PseudoMask:
    height, width = gray.shape
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    sigma = max(15.0, min(width, height) / 18.0)
    background = cv2.GaussianBlur(blur, (0, 0), sigmaX=sigma, sigmaY=sigma)
    contrast = normalize_to_u8(cv2.absdiff(blur, background))

    cutoff = int(np.percentile(contrast, foreground_percentile))
    _, mask = cv2.threshold(contrast, cutoff, 255, cv2.THRESH_BINARY)
    kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel3, iterations=1)
    mask = cv2.dilate(mask, kernel3, iterations=1)

    component, area = largest_foreground_component(mask, min_area=min_area)
    if component is None:
        return PseudoMask(
            polygon=bbox_polygon(width, height),
            mask_area=0,
            method="full_crop_fallback_no_component",
        )

    component = cv2.morphologyEx(component, cv2.MORPH_CLOSE, kernel5, iterations=1)
    filled = np.zeros_like(component, dtype=np.uint8)
    contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)

    polygon = contour_to_polygon(
        filled,
        width=width,
        height=height,
        epsilon_fraction=epsilon_fraction,
        max_points=max_points,
    )
    if polygon is None:
        return PseudoMask(
            polygon=bbox_polygon(width, height),
            mask_area=area,
            method="full_crop_fallback_no_polygon",
        )

    return PseudoMask(polygon=polygon, mask_area=area, method="foreground_contour")


def normalize_polygon(points: np.ndarray, width: int, height: int) -> list[float]:
    normalized: list[float] = []
    for x, y in points:
        normalized.append(float(np.clip(x / max(1, width), 0.0, 1.0)))
        normalized.append(float(np.clip(y / max(1, height), 0.0, 1.0)))
    return normalized


def write_label(path: Path, points: np.ndarray, width: int, height: int) -> None:
    normalized = normalize_polygon(points, width=width, height=height)
    values = " ".join(f"{value:.6f}" for value in normalized)
    path.write_text(f"0 {values}\n", encoding="utf-8")


def write_data_yaml(out_root: Path) -> Path:
    data_yaml = out_root / "data.yaml"
    data_yaml.write_text(
        (
            f"path: {out_root.resolve()}\n"
            "train: images/train\n"
            "val: images/val\n"
            "test: images/test\n"
            "names:\n"
            "  0: full_sperm\n"
        ),
        encoding="utf-8",
    )
    return data_yaml


def build_dataset(
    out_root: Path,
    split_map: dict[str, list[Sample]],
    min_area: int,
    foreground_percentile: float,
    epsilon_fraction: float,
    max_points: int,
) -> tuple[dict[str, dict[str, int]], list[dict[str, Any]]]:
    stats: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    rows: list[dict[str, Any]] = []

    for split, samples in split_map.items():
        for index, sample in enumerate(samples, start=1):
            stem = f"{sample.source}_{index:06d}"
            image_path = out_root / "images" / split / f"{stem}.png"
            label_path = out_root / "labels" / split / f"{stem}.txt"

            gray = load_normalized_gray(sample.path)
            height, width = gray.shape
            mask = build_pseudo_mask(
                gray,
                min_area=min_area,
                foreground_percentile=foreground_percentile,
                epsilon_fraction=epsilon_fraction,
                max_points=max_points,
            )

            rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
            Image.fromarray(rgb).save(image_path, format="PNG")
            write_label(label_path, mask.polygon, width=width, height=height)

            stats[split]["total_images"] += 1
            stats[split][f"{sample.source}_images"] += 1
            stats[split][mask.method] += 1

            rows.append(
                {
                    "split": split,
                    "source": sample.source,
                    "category": sample.category,
                    "source_image": str(sample.path),
                    "output_image": str(image_path),
                    "output_label": str(label_path),
                    "width": width,
                    "height": height,
                    "mask_area": mask.mask_area,
                    "polygon_points": int(len(mask.polygon)),
                    "mask_method": mask.method,
                }
            )

    return {key: dict(value) for key, value in stats.items()}, rows


def write_manifest(out_root: Path, rows: list[dict[str, Any]]) -> Path:
    manifest_path = out_root / "manifest.csv"
    fieldnames = [
        "split",
        "source",
        "category",
        "source_image",
        "output_image",
        "output_label",
        "width",
        "height",
        "mask_area",
        "polygon_points",
        "mask_method",
    ]
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return manifest_path


def main() -> None:
    args = parse_args()
    if not (0 < args.train_ratio < 1):
        raise ValueError("--train-ratio must be in (0, 1)")
    if not (0 <= args.val_ratio < 1):
        raise ValueError("--val-ratio must be in [0, 1)")
    if args.train_ratio + args.val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be < 1")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive when provided")
    if args.max_points < 3:
        raise ValueError("--max-points must be at least 3")
    if not (0 < args.foreground_percentile < 100):
        raise ValueError("--foreground-percentile must be in (0, 100)")

    samples = collect_human_samples(args.human_root, limit=args.limit)
    if not samples:
        raise RuntimeError(f"No valid human images found under {args.human_root}")

    split_map = split_samples(samples, args.train_ratio, args.val_ratio, args.seed)
    ensure_output_dirs(args.out_root, args.overwrite)
    stats, manifest_rows = build_dataset(
        out_root=args.out_root,
        split_map=split_map,
        min_area=args.min_mask_area,
        foreground_percentile=args.foreground_percentile,
        epsilon_fraction=args.polygon_epsilon,
        max_points=args.max_points,
    )
    data_yaml = write_data_yaml(args.out_root)
    manifest_path = write_manifest(args.out_root, manifest_rows)

    categories: dict[str, int] = defaultdict(int)
    methods: dict[str, int] = defaultdict(int)
    for row in manifest_rows:
        categories[str(row["category"])] += 1
        methods[str(row["mask_method"])] += 1

    summary = {
        "input_counts": {"human": len(samples)},
        "splits": stats,
        "categories": dict(sorted(categories.items())),
        "mask_methods": dict(sorted(methods.items())),
        "settings": {
            "include": args.include,
            "human_root": str(args.human_root),
            "seed": args.seed,
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "limit": args.limit,
            "min_mask_area": args.min_mask_area,
            "foreground_percentile": args.foreground_percentile,
            "polygon_epsilon": args.polygon_epsilon,
            "max_points": args.max_points,
        },
        "data_yaml": str(data_yaml),
        "manifest": str(manifest_path),
    }

    summary_path = args.out_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
