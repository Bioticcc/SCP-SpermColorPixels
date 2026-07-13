#!/usr/bin/env python3
"""Prepare a weakly labeled YOLOv8 dataset for full-sperm detection.

This script builds a 1-class detection dataset from cropped sperm positives:
- human positives (clinically labeled cropped sperm)
- mouse positives exported from NMA single-cell images

Each image receives a single weak bounding box that spans almost the full image.
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

from PIL import Image, UnidentifiedImageError

IMAGE_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}


@dataclass(frozen=True)
class Sample:
    source: str
    path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build YOLO dataset from sperm positive crops")
    parser.add_argument(
        "--human-root",
        type=Path,
        default=Path("Training_Data/Human_Positives/unpacked_25621500/extracted"),
        help="Root containing extracted human positive cropped images",
    )
    parser.add_argument(
        "--mouse-root",
        type=Path,
        default=Path("Training_Data/Mouse_Positives_Cropped_Tail"),
        help="Root containing mouse positive images (NMA exports)",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("datasets/yolo_full_sperm_no_annotations"),
        help="Output dataset root",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument(
        "--mouse-oversample",
        type=int,
        default=8,
        help="Number of train-time copies per mouse image (1 means no oversampling)",
    )
    parser.add_argument(
        "--bbox-scale",
        type=float,
        default=0.98,
        help="Normalized bbox width/height for weak labels",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing output folder before writing",
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


def collect_human_samples(root: Path) -> list[Sample]:
    samples: list[Sample] = []
    for path in root.rglob("*"):
        if is_image_file(path) and is_valid_image(path):
            samples.append(Sample(source="human", path=path))
    return samples


def collect_mouse_samples(root: Path) -> list[Sample]:
    samples: list[Sample] = []
    for path in root.rglob("*"):
        if not is_image_file(path):
            continue

        path_str = str(path)
        if "NMS_Software" in path_str:
            continue
        if "/annotations/" in path_str or "\\annotations\\" in path_str:
            continue

        if is_valid_image(path):
            samples.append(Sample(source="mouse", path=path))
    return samples


def split_samples(
    samples: list[Sample],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> dict[str, list[Sample]]:
    rng = random.Random(seed)
    shuffled = samples.copy()
    rng.shuffle(shuffled)

    n_total = len(shuffled)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    n_test = n_total - n_train - n_val

    train = shuffled[:n_train]
    val = shuffled[n_train : n_train + n_val]
    test = shuffled[n_train + n_val : n_train + n_val + n_test]

    return {"train": train, "val": val, "test": test}


def ensure_output_dirs(root: Path, overwrite: bool) -> None:
    if root.exists() and overwrite:
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)


def write_yolo_dataset(
    out_root: Path,
    split_map: dict[str, list[Sample]],
    mouse_oversample: int,
    bbox_scale: float,
) -> dict[str, dict[str, int]]:
    stats = defaultdict(lambda: defaultdict(int))
    manifest_rows: list[dict[str, str]] = []

    bbox_line = f"0 0.5 0.5 {bbox_scale:.6f} {bbox_scale:.6f}\n"

    for split, samples in split_map.items():
        per_source_counter: dict[str, int] = defaultdict(int)

        expanded: list[tuple[Sample, int]] = []
        for sample in samples:
            n_copies = mouse_oversample if split == "train" and sample.source == "mouse" else 1
            n_copies = max(1, int(n_copies))
            for copy_idx in range(n_copies):
                expanded.append((sample, copy_idx))

        for sample, copy_idx in expanded:
            per_source_counter[sample.source] += 1
            idx = per_source_counter[sample.source]

            stem = f"{sample.source}_{idx:06d}"
            if copy_idx > 0:
                stem = f"{stem}_dup{copy_idx:02d}"

            image_name = f"{stem}.png"
            label_name = f"{stem}.txt"

            dst_image = out_root / "images" / split / image_name
            dst_label = out_root / "labels" / split / label_name

            with Image.open(sample.path) as img:
                img.convert("RGB").save(dst_image, format="PNG")
            dst_label.write_text(bbox_line, encoding="utf-8")

            stats[split]["total_images"] += 1
            stats[split][f"{sample.source}_images"] += 1

            manifest_rows.append(
                {
                    "split": split,
                    "source": sample.source,
                    "copy_index": str(copy_idx),
                    "output_image": str(dst_image),
                    "source_image": str(sample.path),
                }
            )

    manifest_path = out_root / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["split", "source", "copy_index", "output_image", "source_image"],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    return {k: dict(v) for k, v in stats.items()}


def write_data_yaml(out_root: Path) -> Path:
    data_yaml = out_root / "data.yaml"
    yaml_text = (
        f"path: {out_root.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n"
        "  0: full_sperm\n"
    )
    data_yaml.write_text(yaml_text, encoding="utf-8")
    return data_yaml


def main() -> None:
    args = parse_args()

    if not (0 < args.train_ratio < 1):
        raise ValueError("--train-ratio must be in (0, 1)")
    if not (0 < args.val_ratio < 1):
        raise ValueError("--val-ratio must be in (0, 1)")
    if args.train_ratio + args.val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be < 1")
    if not (0 < args.bbox_scale <= 1):
        raise ValueError("--bbox-scale must be in (0, 1]")

    human = collect_human_samples(args.human_root)
    mouse = collect_mouse_samples(args.mouse_root)

    if not human:
        raise RuntimeError(f"No valid human images found under {args.human_root}")
    if not mouse:
        raise RuntimeError(f"No valid mouse images found under {args.mouse_root}")

    human_split = split_samples(human, args.train_ratio, args.val_ratio, args.seed)
    mouse_split = split_samples(mouse, args.train_ratio, args.val_ratio, args.seed)

    merged_split = {
        split: human_split[split] + mouse_split[split]
        for split in ("train", "val", "test")
    }

    ensure_output_dirs(args.out_root, args.overwrite)
    stats = write_yolo_dataset(
        out_root=args.out_root,
        split_map=merged_split,
        mouse_oversample=args.mouse_oversample,
        bbox_scale=args.bbox_scale,
    )
    data_yaml = write_data_yaml(args.out_root)

    summary = {
        "input_counts": {
            "human": len(human),
            "mouse": len(mouse),
        },
        "splits": stats,
        "settings": {
            "seed": args.seed,
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "mouse_oversample": args.mouse_oversample,
            "bbox_scale": args.bbox_scale,
        },
        "data_yaml": str(data_yaml),
    }

    summary_path = args.out_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
