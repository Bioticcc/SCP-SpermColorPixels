#!/usr/bin/env python3
"""Train and evaluate a YOLOv8 sperm instance-segmentation model."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp/matplotlib-codex").resolve()))

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLOv8 sperm segmentation model")
    parser.add_argument("--data", type=Path, required=True, help="Path to YOLO segmentation data.yaml")
    parser.add_argument(
        "--model",
        default="yolov8n-seg.pt",
        help="Ultralytics model spec, usually yolov8n-seg.pt for this baseline",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--fraction", type=float, default=1.0, help="Fraction of training data to use")
    parser.add_argument("--project", type=Path, default=Path("runs/segment"))
    parser.add_argument("--name", default="human_pseudo_yolov8nseg")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exist-ok", action="store_true", help="Allow reuse of an existing run name")
    parser.add_argument("--skip-test", action="store_true", help="Skip test split evaluation after training")
    return parser.parse_args()


def maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def extract_metric_block(metric: Any) -> dict[str, float | None]:
    if metric is None:
        return {
            "map50": None,
            "map50_95": None,
            "precision": None,
            "recall": None,
            "f1": None,
        }

    precision = maybe_float(getattr(metric, "mp", None))
    recall = maybe_float(getattr(metric, "mr", None))
    f1 = None
    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)

    return {
        "map50": maybe_float(getattr(metric, "map50", None)),
        "map50_95": maybe_float(getattr(metric, "map", None)),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def extract_metrics(results: Any) -> dict[str, dict[str, float | None]]:
    return {
        "box": extract_metric_block(getattr(results, "box", None)),
        "mask": extract_metric_block(getattr(results, "seg", None)),
    }


def main() -> None:
    args = parse_args()
    if not args.data.exists():
        raise FileNotFoundError(f"Data YAML not found: {args.data}")
    if not (0 < args.fraction <= 1):
        raise ValueError("--fraction must be in (0, 1]")

    args.project.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.model)

    train_results = model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(args.project),
        name=args.name,
        pretrained=True,
        seed=args.seed,
        patience=args.patience,
        fraction=args.fraction,
        save=True,
        plots=True,
        verbose=True,
        exist_ok=args.exist_ok,
    )

    save_dir = Path(train_results.save_dir)
    best_path = save_dir / "weights" / "best.pt"
    last_path = save_dir / "weights" / "last.pt"
    eval_model_path = best_path if best_path.exists() else last_path
    eval_model = YOLO(str(eval_model_path))

    val_results = eval_model.val(
        data=str(args.data),
        split="val",
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        plots=False,
        verbose=True,
    )

    test_metrics = None
    if not args.skip_test:
        test_results = eval_model.val(
            data=str(args.data),
            split="test",
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            plots=False,
            verbose=True,
        )
        test_metrics = extract_metrics(test_results)

    summary = {
        "run_dir": str(save_dir),
        "weights": {
            "best": str(best_path),
            "last": str(last_path),
            "eval_used": str(eval_model_path),
        },
        "metrics": {
            "val": extract_metrics(val_results),
            "test": test_metrics,
        },
        "train_config": {
            "data": str(args.data),
            "model": args.model,
            "epochs": args.epochs,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "device": args.device,
            "workers": args.workers,
            "patience": args.patience,
            "fraction": args.fraction,
            "seed": args.seed,
        },
    }

    summary_path = save_dir / "metrics_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
