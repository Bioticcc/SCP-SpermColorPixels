#!/usr/bin/env python3
"""Train YOLOv8 on weak full-sperm labels and export metric summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLOv8 full-sperm detector")
    parser.add_argument("--data", type=Path, required=True, help="Path to YOLO data.yaml")
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="Model spec for Ultralytics YOLO (e.g., yolov8n.pt or yolov8n.yaml)",
    )
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument(
        "--fraction",
        type=float,
        default=1.0,
        help="Fraction of training data to use each epoch (0, 1]",
    )
    parser.add_argument("--project", type=Path, default=Path("runs/yolo"))
    parser.add_argument("--name", default="full_sperm_no_annotations")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def extract_metrics(results) -> dict[str, float | None]:
    if results is None or not hasattr(results, "box"):
        return {
            "map50": None,
            "map50_95": None,
            "precision": None,
            "recall": None,
            "f1": None,
        }

    box = results.box
    precision = float(box.mp) if box.mp is not None else None
    recall = float(box.mr) if box.mr is not None else None

    f1 = None
    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)

    return {
        "map50": float(box.map50),
        "map50_95": float(box.map),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def main() -> None:
    args = parse_args()
    args.project.mkdir(parents=True, exist_ok=True)

    try:
        model = YOLO(args.model)
    except Exception as exc:
        if str(args.model).endswith(".pt"):
            print(f"Unable to load '{args.model}' ({exc}). Falling back to 'yolov8n.yaml'.")
            model = YOLO("yolov8n.yaml")
        else:
            raise

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

    summary = {
        "run_dir": str(save_dir),
        "weights": {
            "best": str(best_path),
            "last": str(last_path),
            "eval_used": str(eval_model_path),
        },
        "metrics": {
            "val": extract_metrics(val_results),
            "test": extract_metrics(test_results),
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
            "seed": args.seed,
            "fraction": args.fraction,
        },
    }

    summary_path = save_dir / "metrics_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
