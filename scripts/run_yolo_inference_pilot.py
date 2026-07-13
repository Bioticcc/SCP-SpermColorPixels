#!/usr/bin/env python3
"""Run YOLO inference on raw pilot dataset and export per-image summaries."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from ultralytics import YOLO


IMAGE_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLO inference on pilot raw images")
    parser.add_argument("--weights", type=Path, required=True, help="Path to model weights")
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("Training_Data/Raw_Yan_Data/Pilot_Dataset/Tiffs"),
        help="Directory containing raw pilot images",
    )
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--project", type=Path, default=Path("runs/yolo"))
    parser.add_argument("--name", default="pilot_inference_no_annotations")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.project.mkdir(parents=True, exist_ok=True)

    files = sorted(
        p for p in args.source.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
    if not files:
        raise RuntimeError(f"No image files found in {args.source}")

    model = YOLO(str(args.weights))

    rows: list[dict[str, str]] = []
    all_confs: list[float] = []
    save_dir: Path | None = None

    results = model.predict(
        source=[str(p) for p in files],
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        save=True,
        save_txt=True,
        save_conf=True,
        stream=True,
        project=str(args.project),
        name=args.name,
        verbose=False,
    )

    for r in results:
        if save_dir is None and hasattr(r, "save_dir"):
            save_dir = Path(r.save_dir)

        if r.boxes is None:
            n = 0
            confs: list[float] = []
        else:
            n = len(r.boxes)
            confs = [float(c) for c in r.boxes.conf.cpu().tolist()]

        all_confs.extend(confs)

        row = {
            "image": Path(r.path).name,
            "num_detections": str(n),
            "max_conf": f"{max(confs):.6f}" if confs else "",
            "mean_conf": f"{(sum(confs) / len(confs)):.6f}" if confs else "",
        }
        rows.append(row)

    out_dir = args.project / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "inference_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["image", "num_detections", "max_conf", "mean_conf"]
        )
        writer.writeheader()
        writer.writerows(rows)

    det_counts = [int(r["num_detections"]) for r in rows]
    images_with_det = sum(1 for d in det_counts if d > 0)
    total_detections = sum(det_counts)

    summary = {
        "weights": str(args.weights),
        "source": str(args.source),
        "num_images": len(rows),
        "images_with_detections": images_with_det,
        "images_without_detections": len(rows) - images_with_det,
        "total_detections": total_detections,
        "mean_detections_per_image": total_detections / len(rows),
        "mean_detection_confidence": (sum(all_confs) / len(all_confs)) if all_confs else None,
        "inference_summary_csv": str(csv_path),
        "predicted_images_dir": str(save_dir) if save_dir is not None else str(out_dir),
    }

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
