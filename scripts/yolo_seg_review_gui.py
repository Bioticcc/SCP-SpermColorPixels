#!/usr/bin/env python3
"""Interactive Tk GUI for reviewing YOLOv8 sperm segmentation predictions."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp/matplotlib-codex").resolve()))

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageTk
import tkinter as tk
import tkinter.font as tkfont
from tkinter import filedialog, messagebox, ttk
from ultralytics import YOLO


IMAGE_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
PALETTE = [
    (31, 119, 180),
    (255, 127, 14),
    (44, 160, 44),
    (214, 39, 40),
    (148, 103, 189),
    (23, 190, 207),
]
DEFAULT_WEIGHT_CANDIDATES = [
    Path("runs/segment/runs/segment/human_pseudo_yolov8nseg_gpu/weights/best.pt"),
    Path("runs/segment/human_pseudo_yolov8nseg_gpu/weights/best.pt"),
]
DEFAULT_RAW_DIR_CANDIDATES = [
    Path("Training_Data/Raw_Yan_Data/Pilot_Dataset/Tiffs"),
    Path("Training_Data/Raw_Yan_Data/Pilot_Dataset/Png"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Review YOLOv8 segmentation masks on raw images")
    parser.add_argument("--weights", type=Path, default=None, help="Optional weights path to preselect")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-images", type=int, default=10)
    parser.add_argument("--ui-scale", type=float, default=1.65, help="Overall UI scale multiplier")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed for default image sampling")
    return parser.parse_args()


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTS


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


def load_rgb_image(path: Path) -> Image.Image:
    with Image.open(path) as img:
        arr = np.asarray(img.convert("RGB")) if img.mode not in {"I;16", "I", "F", "L"} else np.asarray(img)
    gray = normalize_to_u8(arr)
    return Image.fromarray(gray).convert("RGB")


def safe_float_list(values: Any) -> list[float]:
    return [float(v) for v in np.asarray(values).reshape(-1).tolist()]


def get_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def find_default_weight() -> Path | None:
    for candidate in DEFAULT_WEIGHT_CANDIDATES:
        if candidate.exists():
            return candidate.resolve()

    found = sorted(Path("runs").rglob("best.pt"), key=lambda path: path.stat().st_mtime, reverse=True)
    if found:
        return found[0].resolve()
    return None


def find_default_raw_dir() -> Path | None:
    for candidate in DEFAULT_RAW_DIR_CANDIDATES:
        if candidate.exists():
            return candidate.resolve()
    return None


def collect_image_paths(root: Path) -> list[Path]:
    return sorted(path.resolve() for path in root.iterdir() if is_image_file(path))


def sample_image_paths(root: Path, count: int, seed: int | None) -> list[Path]:
    paths = collect_image_paths(root)
    if len(paths) <= count:
        return paths

    rng = random.Random(seed)
    sampled = rng.sample(paths, count)
    return sorted(sampled)


def result_to_export(result: Any, source_path: Path, weights_path: Path, settings: dict[str, Any]) -> dict[str, Any]:
    names = getattr(result, "names", {}) or {}
    boxes_xyxy: list[list[float]] = []
    confidences: list[float] = []
    class_ids: list[int] = []

    if result.boxes is not None and len(result.boxes) > 0:
        boxes_xyxy = [safe_float_list(row) for row in result.boxes.xyxy.cpu().numpy()]
        confidences = safe_float_list(result.boxes.conf.cpu().numpy())
        class_ids = [int(v) for v in result.boxes.cls.cpu().numpy().reshape(-1).tolist()]

    masks_xy: list[list[list[float]]] = []
    masks_xyn: list[list[list[float]]] = []
    if result.masks is not None:
        masks_xy = [np.asarray(points, dtype=float).tolist() for points in result.masks.xy]
        masks_xyn = [np.asarray(points, dtype=float).tolist() for points in result.masks.xyn]

    detections: list[dict[str, Any]] = []
    count = max(len(boxes_xyxy), len(masks_xy))
    for index in range(count):
        class_id = class_ids[index] if index < len(class_ids) else 0
        detections.append(
            {
                "index": index,
                "class_id": class_id,
                "class_name": str(names.get(class_id, class_id)),
                "confidence": confidences[index] if index < len(confidences) else None,
                "box_xyxy": boxes_xyxy[index] if index < len(boxes_xyxy) else None,
                "polygon_xy": masks_xy[index] if index < len(masks_xy) else [],
                "polygon_xyn": masks_xyn[index] if index < len(masks_xyn) else [],
            }
        )

    image_shape = getattr(result, "orig_shape", None)
    return {
        "source_path": str(source_path),
        "weights_path": str(weights_path),
        "settings": settings,
        "image_shape_hw": list(image_shape) if image_shape is not None else None,
        "num_detections": len(detections),
        "detections": detections,
    }


def render_overlay(
    image: Image.Image,
    export: dict[str, Any],
    scale: float,
    label_font_px: int,
    line_width: int,
) -> Image.Image:
    display_size = (
        max(1, int(round(image.width * scale))),
        max(1, int(round(image.height * scale))),
    )
    base = image.resize(display_size, Image.Resampling.LANCZOS).convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    labels: list[tuple[float, float, str, tuple[int, int, int]]] = []
    font = get_font(label_font_px)

    for detection in export["detections"]:
        index = int(detection["index"])
        color = PALETTE[index % len(PALETTE)]
        polygon = detection.get("polygon_xy") or []
        box = detection.get("box_xyxy")
        confidence = detection.get("confidence")

        if len(polygon) >= 3:
            points = [(float(x) * scale, float(y) * scale) for x, y in polygon]
            draw.polygon(points, fill=(*color, 65), outline=(*color, 240))
            draw.line(points + [points[0]], fill=(*color, 255), width=line_width)
            label_x, label_y = points[0]
        elif box:
            x1, y1, x2, y2 = [float(value) * scale for value in box]
            draw.rectangle((x1, y1, x2, y2), outline=(*color, 255), width=line_width)
            label_x, label_y = x1, y1
        else:
            continue

        label = f"{index + 1}"
        if confidence is not None:
            label = f"{label}: {confidence:.2f}"
        labels.append((label_x + 8, max(0, label_y - label_font_px - 12), label, color))

    composed = Image.alpha_composite(base, overlay)
    label_draw = ImageDraw.Draw(composed)
    for label_x, label_y, label, color in labels:
        text_bbox = label_draw.textbbox((label_x, label_y), label, font=font)
        padding = max(4, label_font_px // 5)
        label_draw.rectangle(
            (
                text_bbox[0] - padding,
                text_bbox[1] - padding,
                text_bbox[2] + padding,
                text_bbox[3] + padding,
            ),
            fill=(*color, 230),
        )
        label_draw.text((label_x, label_y), label, fill=(255, 255, 255), font=font)

    return composed.convert("RGB")


def render_plain_image(image: Image.Image, scale: float) -> Image.Image:
    display_size = (
        max(1, int(round(image.width * scale))),
        max(1, int(round(image.height * scale))),
    )
    return image.resize(display_size, Image.Resampling.LANCZOS)


class SegReviewApp:
    def __init__(self, root: tk.Tk, args: argparse.Namespace) -> None:
        self.root = root
        self.args = args
        self.ui_scale = max(1.2, float(args.ui_scale))
        self.model: YOLO | None = None
        self.model_path_loaded: Path | None = None
        self.image_paths: list[Path] = []
        self.raw_cache: dict[Path, Image.Image] = {}
        self.result_cache: dict[Path, dict[str, Any]] = {}
        self.display_photo: ImageTk.PhotoImage | None = None
        self.current_mode = "raw"
        self.zoom = 1.0
        self.display_scale = 1.0

        default_weight = args.weights.resolve() if args.weights is not None else find_default_weight()
        default_raw_dir = find_default_raw_dir()

        self.weight_var = tk.StringVar(value=str(default_weight) if default_weight is not None else "")
        self.raw_dir_var = tk.StringVar(value=str(default_raw_dir) if default_raw_dir is not None else "")
        self.status_var = tk.StringVar(value="Loading defaults...")
        self.summary_var = tk.StringVar(value="No image selected")
        self.zoom_var = tk.StringVar(value="View 100%")
        self.mode_var = tk.StringVar(value="Raw Preview")
        self.conf_var = tk.DoubleVar(value=args.conf)
        self.iou_var = tk.DoubleVar(value=args.iou)
        self.imgsz_var = tk.IntVar(value=args.imgsz)

        self.configure_root()
        self.configure_style()
        self.build_ui()
        self.root.after(50, self.load_startup_defaults)

    def configure_root(self) -> None:
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        window_w = min(max(1640, int(screen_w * 0.94)), max(1100, screen_w - 40))
        window_h = min(max(1080, int(screen_h * 0.94)), max(860, screen_h - 40))
        self.root.title("Sperm Segmentation Review")
        self.root.geometry(f"{window_w}x{window_h}")
        self.root.minsize(1280, 860)
        self.root.tk.call("tk", "scaling", self.ui_scale)

    def configure_style(self) -> None:
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        base_size = max(12, int(12 * self.ui_scale))
        heading_size = max(15, int(15 * self.ui_scale))
        mono_size = max(11, int(11 * self.ui_scale))

        self.font_normal = tkfont.Font(size=base_size)
        self.font_bold = tkfont.Font(size=base_size, weight="bold")
        self.font_heading = tkfont.Font(size=heading_size, weight="bold")
        self.font_mono = tkfont.Font(family="TkFixedFont", size=mono_size)

        style.configure(".", font=self.font_normal)
        style.configure("TLabel", font=self.font_normal)
        style.configure("TButton", font=self.font_bold, padding=(12, 10))
        style.configure("Header.TLabel", font=self.font_heading)
        style.configure("Panel.TLabelframe.Label", font=self.font_bold)

    def build_ui(self) -> None:
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        left = ttk.Frame(self.root, padding=16)
        left.grid(row=0, column=0, sticky="nsw")
        left.columnconfigure(0, weight=1)

        right = ttk.Frame(self.root, padding=(0, 16, 16, 16))
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)

        self.build_left_panel(left)
        self.build_right_panel(right)

    def build_left_panel(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, text="Review Controls", style="Header.TLabel").grid(row=0, column=0, sticky="w")

        weights_frame = ttk.LabelFrame(parent, text="Model Weights", padding=12, style="Panel.TLabelframe")
        weights_frame.grid(row=1, column=0, sticky="ew", pady=(12, 10))
        weights_frame.columnconfigure(0, weight=1)
        self.weights_entry = tk.Text(weights_frame, height=3, wrap="word", font=self.font_mono)
        self.weights_entry.grid(row=0, column=0, sticky="ew")
        self.set_weights_text(self.weight_var.get())
        ttk.Button(weights_frame, text="Use Default", command=self.use_default_weight).grid(row=1, column=0, sticky="ew", pady=(10, 6))
        ttk.Button(weights_frame, text="Browse Weights", command=self.choose_weights).grid(row=2, column=0, sticky="ew")

        images_frame = ttk.LabelFrame(parent, text="Images", padding=12, style="Panel.TLabelframe")
        images_frame.grid(row=2, column=0, sticky="nsew", pady=10)
        images_frame.columnconfigure(0, weight=1)
        images_frame.rowconfigure(4, weight=1)
        parent.rowconfigure(2, weight=1)

        ttk.Label(images_frame, text="Raw image folder").grid(row=0, column=0, sticky="w")
        self.raw_dir_label = tk.Text(images_frame, height=3, wrap="word", font=self.font_mono)
        self.raw_dir_label.grid(row=1, column=0, sticky="ew", pady=(4, 10))
        self.set_raw_dir_text(self.raw_dir_var.get())

        ttk.Button(images_frame, text="Load 10 Random Raw Images", command=self.load_default_images).grid(row=2, column=0, sticky="ew", pady=(0, 6))
        ttk.Button(images_frame, text="Browse Image Files", command=self.choose_images).grid(row=3, column=0, sticky="ew", pady=(0, 10))

        list_frame = ttk.Frame(images_frame)
        list_frame.grid(row=4, column=0, sticky="nsew")
        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(0, weight=1)
        self.image_listbox = tk.Listbox(
            list_frame,
            activestyle="none",
            font=self.font_normal,
            height=14,
            exportselection=False,
        )
        self.image_listbox.grid(row=0, column=0, sticky="nsew")
        list_scroll = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.image_listbox.yview)
        list_scroll.grid(row=0, column=1, sticky="ns")
        self.image_listbox.configure(yscrollcommand=list_scroll.set)
        self.image_listbox.bind("<<ListboxSelect>>", self.on_image_selected)

        actions_frame = ttk.LabelFrame(parent, text="Actions", padding=12, style="Panel.TLabelframe")
        actions_frame.grid(row=3, column=0, sticky="ew", pady=10)
        actions_frame.columnconfigure(0, weight=1)
        ttk.Button(actions_frame, text="Show Raw Preview", command=self.show_raw_preview).grid(row=0, column=0, sticky="ew", pady=(0, 6))
        ttk.Button(actions_frame, text="Generate Overlay For Selected Image", command=self.generate_current_overlay).grid(row=1, column=0, sticky="ew", pady=(0, 6))
        ttk.Button(actions_frame, text="Generate Overlay For All Loaded Images", command=self.generate_all_overlays).grid(row=2, column=0, sticky="ew", pady=(0, 6))
        ttk.Button(actions_frame, text="Save Current Overlay", command=self.save_current_overlay).grid(row=3, column=0, sticky="ew")

        params_frame = ttk.LabelFrame(parent, text="Inference Settings", padding=12, style="Panel.TLabelframe")
        params_frame.grid(row=4, column=0, sticky="ew", pady=10)
        params_frame.columnconfigure(1, weight=1)
        ttk.Label(params_frame, text="Confidence").grid(row=0, column=0, sticky="w", padx=(0, 8))
        ttk.Spinbox(params_frame, from_=0.01, to=1.0, increment=0.01, textvariable=self.conf_var, width=8).grid(row=0, column=1, sticky="w")
        ttk.Label(params_frame, text="IoU").grid(row=1, column=0, sticky="w", padx=(0, 8), pady=(8, 0))
        ttk.Spinbox(params_frame, from_=0.05, to=1.0, increment=0.05, textvariable=self.iou_var, width=8).grid(row=1, column=1, sticky="w", pady=(8, 0))
        ttk.Label(params_frame, text="Image Size").grid(row=2, column=0, sticky="w", padx=(0, 8), pady=(8, 0))
        ttk.Spinbox(params_frame, from_=320, to=2048, increment=32, textvariable=self.imgsz_var, width=8).grid(row=2, column=1, sticky="w", pady=(8, 0))

    def build_right_panel(self, parent: ttk.Frame) -> None:
        top = ttk.Frame(parent)
        top.grid(row=0, column=0, sticky="ew")
        top.columnconfigure(0, weight=1)

        ttk.Label(top, textvariable=self.summary_var, style="Header.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(top, textvariable=self.mode_var, font=self.font_bold).grid(row=0, column=1, sticky="e", padx=(16, 12))
        ttk.Label(top, textvariable=self.zoom_var, font=self.font_bold).grid(row=0, column=2, sticky="e")

        nav = ttk.Frame(parent)
        nav.grid(row=1, column=0, sticky="ew", pady=(10, 10))
        ttk.Button(nav, text="Previous Image", command=self.previous_image).pack(side=tk.LEFT)
        ttk.Button(nav, text="Next Image", command=self.next_image).pack(side=tk.LEFT, padx=(8, 16))
        ttk.Button(nav, text="Fit", command=self.zoom_fit).pack(side=tk.LEFT)
        ttk.Button(nav, text="100%", command=self.zoom_actual).pack(side=tk.LEFT, padx=8)
        ttk.Button(nav, text="Zoom -", command=self.zoom_out).pack(side=tk.LEFT)
        ttk.Button(nav, text="Zoom +", command=self.zoom_in).pack(side=tk.LEFT, padx=(8, 0))

        canvas_frame = ttk.Frame(parent)
        canvas_frame.grid(row=2, column=0, sticky="nsew")
        canvas_frame.columnconfigure(0, weight=1)
        canvas_frame.rowconfigure(0, weight=1)
        parent.rowconfigure(2, weight=1)

        self.canvas = tk.Canvas(canvas_frame, bg="#161616", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        x_scroll = ttk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        y_scroll = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        x_scroll.grid(row=1, column=0, sticky="ew")
        y_scroll.grid(row=0, column=1, sticky="ns")
        self.canvas.configure(xscrollcommand=x_scroll.set, yscrollcommand=y_scroll.set)
        self.canvas.bind("<Configure>", lambda _event: self.redraw_canvas())
        self.canvas.bind("<MouseWheel>", self.on_mousewheel)
        self.canvas.bind("<Shift-MouseWheel>", self.on_shift_mousewheel)
        self.canvas.bind("<Button-4>", self.on_scroll_up)
        self.canvas.bind("<Button-5>", self.on_scroll_down)
        self.canvas.bind("<Shift-Button-4>", self.on_shift_scroll_up)
        self.canvas.bind("<Shift-Button-5>", self.on_shift_scroll_down)
        self.root.bind("<Control-plus>", lambda _event: self.zoom_in())
        self.root.bind("<Control-equal>", lambda _event: self.zoom_in())
        self.root.bind("<Control-minus>", lambda _event: self.zoom_out())
        self.root.bind("<Control-0>", lambda _event: self.zoom_fit())
        self.root.bind("1", lambda _event: self.zoom_actual())
        self.root.bind("f", lambda _event: self.zoom_fit())

        bottom = ttk.Frame(parent)
        bottom.grid(row=3, column=0, sticky="ew", pady=(10, 0))
        ttk.Label(bottom, textvariable=self.status_var, font=self.font_bold).pack(side=tk.LEFT, fill=tk.X, expand=True)

    def set_weights_text(self, text: str) -> None:
        self.weights_entry.configure(state="normal")
        self.weights_entry.delete("1.0", tk.END)
        self.weights_entry.insert("1.0", text)
        self.weights_entry.configure(state="disabled")

    def set_raw_dir_text(self, text: str) -> None:
        self.raw_dir_label.configure(state="normal")
        self.raw_dir_label.delete("1.0", tk.END)
        self.raw_dir_label.insert("1.0", text)
        self.raw_dir_label.configure(state="disabled")

    def load_startup_defaults(self) -> None:
        if self.weight_var.get():
            self.status_var.set("Default model weights are preselected. Choose an image and click Generate Overlay.")
        else:
            self.status_var.set("No default weights found. Choose weights first, then generate an overlay.")

        self.load_default_images()

    def use_default_weight(self) -> None:
        weight = find_default_weight()
        if weight is None:
            messagebox.showerror("Weights not found", "No default best.pt file was found under runs/.")
            return
        self.weight_var.set(str(weight))
        self.set_weights_text(str(weight))
        self.model = None
        self.model_path_loaded = None
        self.status_var.set("Default weights selected. Click Generate Overlay when ready.")

    def choose_weights(self) -> None:
        selected = filedialog.askopenfilename(
            title="Select YOLO weights",
            filetypes=[("PyTorch weights", "*.pt"), ("All files", "*.*")],
        )
        if not selected:
            return
        self.weight_var.set(selected)
        self.set_weights_text(selected)
        self.model = None
        self.model_path_loaded = None
        self.status_var.set("Weights selected. Click Generate Overlay to run the model.")

    def load_default_images(self) -> None:
        raw_dir_text = self.raw_dir_var.get().strip()
        if not raw_dir_text:
            raw_dir = find_default_raw_dir()
            if raw_dir is None:
                messagebox.showerror("Raw data not found", "No default raw Yan image directory was found.")
                return
            self.raw_dir_var.set(str(raw_dir))
            self.set_raw_dir_text(str(raw_dir))
        else:
            raw_dir = Path(raw_dir_text)

        if not raw_dir.exists():
            messagebox.showerror("Raw data not found", f"Raw image directory does not exist:\n{raw_dir}")
            return

        self.image_paths = sample_image_paths(raw_dir, self.args.max_images, self.args.seed)
        self.raw_cache.clear()
        self.result_cache.clear()
        self.populate_image_list()
        if self.image_paths:
            self.select_image(0)
            self.root.after(10, self.zoom_actual)
            self.status_var.set(
                f"Loaded {len(self.image_paths)} random raw images. Click Generate Overlay to see the model output."
            )
        else:
            self.summary_var.set("No image selected")
            self.status_var.set("No image files were found in the default raw image folder.")

    def choose_images(self) -> None:
        selected = filedialog.askopenfilenames(
            title="Select raw images",
            filetypes=[
                ("Image files", "*.tif *.tiff *.png *.jpg *.jpeg *.bmp"),
                ("All files", "*.*"),
            ],
        )
        if not selected:
            return

        self.image_paths = [Path(path).resolve() for path in selected if is_image_file(Path(path))]
        self.raw_cache.clear()
        self.result_cache.clear()
        self.populate_image_list()
        if self.image_paths:
            parent = self.image_paths[0].parent
            self.raw_dir_var.set(str(parent))
            self.set_raw_dir_text(str(parent))
            self.select_image(0)
            self.root.after(10, self.zoom_actual)
            self.status_var.set(
                f"Loaded {len(self.image_paths)} chosen images. Click Generate Overlay to run the model."
            )

    def populate_image_list(self) -> None:
        self.image_listbox.delete(0, tk.END)
        for path in self.image_paths:
            self.image_listbox.insert(tk.END, path.name)

    def current_path(self) -> Path | None:
        selection = self.image_listbox.curselection()
        if not selection:
            return None
        index = int(selection[0])
        if index < 0 or index >= len(self.image_paths):
            return None
        return self.image_paths[index]

    def select_image(self, index: int) -> None:
        if not self.image_paths:
            return
        index = max(0, min(index, len(self.image_paths) - 1))
        self.image_listbox.selection_clear(0, tk.END)
        self.image_listbox.selection_set(index)
        self.image_listbox.activate(index)
        self.image_listbox.see(index)
        self.current_mode = "overlay" if self.image_paths[index] in self.result_cache else "raw"
        self.show_current_image()

    def on_image_selected(self, _event: tk.Event) -> None:
        path = self.current_path()
        if path is None:
            return
        self.current_mode = "overlay" if path in self.result_cache else "raw"
        self.show_current_image()

    def get_raw_image(self, path: Path) -> Image.Image:
        if path not in self.raw_cache:
            self.raw_cache[path] = load_rgb_image(path)
        return self.raw_cache[path]

    def ensure_model_loaded(self) -> bool:
        weights_path = self.weight_var.get().strip()
        if not weights_path:
            messagebox.showerror("Weights required", "Choose a model weights file first.")
            return False

        path = Path(weights_path)
        if not path.exists():
            messagebox.showerror("Weights not found", f"Weights file does not exist:\n{path}")
            return False

        path = path.resolve()
        if self.model is not None and self.model_path_loaded == path:
            return True

        try:
            self.status_var.set(f"Loading model from {path.name}...")
            self.root.update_idletasks()
            self.model = YOLO(str(path))
            self.model_path_loaded = path
            self.status_var.set(f"Model ready: {path.name}")
            return True
        except Exception as exc:  # noqa: BLE001
            self.model = None
            self.model_path_loaded = None
            messagebox.showerror("Model load failed", str(exc))
            self.status_var.set("Failed to load model.")
            return False

    def run_inference_for_path(self, path: Path) -> bool:
        if not self.ensure_model_loaded():
            return False

        image = self.get_raw_image(path)
        try:
            self.status_var.set(f"Running model on {path.name}...")
            self.root.update_idletasks()
            result = self.model.predict(  # type: ignore[union-attr]
                source=np.asarray(image),
                imgsz=int(self.imgsz_var.get()),
                conf=float(self.conf_var.get()),
                iou=float(self.iou_var.get()),
                device=self.args.device,
                verbose=False,
            )[0]
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Inference failed", str(exc))
            self.status_var.set("Inference failed.")
            return False

        self.result_cache[path] = result_to_export(
            result,
            source_path=path,
            weights_path=self.model_path_loaded or Path(self.weight_var.get()),
            settings={
                "imgsz": int(self.imgsz_var.get()),
                "conf": float(self.conf_var.get()),
                "iou": float(self.iou_var.get()),
                "device": self.args.device,
            },
        )
        return True

    def show_current_image(self) -> None:
        path = self.current_path()
        if path is None:
            self.canvas.delete("all")
            self.summary_var.set("No image selected")
            self.mode_var.set("Idle")
            self.status_var.set("Load images to begin.")
            return

        self.get_raw_image(path)
        self.summary_var.set(path.name)
        if self.current_mode == "overlay" and path in self.result_cache:
            export = self.result_cache[path]
            confs = [
                detection["confidence"]
                for detection in export["detections"]
                if detection.get("confidence") is not None
            ]
            conf_text = ", ".join(f"{value:.2f}" for value in confs[:6]) if confs else "none"
            self.mode_var.set("Model Overlay")
            self.status_var.set(
                f"{export['num_detections']} detections | conf: {conf_text} | Click Show Raw Preview to compare."
            )
        else:
            self.mode_var.set("Raw Preview")
            self.status_var.set("Raw preview loaded. Click Generate Overlay to see the model attempt.")

        self.redraw_canvas()

    def build_display_image(self, path: Path, scale: float) -> Image.Image:
        image = self.get_raw_image(path)
        if self.current_mode == "overlay" and path in self.result_cache:
            return render_overlay(
                image,
                self.result_cache[path],
                scale=scale,
                label_font_px=max(24, int(20 * self.ui_scale)),
                line_width=max(3, int(3 * self.ui_scale)),
            )
        return render_plain_image(image, scale=scale)

    def redraw_canvas(self) -> None:
        path = self.current_path()
        if path is None:
            return

        image = self.get_raw_image(path)
        canvas_w = max(1, self.canvas.winfo_width())
        canvas_h = max(1, self.canvas.winfo_height())
        fit_scale = min(canvas_w / image.width, canvas_h / image.height, 1.0)
        scale = max(0.05, fit_scale * self.zoom)
        self.display_scale = scale

        display = self.build_display_image(path, scale=scale)
        display_w, display_h = display.size
        self.display_photo = ImageTk.PhotoImage(display)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.display_photo)
        self.canvas.configure(scrollregion=(0, 0, display_w, display_h))
        self.zoom_var.set(f"View {self.display_scale * 100:.0f}%")

    def show_raw_preview(self) -> None:
        path = self.current_path()
        if path is None:
            messagebox.showinfo("No image selected", "Choose an image first.")
            return
        self.current_mode = "raw"
        self.show_current_image()

    def generate_current_overlay(self) -> None:
        path = self.current_path()
        if path is None:
            messagebox.showinfo("No image selected", "Choose an image first.")
            return
        if self.run_inference_for_path(path):
            self.current_mode = "overlay"
            self.show_current_image()

    def generate_all_overlays(self) -> None:
        if not self.image_paths:
            messagebox.showinfo("No images loaded", "Load some images first.")
            return
        if not self.ensure_model_loaded():
            return

        total = len(self.image_paths)
        success = 0
        for index, path in enumerate(self.image_paths, start=1):
            self.status_var.set(f"Generating overlay {index}/{total}: {path.name}")
            self.root.update_idletasks()
            if self.run_inference_for_path(path):
                success += 1

        self.current_mode = "overlay"
        self.show_current_image()
        self.status_var.set(f"Generated overlays for {success}/{total} loaded images.")

    def save_current_overlay(self) -> None:
        path = self.current_path()
        if path is None:
            messagebox.showinfo("No image selected", "Choose an image first.")
            return
        if path not in self.result_cache:
            messagebox.showinfo("No overlay yet", "Generate an overlay first, then save it.")
            return

        selected = filedialog.asksaveasfilename(
            title="Save overlay image",
            defaultextension=".png",
            initialfile=f"{path.stem}_overlay.png",
            filetypes=[("PNG image", "*.png"), ("All files", "*.*")],
        )
        if not selected:
            return

        overlay_path = Path(selected)
        json_path = overlay_path.with_suffix(".json")
        image = self.get_raw_image(path)
        overlay = render_overlay(
            image,
            self.result_cache[path],
            scale=1.0,
            label_font_px=34,
            line_width=5,
        )
        overlay.save(overlay_path)
        json_path.write_text(json.dumps(self.result_cache[path], indent=2), encoding="utf-8")
        self.status_var.set(f"Saved {overlay_path.name} and {json_path.name}.")

    def previous_image(self) -> None:
        selection = self.image_listbox.curselection()
        if not selection:
            if self.image_paths:
                self.select_image(0)
            return
        self.select_image(int(selection[0]) - 1)

    def next_image(self) -> None:
        selection = self.image_listbox.curselection()
        if not selection:
            if self.image_paths:
                self.select_image(0)
            return
        self.select_image(int(selection[0]) + 1)

    def zoom_fit(self) -> None:
        self.zoom = 1.0
        self.redraw_canvas()

    def zoom_actual(self) -> None:
        path = self.current_path()
        if path is None:
            return
        image = self.get_raw_image(path)
        canvas_w = max(1, self.canvas.winfo_width())
        canvas_h = max(1, self.canvas.winfo_height())
        fit_scale = min(canvas_w / image.width, canvas_h / image.height, 1.0)
        self.zoom = 1.0 / max(fit_scale, 0.001)
        self.redraw_canvas()

    def zoom_in(self) -> None:
        self.zoom = min(12.0, self.zoom * 1.25)
        self.redraw_canvas()

    def zoom_out(self) -> None:
        self.zoom = max(0.1, self.zoom / 1.25)
        self.redraw_canvas()

    def on_mousewheel(self, event: tk.Event) -> None:
        self.canvas.yview_scroll(-1 * int(event.delta / 120), "units")

    def on_shift_mousewheel(self, event: tk.Event) -> None:
        self.canvas.xview_scroll(-1 * int(event.delta / 120), "units")

    def on_scroll_up(self, _event: tk.Event) -> None:
        self.canvas.yview_scroll(-1, "units")

    def on_scroll_down(self, _event: tk.Event) -> None:
        self.canvas.yview_scroll(1, "units")

    def on_shift_scroll_up(self, _event: tk.Event) -> None:
        self.canvas.xview_scroll(-1, "units")

    def on_shift_scroll_down(self, _event: tk.Event) -> None:
        self.canvas.xview_scroll(1, "units")


def main() -> None:
    args = parse_args()
    root = tk.Tk()
    SegReviewApp(root, args)
    root.mainloop()


if __name__ == "__main__":
    main()
