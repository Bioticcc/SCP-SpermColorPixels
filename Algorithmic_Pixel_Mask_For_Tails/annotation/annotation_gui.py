#!/usr/bin/env python3
"""Tkinter GUI for manual Ward sperm source-image annotation."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageTk, UnidentifiedImageError
import tkinter as tk
import tkinter.font as tkfont
from tkinter import messagebox, ttk

from .annotation_io import (
    collect_source_images,
    load_annotation_for_image,
    resolve_scp_path,
    safe_image_id,
    save_annotation,
)
from .annotation_models import (
    CrossingAnnotation,
    CrossingContinuation,
    HeadAnnotation,
    ImageAnnotation,
    SpermInstance,
)
from .annotation_validation import ValidationReport, validate_annotation


TOOL_SELECT = "select"
TOOL_PAN = "pan"
TOOL_HEAD_ELLIPSE = "head_ellipse"
TOOL_HEAD_POLYGON = "head_polygon"
TOOL_NECK = "neck"
TOOL_TAIL = "tail"
TOOL_CROSSING = "crossing"
FILTERS = {"all", "incomplete", "warnings", "40x", "100x", "crossings"}
PALETTE = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#17becf",
    "#e377c2",
    "#bcbd22",
    "#7f7f7f",
    "#8c564b",
]


@dataclass
class ViewTransform:
    scale: float = 1.0
    offset_x: float = 0.0
    offset_y: float = 0.0

    def image_to_canvas(self, point: tuple[float, float] | list[float]) -> tuple[float, float]:
        return (float(point[0]) * self.scale + self.offset_x, float(point[1]) * self.scale + self.offset_y)

    def canvas_to_image(self, point: tuple[float, float] | list[float]) -> tuple[float, float]:
        return ((float(point[0]) - self.offset_x) / self.scale, (float(point[1]) - self.offset_y) / self.scale)

    def pan(self, dx: float, dy: float) -> None:
        self.offset_x += dx
        self.offset_y += dy

    def zoom_at(self, canvas_point: tuple[float, float], factor: float) -> None:
        image_point = self.canvas_to_image(canvas_point)
        self.scale = max(0.05, min(32.0, self.scale * factor))
        self.offset_x = float(canvas_point[0]) - image_point[0] * self.scale
        self.offset_y = float(canvas_point[1]) - image_point[1] * self.scale

    def fit(self, image_size: tuple[int, int], canvas_size: tuple[int, int]) -> None:
        width, height = image_size
        canvas_w, canvas_h = canvas_size
        self.scale = max(0.05, min(canvas_w / max(1, width), canvas_h / max(1, height), 1.0))
        self.offset_x = max(0.0, (canvas_w - width * self.scale) / 2.0)
        self.offset_y = max(0.0, (canvas_h - height * self.scale) / 2.0)


class UndoStack:
    def __init__(self, limit: int = 100) -> None:
        self.limit = limit
        self.undo_items: list[ImageAnnotation] = []
        self.redo_items: list[ImageAnnotation] = []

    def push(self, annotation: ImageAnnotation) -> None:
        self.undo_items.append(annotation.clone())
        if len(self.undo_items) > self.limit:
            self.undo_items.pop(0)
        self.redo_items.clear()

    def undo(self, current: ImageAnnotation) -> ImageAnnotation | None:
        if not self.undo_items:
            return None
        self.redo_items.append(current.clone())
        return self.undo_items.pop()

    def redo(self, current: ImageAnnotation) -> ImageAnnotation | None:
        if not self.redo_items:
            return None
        self.undo_items.append(current.clone())
        return self.redo_items.pop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manual Ward source-image sperm annotation GUI.")
    parser.add_argument("--input-root", type=Path, default=Path("Raw_Ward_Data"))
    parser.add_argument("--annotation-root", type=Path, default=Path("annotations/ward_gold_v1"))
    parser.add_argument("--scp-output-root", type=Path, default=Path("outputs/updated_outputs_path_v2"))
    parser.add_argument("--image-id", default="")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--filter", choices=sorted(FILTERS), default="all")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--ui-scale", type=float, default=1.2)
    return parser.parse_args()


def instance_color(index: int) -> str:
    return PALETTE[index % len(PALETTE)]


def _distance(a: list[float], b: list[float]) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


class AnnotationApp:
    def __init__(
        self,
        root: tk.Tk,
        image_paths: list[Path],
        input_root: Path,
        annotation_root: Path,
        scp_output_root: Path,
        start_index: int,
        initial_filter: str,
        ui_scale: float,
    ) -> None:
        self.root = root
        self.image_paths = image_paths
        self.input_root = resolve_scp_path(input_root)
        self.annotation_root = resolve_scp_path(annotation_root)
        self.scp_output_root = resolve_scp_path(scp_output_root)
        self.ui_scale = max(1.0, ui_scale)

        self.filtered_indices: list[int] = list(range(len(self.image_paths)))
        self.current_filtered_index = max(0, min(start_index, max(0, len(self.filtered_indices) - 1)))
        self.current_image: Image.Image | None = None
        self.overlay_image: Image.Image | None = None
        self.display_photo: ImageTk.PhotoImage | None = None
        self.annotation: ImageAnnotation | None = None
        self.loaded_mtime_ns: int | None = None
        self.undo_stack = UndoStack()
        self.view = ViewTransform()
        self.tool = tk.StringVar(value=TOOL_SELECT)
        self.filter_var = tk.StringVar(value=initial_filter)
        self.status_var = tk.StringVar(value="")
        self.progress_var = tk.StringVar(value="")
        self.zoom_var = tk.StringVar(value="100%")
        self.selected_instance_var = tk.StringVar(value="")
        self.selected_object: tuple[str, str] | None = None
        self.selected_point: tuple[str, str, int | None] | None = None
        self.show_overlay_var = tk.BooleanVar(value=False)
        self.selected_only_var = tk.BooleanVar(value=False)
        self.dim_unselected_var = tk.BooleanVar(value=True)
        self.space_pan = False
        self.pan_start: tuple[int, int] | None = None
        self.drag_start_image: list[float] | None = None
        self.active_polygon: list[list[float]] = []
        self.active_tail: list[list[float]] = []
        self.active_crossing_start: list[float] | None = None
        self.pending_continuation_field: str | None = None

        self.canvas: tk.Canvas | None = None
        self.tree: ttk.Treeview | None = None
        self.validation_text: tk.Text | None = None
        self.notes_text: tk.Text | None = None
        self.object_frame: ttk.Frame | None = None
        self.scp_aid_text: tk.Text | None = None

        self.configure_root()
        self.build_layout()
        self.apply_filter(initial_filter, keep_image=False)
        self.load_current_image()

    def configure_root(self) -> None:
        self.root.title("Ward Gold Annotation")
        self.root.geometry("1650x1000")
        self.root.minsize(1200, 760)
        try:
            self.root.tk.call("tk", "scaling", self.ui_scale)
        except tk.TclError:
            pass
        base_size = max(10, int(11 * self.ui_scale))
        heading_size = max(13, int(14 * self.ui_scale))
        for font_name in ("TkDefaultFont", "TkTextFont", "TkMenuFont"):
            try:
                tkfont.nametofont(font_name).configure(size=base_size)
            except tk.TclError:
                pass
        style = ttk.Style()
        style.configure(".", font=("TkDefaultFont", base_size))
        style.configure("Header.TLabel", font=("TkDefaultFont", heading_size, "bold"))
        style.configure("Tool.TButton", padding=(8, 6))

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.bind("v", lambda _event: self.set_tool(TOOL_SELECT))
        self.root.bind("h", lambda _event: self.set_tool(TOOL_PAN))
        self.root.bind("n", lambda _event: self.add_instance())
        self.root.bind("e", lambda _event: self.set_tool(TOOL_HEAD_ELLIPSE))
        self.root.bind("g", lambda _event: self.set_tool(TOOL_HEAD_POLYGON))
        self.root.bind("k", lambda _event: self.set_tool(TOOL_NECK))
        self.root.bind("t", lambda _event: self.set_tool(TOOL_TAIL))
        self.root.bind("c", lambda _event: self.set_tool(TOOL_CROSSING))
        self.root.bind("<Return>", lambda _event: self.finish_active_drawing())
        self.root.bind("<Escape>", lambda _event: self.cancel_active_drawing())
        self.root.bind("<Delete>", lambda _event: self.delete_selected())
        self.root.bind("<Control-s>", lambda _event: self.save_current(show_message=False))
        self.root.bind("<Control-z>", lambda _event: self.undo())
        self.root.bind("<Control-y>", lambda _event: self.redo())
        self.root.bind("<Control-Shift-Z>", lambda _event: self.redo())
        self.root.bind("<Control-Shift-V>", lambda _event: self.validate_current())
        self.root.bind("<Prior>", lambda _event: self.previous_image())
        self.root.bind("<Next>", lambda _event: self.next_image())
        self.root.bind("o", lambda _event: self.toggle_overlay())
        self.root.bind("i", lambda _event: self.toggle_selected_only())
        self.root.bind("f", lambda _event: self.fit_image())
        self.root.bind("1", lambda _event: self.zoom_actual())
        self.root.bind("<KeyPress-space>", self.on_space_press)
        self.root.bind("<KeyRelease-space>", self.on_space_release)

    def build_layout(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        toolbar = ttk.Frame(self.root, padding=(8, 8))
        toolbar.grid(row=0, column=0, sticky="ew")
        ttk.Label(toolbar, text="Tool").pack(side=tk.LEFT, padx=(0, 6))
        for label, tool in (
            ("Select", TOOL_SELECT),
            ("Pan", TOOL_PAN),
            ("Head", TOOL_HEAD_ELLIPSE),
            ("Poly", TOOL_HEAD_POLYGON),
            ("Neck", TOOL_NECK),
            ("Tail", TOOL_TAIL),
            ("Crossing", TOOL_CROSSING),
        ):
            ttk.Button(toolbar, text=label, style="Tool.TButton", command=lambda value=tool: self.set_tool(value)).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="New Sperm", command=self.add_instance).pack(side=tk.LEFT, padx=(12, 2))
        ttk.Button(toolbar, text="Save", command=lambda: self.save_current(show_message=True)).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Validate", command=self.validate_current).pack(side=tk.LEFT, padx=2)
        ttk.Checkbutton(toolbar, text="SCP Aid", variable=self.show_overlay_var, command=self.redraw).pack(side=tk.LEFT, padx=(12, 2))
        ttk.Checkbutton(toolbar, text="Selected Only", variable=self.selected_only_var, command=self.redraw).pack(side=tk.LEFT, padx=2)
        ttk.Checkbutton(toolbar, text="Dim Others", variable=self.dim_unselected_var, command=self.redraw).pack(side=tk.LEFT, padx=2)
        ttk.Label(toolbar, text="Filter").pack(side=tk.LEFT, padx=(14, 4))
        filter_box = ttk.Combobox(toolbar, textvariable=self.filter_var, values=sorted(FILTERS), state="readonly", width=12)
        filter_box.pack(side=tk.LEFT)
        filter_box.bind("<<ComboboxSelected>>", lambda _event: self.apply_filter(self.filter_var.get(), keep_image=True))
        ttk.Label(toolbar, textvariable=self.progress_var).pack(side=tk.RIGHT)

        main = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main.grid(row=1, column=0, sticky="nsew")

        canvas_frame = ttk.Frame(main, padding=6)
        canvas_frame.columnconfigure(0, weight=1)
        canvas_frame.rowconfigure(0, weight=1)
        self.canvas = tk.Canvas(canvas_frame, bg="#181818", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        x_scroll = ttk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        y_scroll = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        x_scroll.grid(row=1, column=0, sticky="ew")
        y_scroll.grid(row=0, column=1, sticky="ns")
        self.canvas.configure(xscrollcommand=x_scroll.set, yscrollcommand=y_scroll.set)
        self.canvas.bind("<ButtonPress-1>", self.on_canvas_press)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_release)
        self.canvas.bind("<Double-Button-1>", lambda _event: self.finish_active_drawing())
        self.canvas.bind("<ButtonPress-2>", self.on_pan_press)
        self.canvas.bind("<B2-Motion>", self.on_pan_drag)
        self.canvas.bind("<MouseWheel>", self.on_mousewheel)
        self.canvas.bind("<Button-4>", lambda event: self.zoom_from_event(event, 1.25))
        self.canvas.bind("<Button-5>", lambda event: self.zoom_from_event(event, 0.8))
        self.canvas.bind("<Configure>", lambda _event: self.redraw())
        main.add(canvas_frame, weight=4)

        side = ttk.Frame(main, padding=8)
        side.columnconfigure(0, weight=1)
        side.rowconfigure(1, weight=1)
        main.add(side, weight=1)

        ttk.Label(side, text="Image Objects", style="Header.TLabel").grid(row=0, column=0, sticky="w")
        self.tree = ttk.Treeview(side, height=12, selectmode="browse")
        self.tree.grid(row=1, column=0, sticky="nsew", pady=(4, 8))
        self.tree.bind("<<TreeviewSelect>>", self.on_tree_select)

        self.object_frame = ttk.Frame(side)
        self.object_frame.grid(row=2, column=0, sticky="ew", pady=(0, 8))
        self.object_frame.columnconfigure(1, weight=1)

        ttk.Label(side, text="Image Notes").grid(row=3, column=0, sticky="w")
        self.notes_text = tk.Text(side, height=4, wrap="word")
        self.notes_text.grid(row=4, column=0, sticky="ew", pady=(2, 8))
        self.notes_text.bind("<<Modified>>", self.on_notes_modified)

        ttk.Label(side, text="Validation", style="Header.TLabel").grid(row=5, column=0, sticky="w")
        self.validation_text = tk.Text(side, height=9, wrap="word", state="disabled")
        self.validation_text.grid(row=6, column=0, sticky="ew", pady=(2, 8))

        ttk.Label(side, text="Read-Only SCP Aids", style="Header.TLabel").grid(row=7, column=0, sticky="w")
        self.scp_aid_text = tk.Text(side, height=7, wrap="word", state="disabled")
        self.scp_aid_text.grid(row=8, column=0, sticky="ew")

        status = ttk.Frame(self.root, padding=(8, 4))
        status.grid(row=2, column=0, sticky="ew")
        ttk.Label(status, textvariable=self.status_var).pack(side=tk.LEFT)
        ttk.Label(status, textvariable=self.zoom_var).pack(side=tk.RIGHT)

    def current_image_index(self) -> int:
        if not self.filtered_indices:
            return 0
        return self.filtered_indices[self.current_filtered_index]

    def current_path(self) -> Path:
        return self.image_paths[self.current_image_index()]

    def apply_filter(self, filter_name: str, keep_image: bool) -> None:
        current_path = self.current_path() if keep_image and self.image_paths and self.filtered_indices else None
        indices: list[int] = []
        for index, path in enumerate(self.image_paths):
            if self.image_matches_filter(path, filter_name):
                indices.append(index)
        self.filtered_indices = indices or list(range(len(self.image_paths)))
        if current_path is not None and current_path in [self.image_paths[index] for index in self.filtered_indices]:
            self.current_filtered_index = [self.image_paths[index] for index in self.filtered_indices].index(current_path)
        else:
            self.current_filtered_index = 0
        if keep_image:
            self.load_current_image()

    def image_matches_filter(self, path: Path, filter_name: str) -> bool:
        if filter_name == "all":
            return True
        load_result = load_annotation_for_image(path, self.input_root, self.annotation_root)
        annotation = load_result.annotation
        if filter_name == "incomplete":
            return annotation.annotation_status not in {"complete", "locked"}
        if filter_name == "warnings":
            report = validate_annotation(annotation)
            return bool(report.errors or report.warnings)
        if filter_name == "40x":
            return annotation.magnification == "40x"
        if filter_name == "100x":
            return annotation.magnification == "100x"
        if filter_name == "crossings":
            return bool(annotation.crossings)
        return True

    def load_current_image(self) -> None:
        path = self.current_path()
        load_result = load_annotation_for_image(path, self.input_root, self.annotation_root)
        self.annotation = load_result.annotation
        self.loaded_mtime_ns = load_result.loaded_mtime_ns
        self.undo_stack = UndoStack()
        self.selected_object = None
        self.selected_point = None
        self.selected_instance_var.set("")
        self.cancel_active_drawing(redraw=False)

        try:
            with Image.open(path) as image:
                self.current_image = image.convert("RGB")
        except (OSError, UnidentifiedImageError) as exc:
            messagebox.showerror("Image Load Failed", f"{path}\n{exc}")
            self.current_image = None
        self.overlay_image = self.load_scp_overlay(path)
        if self.notes_text is not None:
            self.notes_text.delete("1.0", tk.END)
            self.notes_text.insert("1.0", self.annotation.notes)
            self.notes_text.edit_modified(False)
        self.populate_tree()
        self.build_object_properties()
        self.update_scp_aids()
        self.update_progress()
        self.fit_image()
        self.validate_current(show_message=False)
        if load_result.warnings:
            self.status_var.set("; ".join(load_result.warnings))

    def load_scp_overlay(self, image_path: Path) -> Image.Image | None:
        image_id = safe_image_id(image_path, self.input_root)
        overlay_path = self.scp_output_root / "overlays" / f"{image_id}_overlay.png"
        if not overlay_path.exists():
            return None
        try:
            with Image.open(overlay_path) as image:
                return image.convert("RGB")
        except OSError:
            return None

    def update_progress(self) -> None:
        if self.annotation is None:
            return
        self.progress_var.set(
            f"Image {self.current_filtered_index + 1}/{len(self.filtered_indices)} "
            f"({self.current_image_index() + 1}/{len(self.image_paths)}) | "
            f"{self.annotation.image_id} | {self.annotation.annotation_status}"
        )

    def update_scp_aids(self) -> None:
        if self.scp_aid_text is None or self.annotation is None:
            return
        image_id = self.annotation.image_id
        aids = [
            self.scp_output_root / "overlays" / f"{image_id}_overlay.png",
            self.scp_output_root / "masks" / f"{image_id}_brown_mask.png",
            self.scp_output_root / "masks" / f"{image_id}_tail_mask.png",
            self.scp_output_root / "masks" / f"{image_id}_head_mask.png",
            self.scp_output_root / "masks" / f"{image_id}_overlap_mask.png",
            self.scp_output_root / "head_connected_overlays" / f"{image_id}_head_connected_overlay.png",
            self.scp_output_root / "json" / f"{image_id}.json",
        ]
        text = "\n".join(f"{'OK' if path.exists() else '--'} {path.name}" for path in aids)
        self.scp_aid_text.configure(state="normal")
        self.scp_aid_text.delete("1.0", tk.END)
        self.scp_aid_text.insert("1.0", text)
        self.scp_aid_text.configure(state="disabled")

    def set_tool(self, tool: str) -> None:
        self.tool.set(tool)
        self.status_var.set(f"Tool: {tool}")

    def push_undo(self) -> None:
        if self.annotation is not None:
            self.undo_stack.push(self.annotation)

    def commit_edit(self, message: str = "Edited") -> None:
        if self.annotation is None:
            return
        self.annotation.annotation_status = "in_progress" if self.annotation.annotation_status == "not_started" else self.annotation.annotation_status
        self.populate_tree()
        self.build_object_properties()
        self.redraw()
        self.autosave(message)

    def autosave(self, message: str) -> None:
        result = self.save_current(show_message=False)
        if result:
            self.status_var.set(f"{message}; autosaved")

    def save_current(self, show_message: bool = False) -> bool:
        if self.annotation is None:
            return False
        if self.notes_text is not None:
            self.annotation.notes = self.notes_text.get("1.0", tk.END).strip()
        try:
            result = save_annotation(self.annotation, self.annotation_root, self.loaded_mtime_ns)
        except PermissionError as exc:
            messagebox.showerror("Save Blocked", str(exc))
            return False
        if result.conflict_path is not None:
            messagebox.showwarning(
                "Save Conflict",
                f"The annotation on disk changed after this image was loaded.\n"
                f"Your current copy was saved as:\n{result.conflict_path}",
            )
            self.loaded_mtime_ns = result.mtime_ns
            return False
        self.loaded_mtime_ns = result.mtime_ns
        if show_message:
            messagebox.showinfo("Saved", f"Saved annotation:\n{result.path}")
        return True

    def validate_current(self, show_message: bool = True) -> ValidationReport | None:
        if self.annotation is None:
            return None
        report = validate_annotation(self.annotation)
        self.render_validation(report)
        if show_message:
            if report.errors:
                messagebox.showwarning("Validation", f"{len(report.errors)} errors, {len(report.warnings)} warnings")
            else:
                messagebox.showinfo("Validation", f"No errors; {len(report.warnings)} warnings")
        return report

    def render_validation(self, report: ValidationReport) -> None:
        if self.validation_text is None:
            return
        lines = [f"Errors: {len(report.errors)} | Warnings: {len(report.warnings)}"]
        for issue in report.errors + report.warnings:
            prefix = "ERROR" if issue.severity == "error" else "WARN"
            target = f" [{issue.object_id}]" if issue.object_id else ""
            lines.append(f"{prefix} {issue.code}{target}: {issue.message}")
        self.validation_text.configure(state="normal")
        self.validation_text.delete("1.0", tk.END)
        self.validation_text.insert("1.0", "\n".join(lines))
        self.validation_text.configure(state="disabled")

    def redraw(self) -> None:
        if self.canvas is None or self.current_image is None:
            return
        base = self.current_image
        if self.show_overlay_var.get() and self.overlay_image is not None:
            overlay = self.overlay_image.resize(base.size, Image.Resampling.BILINEAR)
            base = Image.blend(base, overlay, 0.45)
        display_size = (max(1, int(round(base.width * self.view.scale))), max(1, int(round(base.height * self.view.scale))))
        display = base.resize(display_size, Image.Resampling.LANCZOS)
        self.display_photo = ImageTk.PhotoImage(display)
        self.canvas.delete("all")
        self.canvas.create_image(self.view.offset_x, self.view.offset_y, image=self.display_photo, anchor=tk.NW, tags=("image",))
        self.draw_annotations()
        self.canvas.configure(scrollregion=(0, 0, display_size[0] + max(0, self.view.offset_x), display_size[1] + max(0, self.view.offset_y)))
        self.zoom_var.set(f"{self.view.scale * 100:.0f}%")

    def draw_annotations(self) -> None:
        if self.canvas is None or self.annotation is None:
            return
        selected_instance = self.selected_instance_var.get()
        for index, instance in enumerate(self.annotation.sperm_instances):
            if self.selected_only_var.get() and selected_instance and instance.instance_id != selected_instance:
                continue
            color = instance_color(index)
            stipple = "gray50" if self.dim_unselected_var.get() and selected_instance and instance.instance_id != selected_instance else ""
            self.draw_instance(instance, color, stipple)
        for crossing in self.annotation.crossings:
            if crossing.center is None:
                continue
            cx, cy = self.view.image_to_canvas(crossing.center)
            radius = max(3, crossing.radius_pixels * self.view.scale)
            self.canvas.create_oval(cx - radius, cy - radius, cx + radius, cy + radius, outline="#ffff00", width=2)
            self.canvas.create_text(cx + radius + 8, cy, text=crossing.crossing_id, fill="#ffff00", anchor=tk.W)
        if self.active_polygon:
            self.draw_active_points(self.active_polygon, "#ffffff")
        if self.active_tail:
            self.draw_active_points(self.active_tail, "#00ffff")

    def draw_instance(self, instance: SpermInstance, color: str, stipple: str = "") -> None:
        if self.canvas is None:
            return
        head = instance.head
        width = 3 if self.selected_instance_var.get() == instance.instance_id else 2
        if head.type == "ellipse" and head.center is not None and head.axes is not None:
            cx, cy = self.view.image_to_canvas(head.center)
            rx, ry = head.axes[0] * self.view.scale, head.axes[1] * self.view.scale
            self.canvas.create_oval(cx - rx, cy - ry, cx + rx, cy + ry, outline=color, width=width, stipple=stipple)
        elif head.type == "polygon" and len(head.polygon) >= 2:
            coords = [coord for point in head.polygon + [head.polygon[0]] for coord in self.view.image_to_canvas(point)]
            self.canvas.create_line(*coords, fill=color, width=width, stipple=stipple)
        elif head.type == "point_only" and head.center is not None:
            self.draw_point(head.center, color, 5, fill=True)
        if len(instance.tail_centerline) >= 2:
            coords = [coord for point in instance.tail_centerline for coord in self.view.image_to_canvas(point)]
            self.canvas.create_line(*coords, fill=color, width=width, smooth=True, stipple=stipple)
        for point in instance.tail_centerline:
            self.draw_point(point, color, 3)
        if instance.neck_point is not None:
            self.draw_square(instance.neck_point, color, 5)
        if instance.distal_endpoint is not None:
            self.draw_point(instance.distal_endpoint, color, 5)
        label_point = instance.neck_point or (instance.tail_centerline[0] if instance.tail_centerline else instance.head.center)
        if label_point is not None:
            x, y = self.view.image_to_canvas(label_point)
            self.canvas.create_text(x + 8, y + 8, text=instance.instance_id, fill=color, anchor=tk.NW)

    def draw_active_points(self, points: list[list[float]], color: str) -> None:
        if self.canvas is None:
            return
        if len(points) >= 2:
            coords = [coord for point in points for coord in self.view.image_to_canvas(point)]
            self.canvas.create_line(*coords, fill=color, width=2, dash=(4, 3))
        for point in points:
            self.draw_point(point, color, 4)

    def draw_point(self, point: list[float], color: str, radius: int, fill: bool = False) -> None:
        if self.canvas is None:
            return
        x, y = self.view.image_to_canvas(point)
        kwargs = {"fill": color} if fill else {"outline": color}
        self.canvas.create_oval(x - radius, y - radius, x + radius, y + radius, width=2, **kwargs)

    def draw_square(self, point: list[float], color: str, radius: int) -> None:
        if self.canvas is None:
            return
        x, y = self.view.image_to_canvas(point)
        self.canvas.create_rectangle(x - radius, y - radius, x + radius, y + radius, fill=color, outline=color)

    def event_to_image_point(self, event: tk.Event) -> list[float]:
        if self.canvas is None:
            return [0.0, 0.0]
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        x, y = self.view.canvas_to_image((canvas_x, canvas_y))
        if self.annotation is not None:
            x = min(max(0.0, x), max(0.0, self.annotation.width - 1.0))
            y = min(max(0.0, y), max(0.0, self.annotation.height - 1.0))
        return [float(x), float(y)]

    def selected_or_new_instance(self) -> SpermInstance:
        if self.annotation is None:
            raise RuntimeError("No annotation loaded")
        instance_id = self.selected_instance_var.get()
        instance = self.annotation.get_instance(instance_id) if instance_id else None
        if instance is None:
            instance = SpermInstance(instance_id=self.annotation.next_instance_id())
            self.annotation.sperm_instances.append(instance)
            self.selected_instance_var.set(instance.instance_id)
            self.selected_object = ("instance", instance.instance_id)
        return instance

    def on_canvas_press(self, event: tk.Event) -> None:
        if self.annotation is None:
            return
        if self.tool.get() == TOOL_PAN or self.space_pan:
            self.on_pan_press(event)
            return
        point = self.event_to_image_point(event)
        if self.pending_continuation_field:
            self.set_pending_continuation(point)
            return
        tool = self.tool.get()
        if tool == TOOL_SELECT:
            self.select_nearest(point)
            if self.selected_point is not None:
                self.push_undo()
            self.drag_start_image = point
            return
        if tool == TOOL_HEAD_ELLIPSE:
            self.drag_start_image = point
            return
        if tool == TOOL_HEAD_POLYGON:
            self.active_polygon.append(point)
            self.redraw()
            return
        if tool == TOOL_NECK:
            self.push_undo()
            instance = self.selected_or_new_instance()
            instance.neck_point = point
            self.commit_edit("Neck point set")
            return
        if tool == TOOL_TAIL:
            self.active_tail.append(point)
            self.redraw()
            return
        if tool == TOOL_CROSSING:
            self.active_crossing_start = point

    def on_canvas_drag(self, event: tk.Event) -> None:
        if self.tool.get() == TOOL_PAN or self.space_pan:
            self.on_pan_drag(event)
            return
        point = self.event_to_image_point(event)
        if self.tool.get() == TOOL_SELECT and self.selected_point is not None:
            self.move_selected_point(point)

    def on_canvas_release(self, event: tk.Event) -> None:
        if self.tool.get() == TOOL_HEAD_ELLIPSE and self.drag_start_image is not None:
            end = self.event_to_image_point(event)
            start = self.drag_start_image
            self.push_undo()
            instance = self.selected_or_new_instance()
            instance.head = HeadAnnotation(
                type="ellipse",
                center=[(start[0] + end[0]) / 2.0, (start[1] + end[1]) / 2.0],
                axes=[max(1.0, abs(end[0] - start[0]) / 2.0), max(1.0, abs(end[1] - start[1]) / 2.0)],
                angle_degrees=0.0,
            )
            instance.head_visibility = "full"
            self.drag_start_image = None
            self.commit_edit("Head ellipse set")
        elif self.tool.get() == TOOL_CROSSING and self.active_crossing_start is not None:
            end = self.event_to_image_point(event)
            start = self.active_crossing_start
            radius = max(3.0, _distance(start, end))
            self.push_undo()
            crossing = CrossingAnnotation(
                crossing_id=self.annotation.next_crossing_id() if self.annotation else "crossing_001",
                center=start,
                radius_pixels=radius,
                involved_instance_ids=self.nearby_instance_ids(start, radius + 10.0),
            )
            if self.annotation is not None:
                self.annotation.crossings.append(crossing)
                for instance_id in crossing.involved_instance_ids:
                    instance = self.annotation.get_instance(instance_id)
                    if instance is not None and crossing.crossing_id not in instance.crossing_ids:
                        instance.crossing_ids.append(crossing.crossing_id)
                self.selected_object = ("crossing", crossing.crossing_id)
            self.active_crossing_start = None
            self.commit_edit("Crossing added")
        elif self.tool.get() == TOOL_SELECT and self.selected_point is not None:
            self.selected_point = None
            self.commit_edit("Point moved")

    def on_pan_press(self, event: tk.Event) -> None:
        self.pan_start = (event.x, event.y)

    def on_pan_drag(self, event: tk.Event) -> None:
        if self.pan_start is None:
            return
        dx = event.x - self.pan_start[0]
        dy = event.y - self.pan_start[1]
        self.pan_start = (event.x, event.y)
        self.view.pan(dx, dy)
        self.redraw()

    def on_mousewheel(self, event: tk.Event) -> None:
        self.zoom_from_event(event, 1.25 if event.delta > 0 else 0.8)

    def zoom_from_event(self, event: tk.Event, factor: float) -> None:
        if self.canvas is None:
            return
        self.view.zoom_at((self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)), factor)
        self.redraw()

    def on_space_press(self, _event: tk.Event) -> None:
        self.space_pan = True

    def on_space_release(self, _event: tk.Event) -> None:
        self.space_pan = False
        self.pan_start = None

    def finish_active_drawing(self) -> None:
        if self.annotation is None:
            return
        if self.active_polygon:
            if len(self.active_polygon) >= 3:
                self.push_undo()
                instance = self.selected_or_new_instance()
                instance.head = HeadAnnotation(type="polygon", polygon=list(self.active_polygon))
                instance.head_visibility = "full"
                self.active_polygon = []
                self.commit_edit("Head polygon set")
            return
        if self.active_tail:
            if len(self.active_tail) >= 2:
                self.push_undo()
                instance = self.selected_or_new_instance()
                instance.tail_centerline = list(self.active_tail)
                instance.distal_endpoint = list(self.active_tail[-1])
                instance.tail_visibility = "full"
                self.active_tail = []
                self.commit_edit("Tail centerline set")
            return

    def cancel_active_drawing(self, redraw: bool = True) -> None:
        self.active_polygon = []
        self.active_tail = []
        self.active_crossing_start = None
        self.pending_continuation_field = None
        if redraw:
            self.redraw()

    def add_instance(self) -> None:
        if self.annotation is None:
            return
        self.push_undo()
        instance = SpermInstance(instance_id=self.annotation.next_instance_id())
        self.annotation.sperm_instances.append(instance)
        self.selected_instance_var.set(instance.instance_id)
        self.selected_object = ("instance", instance.instance_id)
        self.commit_edit("Sperm instance added")

    def delete_selected(self) -> None:
        if self.annotation is None or self.selected_object is None:
            return
        kind, object_id = self.selected_object
        if kind == "instance":
            if not messagebox.askyesno("Delete Instance", f"Delete {object_id}?"):
                return
            self.push_undo()
            self.annotation.sperm_instances = [row for row in self.annotation.sperm_instances if row.instance_id != object_id]
            for crossing in self.annotation.crossings:
                crossing.involved_instance_ids = [row for row in crossing.involved_instance_ids if row != object_id]
                crossing.continuations = [row for row in crossing.continuations if row.instance_id != object_id]
            self.selected_object = None
            self.selected_instance_var.set("")
            self.commit_edit("Sperm instance deleted")
        elif kind == "crossing":
            self.push_undo()
            self.annotation.crossings = [row for row in self.annotation.crossings if row.crossing_id != object_id]
            for instance in self.annotation.sperm_instances:
                instance.crossing_ids = [row for row in instance.crossing_ids if row != object_id]
            self.selected_object = None
            self.commit_edit("Crossing deleted")

    def select_nearest(self, point: list[float]) -> None:
        if self.annotation is None:
            return
        best: tuple[float, tuple[str, str, int | None], tuple[str, str]] | None = None
        for instance in self.annotation.sperm_instances:
            candidates: list[tuple[str, int | None, list[float] | None]] = [
                ("head", None, instance.head.center),
                ("neck", None, instance.neck_point),
                ("endpoint", None, instance.distal_endpoint),
            ]
            candidates.extend(("tail", index, row) for index, row in enumerate(instance.tail_centerline))
            for part, index, candidate in candidates:
                if candidate is None:
                    continue
                distance = _distance(point, candidate)
                if best is None or distance < best[0]:
                    best = (distance, (part, instance.instance_id, index), ("instance", instance.instance_id))
        for crossing in self.annotation.crossings:
            if crossing.center is None:
                continue
            distance = _distance(point, crossing.center)
            if best is None or distance < best[0]:
                best = (distance, ("crossing_center", crossing.crossing_id, None), ("crossing", crossing.crossing_id))
        if best is not None and best[0] <= 20.0 / max(0.1, self.view.scale):
            self.selected_point = best[1]
            self.selected_object = best[2]
            if best[2][0] == "instance":
                self.selected_instance_var.set(best[2][1])
        else:
            self.selected_point = None
        self.populate_tree()
        self.build_object_properties()
        self.redraw()

    def move_selected_point(self, point: list[float]) -> None:
        if self.annotation is None or self.selected_point is None:
            return
        part, object_id, index = self.selected_point
        if part == "crossing_center":
            crossing = self.annotation.get_crossing(object_id)
            if crossing is not None:
                crossing.center = point
        else:
            instance = self.annotation.get_instance(object_id)
            if instance is None:
                return
            if part == "head":
                instance.head.center = point
            elif part == "neck":
                instance.neck_point = point
            elif part == "endpoint":
                instance.distal_endpoint = point
            elif part == "tail" and index is not None and 0 <= index < len(instance.tail_centerline):
                instance.tail_centerline[index] = point
        self.redraw()

    def nearby_instance_ids(self, center: list[float], radius: float) -> list[str]:
        if self.annotation is None:
            return []
        involved: list[str] = []
        for instance in self.annotation.sperm_instances:
            if any(_distance(center, point) <= radius for point in instance.tail_centerline):
                involved.append(instance.instance_id)
        return involved

    def set_pending_continuation(self, point: list[float]) -> None:
        if self.annotation is None or self.selected_object is None or self.pending_continuation_field is None:
            return
        if self.selected_object[0] != "crossing":
            return
        crossing = self.annotation.get_crossing(self.selected_object[1])
        instance_id = self.selected_instance_var.get()
        if crossing is None or not instance_id:
            return
        self.push_undo()
        continuation = next((row for row in crossing.continuations if row.instance_id == instance_id), None)
        if continuation is None:
            continuation = CrossingContinuation(instance_id=instance_id)
            crossing.continuations.append(continuation)
        if self.pending_continuation_field == "incoming":
            continuation.incoming_point = point
        else:
            continuation.outgoing_point = point
        if instance_id not in crossing.involved_instance_ids:
            crossing.involved_instance_ids.append(instance_id)
        instance = self.annotation.get_instance(instance_id)
        if instance is not None and crossing.crossing_id not in instance.crossing_ids:
            instance.crossing_ids.append(crossing.crossing_id)
        self.pending_continuation_field = None
        self.commit_edit("Crossing continuation updated")

    def populate_tree(self) -> None:
        if self.tree is None or self.annotation is None:
            return
        self.tree.delete(*self.tree.get_children())
        root_id = self.tree.insert("", tk.END, iid="image", text=f"Image: {self.annotation.image_id}", open=True)
        for instance in self.annotation.sperm_instances:
            inst_iid = f"instance:{instance.instance_id}"
            self.tree.insert(root_id, tk.END, iid=inst_iid, text=instance.instance_id, open=True)
            self.tree.insert(inst_iid, tk.END, iid=f"head:{instance.instance_id}", text=f"head: {instance.head.type}")
            self.tree.insert(inst_iid, tk.END, iid=f"neck:{instance.instance_id}", text="neck")
            self.tree.insert(inst_iid, tk.END, iid=f"tail:{instance.instance_id}", text=f"tail centerline ({len(instance.tail_centerline)} points)")
            self.tree.insert(inst_iid, tk.END, iid=f"endpoint:{instance.instance_id}", text="endpoint")
        for crossing in self.annotation.crossings:
            self.tree.insert(root_id, tk.END, iid=f"crossing:{crossing.crossing_id}", text=crossing.crossing_id)
        if self.selected_object is not None:
            iid = f"{self.selected_object[0]}:{self.selected_object[1]}"
            if self.tree.exists(iid):
                self.tree.selection_set(iid)

    def on_tree_select(self, _event: tk.Event) -> None:
        if self.tree is None:
            return
        selected = self.tree.selection()
        if not selected:
            return
        iid = str(selected[0])
        parts = iid.split(":", 1)
        if len(parts) != 2:
            return
        kind, object_id = parts
        if kind in {"head", "neck", "tail", "endpoint"}:
            kind = "instance"
        self.selected_object = (kind, object_id)
        if kind == "instance":
            self.selected_instance_var.set(object_id)
        self.build_object_properties()
        self.redraw()

    def build_object_properties(self) -> None:
        if self.object_frame is None or self.annotation is None:
            return
        for child in self.object_frame.winfo_children():
            child.destroy()
        ttk.Label(self.object_frame, text="Properties", style="Header.TLabel").grid(row=0, column=0, columnspan=2, sticky="w")
        row = 1
        self.add_image_properties(row)
        row += 4
        if self.selected_object is None:
            return
        kind, object_id = self.selected_object
        if kind == "instance":
            instance = self.annotation.get_instance(object_id)
            if instance is not None:
                self.add_instance_properties(instance, row)
        elif kind == "crossing":
            crossing = self.annotation.get_crossing(object_id)
            if crossing is not None:
                self.add_crossing_properties(crossing, row)

    def add_image_properties(self, row: int) -> None:
        if self.object_frame is None or self.annotation is None:
            return
        status = tk.StringVar(value=self.annotation.annotation_status)
        ttk.Label(self.object_frame, text="Image status").grid(row=row, column=0, sticky="w")
        combo = ttk.Combobox(self.object_frame, textvariable=status, values=["not_started", "in_progress", "review_needed", "complete", "locked"], state="readonly")
        combo.grid(row=row, column=1, sticky="ew")
        combo.bind("<<ComboboxSelected>>", lambda _event: self.update_image_status(status.get()))
        ttk.Label(self.object_frame, text="Selected sperm").grid(row=row + 1, column=0, sticky="w")
        values = [instance.instance_id for instance in self.annotation.sperm_instances]
        sel = ttk.Combobox(self.object_frame, textvariable=self.selected_instance_var, values=values, state="readonly")
        sel.grid(row=row + 1, column=1, sticky="ew")
        sel.bind("<<ComboboxSelected>>", lambda _event: self.on_selected_instance_combo())
        ttk.Label(self.object_frame, text=f"{self.annotation.width}x{self.annotation.height} {self.annotation.magnification}").grid(row=row + 2, column=0, columnspan=2, sticky="w")

    def add_instance_properties(self, instance: SpermInstance, row: int) -> None:
        if self.object_frame is None:
            return
        fields = [
            ("instance_status", ["full", "partial", "boundary_truncated", "occluded", "uncertain"]),
            ("head_visibility", ["full", "partial", "not_visible", "uncertain"]),
            ("tail_visibility", ["full", "partial", "uncertain"]),
            ("certainty", ["high", "medium", "low", "indeterminate"]),
        ]
        for label, values in fields:
            var = tk.StringVar(value=getattr(instance, label))
            ttk.Label(self.object_frame, text=label).grid(row=row, column=0, sticky="w")
            combo = ttk.Combobox(self.object_frame, textvariable=var, values=values, state="readonly")
            combo.grid(row=row, column=1, sticky="ew")
            combo.bind("<<ComboboxSelected>>", lambda _event, field=label, value_var=var: self.update_instance_field(instance.instance_id, field, value_var.get()))
            row += 1
        endpoint_visible = tk.BooleanVar(value=instance.distal_endpoint_visible)
        ttk.Checkbutton(
            self.object_frame,
            text="Endpoint visible",
            variable=endpoint_visible,
            command=lambda: self.update_instance_field(instance.instance_id, "distal_endpoint_visible", endpoint_visible.get()),
        ).grid(row=row, column=0, columnspan=2, sticky="w")
        row += 1
        ttk.Button(self.object_frame, text="Clear Tail", command=lambda: self.clear_instance_tail(instance.instance_id)).grid(row=row, column=0, sticky="ew")
        ttk.Button(self.object_frame, text="Delete Sperm", command=self.delete_selected).grid(row=row, column=1, sticky="ew")

    def add_crossing_properties(self, crossing: CrossingAnnotation, row: int) -> None:
        if self.object_frame is None:
            return
        determinacy = tk.StringVar(value=crossing.determinacy)
        ttk.Label(self.object_frame, text="determinacy").grid(row=row, column=0, sticky="w")
        combo = ttk.Combobox(self.object_frame, textvariable=determinacy, values=["determinable", "indeterminate", "uncertain"], state="readonly")
        combo.grid(row=row, column=1, sticky="ew")
        combo.bind("<<ComboboxSelected>>", lambda _event: self.update_crossing_determinacy(crossing.crossing_id, determinacy.get()))
        row += 1
        ttk.Button(self.object_frame, text="Set incoming point", command=lambda: self.begin_continuation("incoming")).grid(row=row, column=0, sticky="ew")
        ttk.Button(self.object_frame, text="Set outgoing point", command=lambda: self.begin_continuation("outgoing")).grid(row=row, column=1, sticky="ew")
        row += 1
        ttk.Button(self.object_frame, text="Delete Crossing", command=self.delete_selected).grid(row=row, column=0, columnspan=2, sticky="ew")

    def update_image_status(self, status: str) -> None:
        if self.annotation is None:
            return
        if status in {"complete", "locked"}:
            report = validate_annotation(self.annotation)
            self.render_validation(report)
            if report.errors:
                messagebox.showwarning("Validation Required", "Resolve validation errors before marking complete or locked.")
                self.build_object_properties()
                return
        self.push_undo()
        self.annotation.annotation_status = status
        self.commit_edit("Image status updated")

    def update_instance_field(self, instance_id: str, field: str, value: Any) -> None:
        if self.annotation is None:
            return
        instance = self.annotation.get_instance(instance_id)
        if instance is None:
            return
        self.push_undo()
        setattr(instance, field, value)
        self.commit_edit("Instance property updated")

    def update_crossing_determinacy(self, crossing_id: str, determinacy: str) -> None:
        if self.annotation is None:
            return
        crossing = self.annotation.get_crossing(crossing_id)
        if crossing is None:
            return
        self.push_undo()
        crossing.determinacy = determinacy
        self.commit_edit("Crossing property updated")

    def clear_instance_tail(self, instance_id: str) -> None:
        if self.annotation is None:
            return
        instance = self.annotation.get_instance(instance_id)
        if instance is None:
            return
        self.push_undo()
        instance.tail_centerline = []
        instance.distal_endpoint = None
        self.commit_edit("Tail cleared")

    def begin_continuation(self, field: str) -> None:
        if self.selected_object is None or self.selected_object[0] != "crossing":
            messagebox.showinfo("Select Crossing", "Select a crossing first.")
            return
        if not self.selected_instance_var.get():
            messagebox.showinfo("Select Sperm", "Select the sperm instance for this continuation.")
            return
        self.pending_continuation_field = field
        self.status_var.set(f"Click {field} continuation point for {self.selected_instance_var.get()}")

    def on_selected_instance_combo(self) -> None:
        instance_id = self.selected_instance_var.get()
        if instance_id:
            self.selected_object = ("instance", instance_id)
        self.populate_tree()
        self.build_object_properties()
        self.redraw()

    def on_notes_modified(self, _event: tk.Event) -> None:
        if self.notes_text is not None and self.notes_text.edit_modified():
            if self.annotation is not None:
                self.annotation.notes = self.notes_text.get("1.0", tk.END).strip()
            self.notes_text.edit_modified(False)

    def undo(self) -> None:
        if self.annotation is None:
            return
        previous = self.undo_stack.undo(self.annotation)
        if previous is None:
            return
        self.annotation = previous
        self.commit_edit("Undo")

    def redo(self) -> None:
        if self.annotation is None:
            return
        next_annotation = self.undo_stack.redo(self.annotation)
        if next_annotation is None:
            return
        self.annotation = next_annotation
        self.commit_edit("Redo")

    def previous_image(self) -> None:
        if not self.filtered_indices:
            return
        self.save_current(show_message=False)
        self.current_filtered_index = max(0, self.current_filtered_index - 1)
        self.load_current_image()

    def next_image(self) -> None:
        if not self.filtered_indices:
            return
        self.save_current(show_message=False)
        self.current_filtered_index = min(len(self.filtered_indices) - 1, self.current_filtered_index + 1)
        self.load_current_image()

    def toggle_overlay(self) -> None:
        self.show_overlay_var.set(not self.show_overlay_var.get())
        self.redraw()

    def toggle_selected_only(self) -> None:
        self.selected_only_var.set(not self.selected_only_var.get())
        self.redraw()

    def fit_image(self) -> None:
        if self.canvas is None or self.current_image is None:
            return
        self.root.update_idletasks()
        self.view.fit(self.current_image.size, (max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height())))
        self.redraw()

    def zoom_actual(self) -> None:
        self.view.scale = 1.0
        self.view.offset_x = 0.0
        self.view.offset_y = 0.0
        self.redraw()

    def on_close(self) -> None:
        self.save_current(show_message=False)
        self.root.destroy()


def run_check_only(input_root: Path, annotation_root: Path, scp_output_root: Path) -> dict[str, Any]:
    images = collect_source_images(input_root)
    readable = 0
    annotations_loaded = 0
    scp_aids_found = 0
    for image_path in images:
        with Image.open(image_path) as image:
            image.verify()
        readable += 1
        load_annotation_for_image(image_path, input_root, annotation_root)
        annotations_loaded += 1
        image_id = safe_image_id(image_path, input_root)
        if (resolve_scp_path(scp_output_root) / "overlays" / f"{image_id}_overlay.png").exists():
            scp_aids_found += 1
    return {
        "images": len(images),
        "readable_images": readable,
        "annotations_loadable_or_blank": annotations_loaded,
        "scp_overlay_aids_found": scp_aids_found,
        "annotation_root": str(resolve_scp_path(annotation_root)),
    }


def main() -> None:
    args = parse_args()
    input_root = resolve_scp_path(args.input_root)
    annotation_root = resolve_scp_path(args.annotation_root)
    scp_output_root = resolve_scp_path(args.scp_output_root)

    if args.check_only:
        print(json.dumps(run_check_only(input_root, annotation_root, scp_output_root), indent=2))
        return

    image_paths = collect_source_images(input_root)
    if not image_paths:
        raise RuntimeError(f"No real Ward images found under {input_root}")

    start_index = max(0, args.start_index)
    if args.image_id:
        for index, image_path in enumerate(image_paths):
            if safe_image_id(image_path, input_root) == args.image_id:
                start_index = index
                break
        else:
            raise RuntimeError(f"Image ID not found: {args.image_id}")

    root = tk.Tk()
    AnnotationApp(
        root=root,
        image_paths=image_paths,
        input_root=input_root,
        annotation_root=annotation_root,
        scp_output_root=scp_output_root,
        start_index=start_index,
        initial_filter=args.filter,
        ui_scale=args.ui_scale,
    )
    root.mainloop()


if __name__ == "__main__":
    main()
