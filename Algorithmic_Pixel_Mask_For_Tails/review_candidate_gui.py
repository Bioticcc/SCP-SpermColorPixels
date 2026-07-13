#!/usr/bin/env python3
"""Tkinter GUI for manually reviewing SCP crop candidates."""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path
from typing import Any

from PIL import Image, ImageTk, UnidentifiedImageError
import tkinter as tk
import tkinter.font as tkfont
from tkinter import messagebox, ttk


SCP_ROOT = Path(__file__).resolve().parent

CROP_LABEL_OPTIONS = [
    "",
    "good_full_sperm",
    "partial_or_cut_off",
    "merged_multi_sperm",
    "duplicate",
    "debris_or_false_positive",
    "unclear",
]

MASK_LABEL_OPTIONS = [
    "",
    "mask_good",
    "tail_mask_partial",
    "head_mask_wrong",
    "wrong_tail_assigned",
    "extra_tail_assigned",
    "mask_unclear",
]

HUMAN_COLUMNS = ["crop_label", "mask_label", "review_notes", "reviewer", "reviewed_at"]

IMPORTANT_STATS = [
    ("Current Status", "current_status"),
    ("Selection", "selection_reason"),
    ("Score", "score"),
    ("Path Mode", "path_mode"),
    ("Path Scores", "path_score_hybrid"),
    ("Raw/Norm Color", "color_score_raw"),
    ("Ambiguity", "ambiguity_reason"),
    ("Rejection", "rejection_reason"),
    ("Cutoff Stage", "tail_cutoff_stage"),
    ("Boundary Risk", "crop_boundary_risk"),
    ("Image", "relative_image"),
    ("Crop ID", "crop_id"),
    ("Tail ID", "tail_id"),
    ("Head ID", "head_label_id"),
    ("Assignment", "tail_assignment_mode"),
    ("Overlap", "tail_has_overlap"),
    ("Overlap Reasons", "tail_overlap_reasons"),
    ("Touching Heads", "touching_head_count"),
    ("Foreign Content", "foreign_content_score"),
    ("Foreign Crop Area", "foreign_content_by_crop_area"),
    ("Head Anchor Dist", "head_anchor_distance_px"),
    ("Endpoint Crop Edge", "endpoint_to_crop_edge_px"),
    ("Tail Reuse", "tail_reuse_accepted_count"),
    ("Shared Path", "shared_path_fraction"),
    ("Tail Pixels", "assigned_tail_pixels"),
    ("Skeleton Pixels", "assigned_skeleton_pixels"),
    ("Branchpoints", "tail_branchpoint_count"),
    ("Endpoints", "tail_endpoint_count"),
    ("Duplicate Of", "duplicate_of_crop_id"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Review SCP candidate crops from a CSV queue.")
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("outputs/updated_outputs/candidate_review/candidate_review_queue.csv"),
        help="Candidate review CSV. Relative paths resolve inside Algorithmic_Pixel_Mask_For_Tails/.",
    )
    parser.add_argument(
        "--ui-scale",
        type=float,
        default=1.25,
        help="Scale factor for fonts and controls.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate the CSV and image paths without opening the GUI.",
    )
    return parser.parse_args()


def resolve_csv_path(csv_path: Path) -> Path:
    if csv_path.is_absolute():
        return csv_path
    return (SCP_ROOT / csv_path).resolve()


def load_review_csv(csv_path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Review CSV not found: {csv_path}")

    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]

    for column in HUMAN_COLUMNS:
        if column not in fieldnames:
            fieldnames.append(column)
        for row in rows:
            row.setdefault(column, "")

    return fieldnames, rows


def write_review_csv(csv_path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    tmp_path = csv_path.with_suffix(csv_path.suffix + ".tmp")
    with tmp_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    tmp_path.replace(csv_path)


def validate_rows(rows: list[dict[str, str]]) -> dict[str, int]:
    missing_review_images = 0
    missing_highlighted = 0
    missing_raw = 0
    reviewed = 0

    for row in rows:
        if row.get("crop_label") or row.get("mask_label"):
            reviewed += 1
        if row.get("review_image_path") and not Path(row["review_image_path"]).exists():
            missing_review_images += 1
        if row.get("highlighted_crop_path") and not Path(row["highlighted_crop_path"]).exists():
            missing_highlighted += 1
        if row.get("raw_crop_path") and not Path(row["raw_crop_path"]).exists():
            missing_raw += 1

    return {
        "rows": len(rows),
        "reviewed": reviewed,
        "missing_review_images": missing_review_images,
        "missing_highlighted": missing_highlighted,
        "missing_raw": missing_raw,
    }


class CandidateReviewApp:
    def __init__(
        self,
        root: tk.Tk,
        csv_path: Path,
        fieldnames: list[str],
        rows: list[dict[str, str]],
        ui_scale: float,
    ) -> None:
        self.root = root
        self.csv_path = csv_path
        self.fieldnames = fieldnames
        self.rows = rows
        self.index = self.first_unreviewed_index()
        self.photo: ImageTk.PhotoImage | None = None
        self.current_image: Image.Image | None = None
        self.unsaved_changes = False
        self.backup_written = False
        self.ui_scale = max(1.0, ui_scale)

        self.crop_label_var = tk.StringVar()
        self.mask_label_var = tk.StringVar()
        self.reviewer_var = tk.StringVar()
        self.reviewed_at_var = tk.StringVar()
        self.view_mode_var = tk.StringVar(value="Highlighted")
        self.status_var = tk.StringVar()
        self.notes_text: tk.Text | None = None
        self.stats_text: tk.Text | None = None
        self.image_label: ttk.Label | None = None
        self.progress_label: ttk.Label | None = None
        self.file_label: ttk.Label | None = None

        self.configure_root()
        self.build_layout()
        self.load_row()

    def configure_root(self) -> None:
        self.root.title("SCP Candidate Review")
        self.root.geometry("1500x950")
        self.root.minsize(1100, 720)

        base_size = int(round(12 * self.ui_scale))
        title_size = int(round(15 * self.ui_scale))
        button_size = int(round(13 * self.ui_scale))

        for font_name in ("TkDefaultFont", "TkTextFont", "TkMenuFont"):
            try:
                tkfont.nametofont(font_name).configure(size=base_size)
            except tk.TclError:
                pass
        try:
            tkfont.nametofont("TkHeadingFont").configure(size=title_size, weight="bold")
        except tk.TclError:
            pass

        self.root.option_add("*TButton.Font", ("TkDefaultFont", button_size))
        self.root.option_add("*TCombobox.Font", ("TkDefaultFont", base_size))
        self.root.option_add("*TLabel.Font", ("TkDefaultFont", base_size))

        style = ttk.Style()
        style.configure("TButton", padding=(10, 8), font=("TkDefaultFont", button_size))
        style.configure("TLabel", font=("TkDefaultFont", base_size))
        style.configure("Primary.TButton", padding=(14, 10))
        style.configure("Nav.TButton", padding=(12, 9))
        style.configure("Panel.TFrame", padding=10)
        style.configure("Title.TLabel", font=("TkDefaultFont", title_size, "bold"))

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.bind("<Control-s>", lambda _event: self.save_current())
        self.root.bind("<Left>", lambda _event: self.previous_row())
        self.root.bind("<Right>", lambda _event: self.next_row())
        self.root.bind("<Control-Right>", lambda _event: self.save_and_next())

    def build_layout(self) -> None:
        outer = ttk.Frame(self.root, padding=12)
        outer.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        outer.columnconfigure(0, weight=4)
        outer.columnconfigure(1, weight=2)
        outer.rowconfigure(1, weight=1)

        header = ttk.Frame(outer)
        header.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        header.columnconfigure(1, weight=1)
        ttk.Label(header, text="SCP Candidate Review", style="Title.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        self.progress_label = ttk.Label(header, text="")
        self.progress_label.grid(row=0, column=1, sticky="e")

        image_panel = ttk.Frame(outer, style="Panel.TFrame")
        image_panel.grid(row=1, column=0, sticky="nsew", padx=(0, 10))
        image_panel.columnconfigure(0, weight=1)
        image_panel.rowconfigure(1, weight=1)

        view_bar = ttk.Frame(image_panel)
        view_bar.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        ttk.Label(view_bar, text="View").grid(row=0, column=0, sticky="w", padx=(0, 8))
        view_menu = ttk.Combobox(
            view_bar,
            textvariable=self.view_mode_var,
            values=["Highlighted", "Raw Crop", "Skeleton", "Mask"],
            state="readonly",
            width=16,
        )
        view_menu.grid(row=0, column=1, sticky="w")
        view_menu.bind("<<ComboboxSelected>>", lambda _event: self.refresh_image())
        self.file_label = ttk.Label(view_bar, text="")
        self.file_label.grid(row=0, column=2, sticky="e", padx=(12, 0))
        view_bar.columnconfigure(2, weight=1)

        image_frame = ttk.Frame(image_panel, relief="solid", borderwidth=1)
        image_frame.grid(row=1, column=0, sticky="nsew")
        image_frame.columnconfigure(0, weight=1)
        image_frame.rowconfigure(0, weight=1)
        self.image_label = ttk.Label(image_frame, anchor="center")
        self.image_label.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
        self.image_label.bind("<Configure>", lambda _event: self.render_image())

        nav = ttk.Frame(image_panel)
        nav.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        for col in range(5):
            nav.columnconfigure(col, weight=1)
        ttk.Button(nav, text="Previous", style="Nav.TButton", command=self.previous_row).grid(
            row=0, column=0, sticky="ew", padx=4
        )
        ttk.Button(nav, text="Next", style="Nav.TButton", command=self.next_row).grid(
            row=0, column=1, sticky="ew", padx=4
        )
        ttk.Button(nav, text="Next Unlabeled", style="Nav.TButton", command=self.next_unreviewed).grid(
            row=0, column=2, sticky="ew", padx=4
        )
        ttk.Button(nav, text="Save", style="Primary.TButton", command=self.save_current).grid(
            row=0, column=3, sticky="ew", padx=4
        )
        ttk.Button(nav, text="Save & Next", style="Primary.TButton", command=self.save_and_next).grid(
            row=0, column=4, sticky="ew", padx=4
        )

        side = ttk.Frame(outer, style="Panel.TFrame")
        side.grid(row=1, column=1, sticky="nsew")
        side.columnconfigure(0, weight=1)
        side.rowconfigure(7, weight=1)

        ttk.Label(side, text="Manual Labels", style="Title.TLabel").grid(
            row=0, column=0, sticky="w", pady=(0, 10)
        )
        self.add_labeled_combo(side, "Crop Label", self.crop_label_var, CROP_LABEL_OPTIONS, 1)
        self.add_labeled_combo(side, "Mask Label", self.mask_label_var, MASK_LABEL_OPTIONS, 2)
        self.add_labeled_entry(side, "Reviewer", self.reviewer_var, 3)
        self.add_labeled_entry(side, "Reviewed At", self.reviewed_at_var, 4)

        ttk.Label(side, text="Notes").grid(row=5, column=0, sticky="w", pady=(10, 4))
        self.notes_text = tk.Text(side, height=6, wrap="word", font=("TkTextFont", int(12 * self.ui_scale)))
        self.notes_text.grid(row=6, column=0, sticky="ew")
        self.notes_text.bind("<<Modified>>", self.on_notes_modified)

        ttk.Label(side, text="Candidate Stats", style="Title.TLabel").grid(
            row=7, column=0, sticky="sw", pady=(16, 6)
        )
        self.stats_text = tk.Text(
            side,
            height=18,
            wrap="word",
            font=("TkTextFont", int(11 * self.ui_scale)),
            state="disabled",
        )
        self.stats_text.grid(row=8, column=0, sticky="nsew")

        self.status_var.set("")
        ttk.Label(side, textvariable=self.status_var).grid(row=9, column=0, sticky="ew", pady=(10, 0))

        for var in (self.crop_label_var, self.mask_label_var, self.reviewer_var, self.reviewed_at_var):
            var.trace_add("write", self.mark_unsaved)

    def add_labeled_combo(
        self,
        parent: ttk.Frame,
        label: str,
        variable: tk.StringVar,
        values: list[str],
        row: int,
    ) -> None:
        frame = ttk.Frame(parent)
        frame.grid(row=row, column=0, sticky="ew", pady=5)
        frame.columnconfigure(1, weight=1)
        ttk.Label(frame, text=label).grid(row=0, column=0, sticky="w", padx=(0, 10))
        combo = ttk.Combobox(frame, textvariable=variable, values=values, state="readonly")
        combo.grid(row=0, column=1, sticky="ew")

    def add_labeled_entry(
        self,
        parent: ttk.Frame,
        label: str,
        variable: tk.StringVar,
        row: int,
    ) -> None:
        frame = ttk.Frame(parent)
        frame.grid(row=row, column=0, sticky="ew", pady=5)
        frame.columnconfigure(1, weight=1)
        ttk.Label(frame, text=label).grid(row=0, column=0, sticky="w", padx=(0, 10))
        entry = ttk.Entry(frame, textvariable=variable)
        entry.grid(row=0, column=1, sticky="ew")

    def first_unreviewed_index(self) -> int:
        for index, row in enumerate(self.rows):
            if not row.get("crop_label") and not row.get("mask_label"):
                return index
        return 0

    def mark_unsaved(self, *_args: Any) -> None:
        self.unsaved_changes = True

    def on_notes_modified(self, _event: tk.Event[tk.Text]) -> None:
        if self.notes_text is not None and self.notes_text.edit_modified():
            self.unsaved_changes = True
            self.notes_text.edit_modified(False)

    def active_row(self) -> dict[str, str]:
        return self.rows[self.index]

    def image_path_for_mode(self, row: dict[str, str]) -> str:
        mode = self.view_mode_var.get()
        if mode == "Raw Crop":
            return row.get("raw_crop_path") or row.get("review_image_path", "")
        if mode == "Skeleton":
            return row.get("skeleton_overlay_path") or row.get("review_image_path", "")
        if mode == "Mask":
            return row.get("crop_mask_path") or row.get("review_image_path", "")
        return row.get("highlighted_crop_path") or row.get("review_image_path", "")

    def load_row(self) -> None:
        row = self.active_row()
        self.unsaved_changes = False
        self.crop_label_var.set(row.get("crop_label", ""))
        self.mask_label_var.set(row.get("mask_label", ""))
        self.reviewer_var.set(row.get("reviewer", ""))
        self.reviewed_at_var.set(row.get("reviewed_at", ""))
        if self.notes_text is not None:
            self.notes_text.delete("1.0", "end")
            self.notes_text.insert("1.0", row.get("review_notes", ""))
            self.notes_text.edit_modified(False)
        self.update_progress()
        self.update_stats()
        self.refresh_image()
        self.unsaved_changes = False
        self.status_var.set("")

    def update_progress(self) -> None:
        reviewed = sum(1 for row in self.rows if row.get("crop_label") or row.get("mask_label"))
        current = self.index + 1
        total = len(self.rows)
        if self.progress_label is not None:
            self.progress_label.configure(text=f"Candidate {current} of {total}   Reviewed {reviewed}/{total}")

    def update_stats(self) -> None:
        if self.stats_text is None:
            return
        row = self.active_row()
        lines: list[str] = []
        for label, key in IMPORTANT_STATS:
            value = row.get(key, "")
            if value:
                lines.append(f"{label}: {value}")
        lines.append("")
        lines.append(f"Highlighted: {row.get('highlighted_crop_path', '')}")
        lines.append(f"Raw: {row.get('raw_crop_path', '')}")

        self.stats_text.configure(state="normal")
        self.stats_text.delete("1.0", "end")
        self.stats_text.insert("1.0", "\n".join(lines))
        self.stats_text.configure(state="disabled")

    def refresh_image(self) -> None:
        if self.image_label is None:
            return
        row = self.active_row()
        image_path = self.image_path_for_mode(row)
        if self.file_label is not None:
            self.file_label.configure(text=Path(image_path).name if image_path else "")

        if not image_path or not Path(image_path).exists():
            self.image_label.configure(text=f"Image not found:\n{image_path}", image="")
            self.photo = None
            return

        try:
            with Image.open(image_path) as image:
                self.current_image = image.convert("RGB")
        except (OSError, UnidentifiedImageError) as exc:
            self.image_label.configure(text=f"Could not open image:\n{image_path}\n{exc}", image="")
            self.photo = None
            return

        self.render_image()

    def render_image(self) -> None:
        if self.current_image is None or self.image_label is None:
            return
        self.root.update_idletasks()
        max_width = max(400, self.image_label.winfo_width() - 20)
        max_height = max(300, self.image_label.winfo_height() - 20)
        image = self.current_image
        scale = min(max_width / image.width, max_height / image.height, 1.0)
        display_size = (
            max(1, int(round(image.width * scale))),
            max(1, int(round(image.height * scale))),
        )
        display = image.resize(display_size, Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(display)
        self.image_label.configure(image=self.photo, text="")

    def capture_form(self) -> None:
        row = self.active_row()
        row["crop_label"] = self.crop_label_var.get().strip()
        row["mask_label"] = self.mask_label_var.get().strip()
        row["reviewer"] = self.reviewer_var.get().strip()
        row["reviewed_at"] = self.reviewed_at_var.get().strip()
        if self.notes_text is not None:
            row["review_notes"] = self.notes_text.get("1.0", "end").strip()

    def maybe_save_before_navigation(self) -> bool:
        if not self.unsaved_changes:
            return True
        self.save_current(show_message=False)
        return True

    def write_backup_once(self) -> None:
        if self.backup_written:
            return
        backup_path = self.csv_path.with_suffix(self.csv_path.suffix + ".bak")
        if not backup_path.exists():
            shutil.copy2(self.csv_path, backup_path)
        self.backup_written = True

    def save_current(self, show_message: bool = True) -> None:
        self.capture_form()
        self.write_backup_once()
        write_review_csv(self.csv_path, self.fieldnames, self.rows)
        self.unsaved_changes = False
        self.update_progress()
        self.status_var.set(f"Saved to {self.csv_path.name}")
        if show_message:
            messagebox.showinfo("Saved", f"Saved review CSV:\n{self.csv_path}")

    def save_and_next(self) -> None:
        self.save_current(show_message=False)
        self.next_row()

    def previous_row(self) -> None:
        if not self.maybe_save_before_navigation():
            return
        self.index = max(0, self.index - 1)
        self.load_row()

    def next_row(self) -> None:
        if not self.maybe_save_before_navigation():
            return
        self.index = min(len(self.rows) - 1, self.index + 1)
        self.load_row()

    def next_unreviewed(self) -> None:
        if not self.maybe_save_before_navigation():
            return
        for next_index in range(self.index + 1, len(self.rows)):
            row = self.rows[next_index]
            if not row.get("crop_label") and not row.get("mask_label"):
                self.index = next_index
                self.load_row()
                return
        messagebox.showinfo("Review Complete", "No later unlabeled candidates found.")

    def on_close(self) -> None:
        if self.unsaved_changes:
            answer = messagebox.askyesnocancel("Unsaved Changes", "Save changes before closing?")
            if answer is None:
                return
            if answer:
                self.save_current(show_message=False)
        self.root.destroy()


def main() -> None:
    args = parse_args()
    csv_path = resolve_csv_path(args.csv)
    fieldnames, rows = load_review_csv(csv_path)

    if args.check_only:
        print(validate_rows(rows))
        return

    if not rows:
        raise RuntimeError(f"No review rows found in {csv_path}")

    root = tk.Tk()
    CandidateReviewApp(
        root=root,
        csv_path=csv_path,
        fieldnames=fieldnames,
        rows=rows,
        ui_scale=args.ui_scale,
    )
    root.mainloop()


if __name__ == "__main__":
    main()
