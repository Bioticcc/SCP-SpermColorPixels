#!/usr/bin/env python3
"""Summarize completed SCP candidate-review labels for before/after tuning."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Iterable


SCP_ROOT = Path(__file__).resolve().parent
DEFAULT_REVIEW_CSV = Path("outputs/updated_outputs/candidate_review/candidate_review_queue.csv")
SCORE_BANDS = [
    ("<40", None, 40.0),
    ("40-60", 40.0, 60.0),
    ("60-75", 60.0, 75.0),
    ("75-90", 75.0, 90.0),
    (">=90", 90.0, None),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create reusable summary tables from a labeled SCP candidate-review CSV."
    )
    parser.add_argument(
        "--review-csv",
        type=Path,
        default=DEFAULT_REVIEW_CSV,
        help="Completed candidate_review_queue.csv. Relative paths resolve inside this SCP folder.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Folder for summary tables. Defaults to <review-csv parent>/baseline_analysis.",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (SCP_ROOT / path).resolve()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def clean_value(value: str | None, blank: str = "(blank)") -> str:
    value = (value or "").strip()
    return value if value else blank


def assignment_reason(row: dict[str, str]) -> str:
    for field in (
        "rejection_reason",
        "ambiguity_reason",
        "path_rejection_reason",
        "path_ambiguity_reason",
    ):
        value = clean_value(row.get(field), blank="")
        if value:
            return value
    return clean_value(row.get("tail_assignment_mode"), blank="no_reason_recorded")


def score_as_float(row: dict[str, str], field: str = "score") -> float | None:
    try:
        return float(row.get(field, ""))
    except ValueError:
        return None


def score_band(score: float | None) -> str:
    if score is None:
        return "(missing)"
    for label, lower, upper in SCORE_BANDS:
        if lower is not None and score < lower:
            continue
        if upper is not None and score >= upper:
            continue
        return label
    return "(missing)"


def counter_rows(counter: Counter[tuple[str, ...]], fields: list[str], total: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for key, count in counter.most_common():
        row = {field: value for field, value in zip(fields, key)}
        row["count"] = str(count)
        row["percent"] = f"{(100.0 * count / max(1, total)):.2f}"
        rows.append(row)
    return rows


def grouped_summary(
    rows: Iterable[dict[str, str]],
    fields: list[str],
    total: int,
) -> list[dict[str, str]]:
    counter: Counter[tuple[str, ...]] = Counter()
    for row in rows:
        key = tuple(clean_value(row.get(field), blank="(unlabeled)") for field in fields)
        counter[key] += 1
    return counter_rows(counter, fields, total)


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def note_summary(rows: list[dict[str, str]], total: int) -> list[dict[str, str]]:
    counter: Counter[tuple[str, str, str]] = Counter()
    for row in rows:
        note = clean_value(row.get("review_notes"), blank="")
        if not note:
            continue
        normalized = " ".join(note.lower().split())
        if len(normalized) > 160:
            normalized = normalized[:157] + "..."
        counter[
            (
                normalized,
                clean_value(row.get("crop_label"), blank="(unlabeled)"),
                clean_value(row.get("mask_label"), blank="(unlabeled)"),
            )
        ] += 1
    return counter_rows(counter, ["review_notes_normalized", "crop_label", "mask_label"], total)


def main() -> None:
    args = parse_args()
    review_csv = resolve_path(args.review_csv)
    output_dir = (
        resolve_path(args.output_dir)
        if args.output_dir is not None
        else review_csv.parent / "baseline_analysis"
    )
    rows = read_rows(review_csv)
    total = len(rows)
    labeled_rows = [
        row
        for row in rows
        if clean_value(row.get("crop_label"), blank="")
        or clean_value(row.get("mask_label"), blank="")
        or clean_value(row.get("review_notes"), blank="")
    ]

    assignment_rows = []
    for row in rows:
        enriched = dict(row)
        enriched["assignment_reason"] = assignment_reason(row)
        enriched["score_band"] = score_band(score_as_float(row))
        assignment_rows.append(enriched)

    outputs: dict[str, str] = {}
    summaries = {
        "crop_label_summary.csv": (
            grouped_summary(rows, ["crop_label"], total),
            ["crop_label", "count", "percent"],
        ),
        "mask_label_summary.csv": (
            grouped_summary(rows, ["mask_label"], total),
            ["mask_label", "count", "percent"],
        ),
        "status_summary.csv": (
            grouped_summary(rows, ["current_status"], total),
            ["current_status", "count", "percent"],
        ),
        "label_status_summary.csv": (
            grouped_summary(rows, ["crop_label", "mask_label", "current_status"], total),
            ["crop_label", "mask_label", "current_status", "count", "percent"],
        ),
        "assignment_reason_summary.csv": (
            grouped_summary(assignment_rows, ["current_status", "assignment_reason"], total),
            ["current_status", "assignment_reason", "count", "percent"],
        ),
        "score_band_summary.csv": (
            grouped_summary(assignment_rows, ["score_band", "current_status", "mask_label"], total),
            ["score_band", "current_status", "mask_label", "count", "percent"],
        ),
        "review_notes_summary.csv": (
            note_summary(rows, total),
            ["review_notes_normalized", "crop_label", "mask_label", "count", "percent"],
        ),
    }

    for filename, (summary_rows, fieldnames) in summaries.items():
        path = output_dir / filename
        write_csv(path, summary_rows, fieldnames)
        outputs[filename] = str(path)

    summary = {
        "review_csv": str(review_csv),
        "output_dir": str(output_dir),
        "total_rows": total,
        "labeled_rows": len(labeled_rows),
        "unlabeled_rows": total - len(labeled_rows),
        "tables": outputs,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({**summary, "summary_json": str(summary_path)}, indent=2))


if __name__ == "__main__":
    main()
