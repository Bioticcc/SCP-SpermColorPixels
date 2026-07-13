#!/usr/bin/env python3
"""Build CSV review queues from SCP crop-candidate JSON outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


SCP_ROOT = Path(__file__).resolve().parent

CROP_LABEL_OPTIONS = [
    "good_full_sperm",
    "partial_or_cut_off",
    "merged_multi_sperm",
    "duplicate",
    "debris_or_false_positive",
    "unclear",
]

MASK_LABEL_OPTIONS = [
    "mask_good",
    "tail_mask_partial",
    "head_mask_wrong",
    "wrong_tail_assigned",
    "extra_tail_assigned",
    "mask_unclear",
]

KNOWN_ARTIFACT_DIRS = [
    "candidate_review",
    "sperm_crops",
    "highlighted_sperm_crops",
    "tail_skeletons",
    "masks",
    "overlays",
    "head_connected_overlays",
    "tail_assignment_debug",
    "intermediates",
]

FIELDNAMES = [
    "selection_rank",
    "selection_reason",
    "crop_label",
    "mask_label",
    "review_notes",
    "reviewer",
    "reviewed_at",
    "review_image_path",
    "highlighted_crop_path",
    "raw_crop_path",
    "skeleton_overlay_path",
    "crop_mask_path",
    "crop_id",
    "relative_image",
    "source_image_path",
    "current_status",
    "score",
    "crop_quality_score",
    "mask_quality_score",
    "path_confidence_score",
    "candidate_risk_flags",
    "rejection_reason",
    "ambiguity_reason",
    "tail_id",
    "head_label_id",
    "tail_assignment_mode",
    "tail_has_overlap",
    "tail_overlap_reasons",
    "tail_endpoint_count",
    "tail_branchpoint_count",
    "assigned_tail_pixels",
    "assigned_skeleton_pixels",
    "shared_tail_pixels",
    "touching_head_count",
    "touching_head_ids",
    "foreign_head_pixels",
    "foreign_tail_pixels",
    "foreign_content_score",
    "foreign_content_by_crop_area",
    "head_anchor_xy",
    "head_anchor_distance_px",
    "path_v2",
    "path_mode",
    "path_endpoint_xy",
    "path_length_px",
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
    "endpoint_to_source_bbox_edge_px",
    "endpoint_to_crop_edge_px",
    "crop_boundary_risk",
    "crop_adaptive_padding_px",
    "tail_cutoff_stage",
    "head_reuse_group_id",
    "head_reuse_candidate_count",
    "head_reuse_accepted_count",
    "tail_reuse_group_id",
    "tail_reuse_candidate_count",
    "tail_reuse_accepted_count",
    "shared_path_fraction",
    "acceptance_profile",
    "source_bbox_touches_image_border",
    "crop_bbox_touches_image_border",
    "duplicate_group_id",
    "duplicate_of_crop_id",
    "bbox_xyxy",
    "source_bbox_xyxy",
    "score_details_json",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create manual review CSVs from SCP crop candidates."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/updated_outputs"),
        help="SCP output root. Relative paths resolve inside Algorithmic_Pixel_Mask_For_Tails/.",
    )
    parser.add_argument(
        "--accepted-boundary-count",
        type=int,
        default=30,
        help="Lowest-scoring accepted candidates to include in the targeted queue.",
    )
    parser.add_argument(
        "--rejected-boundary-count",
        type=int,
        default=20,
        help="Highest-scoring rejected candidates to include in the targeted queue.",
    )
    return parser.parse_args()


def resolve_output_root(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (SCP_ROOT / path).resolve()


def json_dumps(value: Any) -> str:
    if value in (None, ""):
        return ""
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def resolve_artifact_path(stored_path: str | None, output_root: Path) -> str:
    if not stored_path:
        return ""

    path = Path(stored_path)
    for index, part in enumerate(path.parts):
        if part in KNOWN_ARTIFACT_DIRS:
            local_path = output_root.joinpath(*path.parts[index:])
            if local_path.exists():
                return str(local_path.resolve())
            break

    if path.exists():
        return str(path.resolve())

    for index, part in enumerate(path.parts):
        if part in KNOWN_ARTIFACT_DIRS:
            local_path = output_root.joinpath(*path.parts[index:])
            return str(local_path.resolve())

    return str(path)


def first_existing_path(paths: list[str]) -> str:
    for path in paths:
        if path and Path(path).exists():
            return path
    return next((path for path in paths if path), "")


def candidate_to_row(candidate: dict[str, Any], image_json: dict[str, Any], output_root: Path) -> dict[str, str]:
    highlighted_crop_path = resolve_artifact_path(candidate.get("highlighted_crop"), output_root)
    raw_crop_path = resolve_artifact_path(candidate.get("crop"), output_root)
    skeleton_overlay_path = resolve_artifact_path(candidate.get("skeleton_overlay"), output_root)
    crop_mask_path = resolve_artifact_path(candidate.get("crop_mask"), output_root)
    review_image_path = first_existing_path([highlighted_crop_path, raw_crop_path, skeleton_overlay_path])

    row = {
        "selection_rank": "",
        "selection_reason": "",
        "crop_label": "",
        "mask_label": "",
        "review_notes": "",
        "reviewer": "",
        "reviewed_at": "",
        "review_image_path": review_image_path,
        "highlighted_crop_path": highlighted_crop_path,
        "raw_crop_path": raw_crop_path,
        "skeleton_overlay_path": skeleton_overlay_path,
        "crop_mask_path": crop_mask_path,
        "crop_id": str(candidate.get("crop_id", "")),
        "relative_image": str(image_json.get("relative_image", "")),
        "source_image_path": str(image_json.get("image", "")),
        "current_status": str(candidate.get("candidate_status", "")),
        "score": str(candidate.get("score", "")),
        "crop_quality_score": str(candidate.get("crop_quality_score", "")),
        "mask_quality_score": str(candidate.get("mask_quality_score", "")),
        "path_confidence_score": str(candidate.get("path_confidence_score", "")),
        "candidate_risk_flags": ";".join(candidate.get("candidate_risk_flags") or []),
        "rejection_reason": str(candidate.get("rejection_reason") or ""),
        "ambiguity_reason": str(candidate.get("ambiguity_reason") or ""),
        "tail_id": str(candidate.get("tail_id", "")),
        "head_label_id": str(candidate.get("head_label_id", "")),
        "tail_assignment_mode": str(candidate.get("tail_assignment_mode", "")),
        "tail_has_overlap": str(candidate.get("tail_has_overlap", "")),
        "tail_overlap_reasons": ";".join(candidate.get("tail_overlap_reasons") or []),
        "tail_endpoint_count": str(candidate.get("tail_endpoint_count", "")),
        "tail_branchpoint_count": str(candidate.get("tail_branchpoint_count", "")),
        "assigned_tail_pixels": str(candidate.get("assigned_tail_pixels", "")),
        "assigned_skeleton_pixels": str(candidate.get("assigned_skeleton_pixels", "")),
        "shared_tail_pixels": str(candidate.get("shared_tail_pixels", "")),
        "touching_head_count": str(candidate.get("touching_head_count", "")),
        "touching_head_ids": json_dumps(candidate.get("touching_head_ids")),
        "foreign_head_pixels": str(candidate.get("foreign_head_pixels", "")),
        "foreign_tail_pixels": str(candidate.get("foreign_tail_pixels", "")),
        "foreign_content_score": str(candidate.get("foreign_content_score", "")),
        "foreign_content_by_crop_area": str(candidate.get("foreign_content_by_crop_area", "")),
        "head_anchor_xy": json_dumps(candidate.get("head_anchor_xy")),
        "head_anchor_distance_px": str(candidate.get("head_anchor_distance_px", "")),
        "path_v2": str(candidate.get("path_v2", "")),
        "path_mode": str(candidate.get("path_mode", "")),
        "path_endpoint_xy": json_dumps(candidate.get("path_endpoint_xy")),
        "path_length_px": str(candidate.get("path_length_px", "")),
        "path_score_geometry": str(candidate.get("path_score_geometry", "")),
        "path_score_color": str(candidate.get("path_score_color", "")),
        "path_score_hybrid": str(candidate.get("path_score_hybrid", "")),
        "path_score_margin": str(candidate.get("path_score_margin", "")),
        "path_score_margin_geometry": str(candidate.get("path_score_margin_geometry", "")),
        "path_score_margin_color": str(candidate.get("path_score_margin_color", "")),
        "path_score_margin_hybrid": str(candidate.get("path_score_margin_hybrid", "")),
        "path_rank_geometry": str(candidate.get("path_rank_geometry", "")),
        "path_rank_color": str(candidate.get("path_rank_color", "")),
        "path_rank_hybrid": str(candidate.get("path_rank_hybrid", "")),
        "path_low_margin": str(candidate.get("path_low_margin", "")),
        "competing_path_count": str(candidate.get("competing_path_count", "")),
        "path_color_continuity_score": str(candidate.get("path_color_continuity_score", "")),
        "path_color_delta_mean": str(candidate.get("path_color_delta_mean", "")),
        "path_color_delta_p75": str(candidate.get("path_color_delta_p75", "")),
        "path_color_reference_points": str(candidate.get("path_color_reference_points", "")),
        "path_color_normalization": str(candidate.get("path_color_normalization", "")),
        "color_score_raw": str(candidate.get("color_score_raw", "")),
        "color_score_normalized": str(candidate.get("color_score_normalized", "")),
        "color_delta_p75_raw": str(candidate.get("color_delta_p75_raw", "")),
        "color_delta_p75_normalized": str(candidate.get("color_delta_p75_normalized", "")),
        "path_curvature_total": str(candidate.get("path_curvature_total", "")),
        "path_branchpoint_count": str(candidate.get("path_branchpoint_count", "")),
        "path_width_spike_ratio": str(candidate.get("path_width_spike_ratio", "")),
        "path_trimmed_branch_pixels": str(candidate.get("path_trimmed_branch_pixels", "")),
        "path_trimmed_branch_fraction": str(candidate.get("path_trimmed_branch_fraction", "")),
        "tail_pixels_inside_head_collar": str(candidate.get("tail_pixels_inside_head_collar", "")),
        "tail_pixels_outside_head_collar": str(candidate.get("tail_pixels_outside_head_collar", "")),
        "head_exit_count": str(candidate.get("head_exit_count", "")),
        "head_exit_path_length_px": str(candidate.get("head_exit_path_length_px", "")),
        "head_exit_xy": json_dumps(candidate.get("head_exit_xy")),
        "valid_tail_continuation": str(candidate.get("valid_tail_continuation", "")),
        "path_rejection_reason": str(candidate.get("path_rejection_reason") or ""),
        "path_ambiguity_reason": str(candidate.get("path_ambiguity_reason") or ""),
        "path_conflict_fraction": str(candidate.get("path_conflict_fraction", "")),
        "endpoint_to_source_bbox_edge_px": str(candidate.get("endpoint_to_source_bbox_edge_px", "")),
        "endpoint_to_crop_edge_px": str(candidate.get("endpoint_to_crop_edge_px", "")),
        "crop_boundary_risk": str(candidate.get("crop_boundary_risk", "")),
        "crop_adaptive_padding_px": str(candidate.get("crop_adaptive_padding_px", "")),
        "tail_cutoff_stage": str(candidate.get("tail_cutoff_stage", "")),
        "head_reuse_group_id": str(candidate.get("head_reuse_group_id", "")),
        "head_reuse_candidate_count": str(candidate.get("head_reuse_candidate_count", "")),
        "head_reuse_accepted_count": str(candidate.get("head_reuse_accepted_count", "")),
        "tail_reuse_group_id": str(candidate.get("tail_reuse_group_id", "")),
        "tail_reuse_candidate_count": str(candidate.get("tail_reuse_candidate_count", "")),
        "tail_reuse_accepted_count": str(candidate.get("tail_reuse_accepted_count", "")),
        "shared_path_fraction": str(candidate.get("shared_path_fraction", "")),
        "acceptance_profile": str(candidate.get("acceptance_profile", "")),
        "source_bbox_touches_image_border": str(candidate.get("source_bbox_touches_image_border", "")),
        "crop_bbox_touches_image_border": str(candidate.get("crop_bbox_touches_image_border", "")),
        "duplicate_group_id": str(candidate.get("duplicate_group_id", "")),
        "duplicate_of_crop_id": str(candidate.get("duplicate_of_crop_id") or ""),
        "bbox_xyxy": json_dumps(candidate.get("bbox_xyxy")),
        "source_bbox_xyxy": json_dumps(candidate.get("source_bbox_xyxy")),
        "score_details_json": json_dumps(candidate.get("score_details")),
    }
    return row


def load_candidate_rows(output_root: Path) -> list[dict[str, str]]:
    json_root = output_root / "json"
    if not json_root.exists():
        raise FileNotFoundError(f"Could not find SCP JSON folder: {json_root}")

    rows: list[dict[str, str]] = []
    for json_path in sorted(json_root.glob("*.json")):
        image_json = json.loads(json_path.read_text(encoding="utf-8"))
        candidates = (
            image_json.get("head_connected_postprocessing", {})
            .get("crop_candidates_detail", [])
        )
        for candidate in candidates:
            rows.append(candidate_to_row(candidate, image_json, output_root))
    return rows


def score_as_float(row: dict[str, str]) -> float:
    try:
        return float(row["score"])
    except ValueError:
        return 0.0


def make_review_queue(
    all_rows: list[dict[str, str]],
    accepted_boundary_count: int,
    rejected_boundary_count: int,
) -> list[dict[str, str]]:
    queue: list[dict[str, str]] = []
    seen_crop_ids: set[str] = set()

    def add_rows(rows: list[dict[str, str]], reason: str) -> None:
        for row in rows:
            crop_id = row["crop_id"]
            if crop_id in seen_crop_ids:
                continue
            queued = dict(row)
            queued["selection_reason"] = reason
            queue.append(queued)
            seen_crop_ids.add(crop_id)

    ambiguous = [
        row for row in all_rows if row["current_status"] == "ambiguous"
    ]
    ambiguous.sort(key=lambda row: (row["ambiguity_reason"], -score_as_float(row), row["crop_id"]))
    add_rows(ambiguous, "ambiguous_current_status")

    accepted = [
        row for row in all_rows if row["current_status"] == "accepted"
    ]
    accepted.sort(key=lambda row: (score_as_float(row), row["crop_id"]))
    add_rows(accepted[: max(0, accepted_boundary_count)], "low_scoring_accepted_boundary")

    rejected = [
        row for row in all_rows if row["current_status"] == "rejected"
    ]
    rejected.sort(key=lambda row: (-score_as_float(row), row["crop_id"]))
    add_rows(rejected[: max(0, rejected_boundary_count)], "high_scoring_rejected_boundary")

    for rank, row in enumerate(queue, start=1):
        row["selection_rank"] = str(rank)
    return queue


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_label_key(path: Path) -> None:
    crop_options = "\n".join(f"- `{label}`" for label in CROP_LABEL_OPTIONS)
    mask_options = "\n".join(f"- `{label}`" for label in MASK_LABEL_OPTIONS)
    text = f"""# SCP Candidate Review Label Key

Review the cropped image in its entirety first, then use the overlay/mask as a
diagnostic layer.

## Human Columns

- `crop_label`: Is this a good one-sperm crop?
- `mask_label`: Is the SCP overlay/mask assignment correct?
- `review_notes`: Optional. Use this for uncertainty, crop/mask disagreement,
  or recurring failure patterns.
- `reviewer`: Optional reviewer name or initials.
- `reviewed_at`: Optional review date.

## Crop Label Options

{crop_options}

## Mask Label Options

{mask_options}

Notes are not required for obvious cases. They are most useful when the crop and
mask disagree, or when the candidate is difficult to classify.
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_root = resolve_output_root(args.output_root)
    review_root = output_root / "candidate_review"

    all_rows = load_candidate_rows(output_root)
    review_queue = make_review_queue(
        all_rows=all_rows,
        accepted_boundary_count=args.accepted_boundary_count,
        rejected_boundary_count=args.rejected_boundary_count,
    )

    all_csv = review_root / "candidate_review_all_candidates.csv"
    queue_csv = review_root / "candidate_review_queue.csv"
    label_key = review_root / "candidate_review_label_key.md"

    write_csv(all_csv, all_rows)
    write_csv(queue_csv, review_queue)
    write_label_key(label_key)

    summary = {
        "output_root": str(output_root),
        "all_candidates_csv": str(all_csv),
        "review_queue_csv": str(queue_csv),
        "label_key": str(label_key),
        "total_candidates": len(all_rows),
        "review_queue_candidates": len(review_queue),
        "accepted_boundary_count": args.accepted_boundary_count,
        "rejected_boundary_count": args.rejected_boundary_count,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
