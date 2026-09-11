"""Pure R3 per-instance export invariants used by the release audit."""
from __future__ import annotations
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

_FILES = ("original_crop", "tail_mask", "head_mask", "instance_mask", "centerline", "highlighted_crop")
_V2 = _FILES + ("head_evidence_mask", "ownership_uncertainty_mask", "head_priority_instance_mask")

def _read(path: Path) -> np.ndarray:
    if not path.is_file(): raise ValueError(f"missing instance export: {path.name}")
    return np.asarray(Image.open(path))

def _binary(value: np.ndarray, name: str, shape: tuple[int,int]) -> np.ndarray:
    if value.shape != shape or value.ndim != 2 or not np.isin(value, (0,255)).all(): raise ValueError(f"invalid binary {name}")
    return value.astype(bool)

def check_instance(directory: Path, instance: dict, source_rgb: np.ndarray, head_labels: np.ndarray,
                   tail_labels: np.ndarray, chosen_hypothesis: dict, r1_component: dict,
                   selected_centerline_counts=None, selected_head_ids=None) -> dict:
    """Raise on an altered R3 instance export and return compact pixel counts."""
    if source_rgb.ndim != 3 or source_rgb.shape[2] != 3 or head_labels.shape != source_rgb.shape[:2] or tail_labels.shape != head_labels.shape: raise ValueError("source dimensions disagree")
    files = instance.get("files", {}); v2 = selected_centerline_counts is not None or selected_head_ids is not None
    if v2 and (selected_centerline_counts is None or selected_head_ids is None): raise ValueError("v2 requires selected centerlines and heads")
    required = set(_V2 if v2 else _FILES)
    if set(files) != required: raise ValueError("instance file manifest differs")
    try: x0,y0,x1,y1 = instance["roi_xyxy"]
    except (KeyError, ValueError): raise ValueError("missing ROI")
    if any(type(v) is not int for v in (x0,y0,x1,y1)) or not (0 <= x0 < x1 <= source_rgb.shape[1] and 0 <= y0 < y1 <= source_rgb.shape[0]): raise ValueError("invalid ROI")
    for key in ("id", "head_id", "termination"):
        if key not in chosen_hypothesis or str(instance.get(key)) != str(chosen_hypothesis[key]):
            raise ValueError("selected hypothesis metadata mismatch")
    tail_identity = str(chosen_hypothesis.get("metadata", {}).get("tail_id"))
    if str(instance.get("tail_id")) != tail_identity or str(r1_component.get("tail_id")) != tail_identity:
        raise ValueError("selected tail component mismatch")
    crop = _read(directory/files["original_crop"])
    expected = source_rgb[y0:y1,x0:x1]
    if crop.shape != expected.shape or not np.array_equal(crop, expected): raise ValueError("original crop differs from source RGB")
    shape = expected.shape[:2]
    tail, head, logical, line = (_binary(_read(directory/files[key]), key, shape) for key in ("tail_mask","head_mask","instance_mask","centerline"))
    highlight = _read(directory/files["highlighted_crop"])
    if highlight.shape != expected.shape: raise ValueError("highlighted crop dimensions differ")
    head_id, tail_id = int(instance["head_id"]), int(instance["tail_id"])
    raw_head = head_labels[y0:y1,x0:x1] == head_id
    if not v2 and not np.array_equal(head, raw_head): raise ValueError("head mask is not complete production label")
    if not v2 and int(head.sum()) != int((head_labels == head_id).sum()): raise ValueError("head label was truncated by the export ROI")
    if np.any(tail & (tail_labels[y0:y1,x0:x1] != tail_id)) or not np.array_equal(logical, head | tail): raise ValueError("tail/logical ownership mismatch")
    roi = r1_component["roi_xyxy"]; rx0,ry0,_,_ = map(int,roi); evidence = tail_labels[ry0:roi[3],rx0:roi[2]] == tail_id
    meta = chosen_hypothesis.get("metadata", {}); points = meta.get("pixels_xy_unordered", []) if meta.get("kind") == "baseline_control" else meta.get("points_xy_ordered", [])
    local = np.zeros_like(evidence, np.uint8)
    if meta.get("kind") == "baseline_control":
        for x,y in points:
            if 0 <= int(x) < local.shape[1] and 0 <= int(y) < local.shape[0]: local[int(y),int(x)] = 1
    elif points:
        cv2.polylines(local, [np.asarray(points,np.int32)], False, 1, 1, lineType=cv2.LINE_8)
    expected_line = np.zeros_like(head_labels,bool); expected_line[ry0:roi[3],rx0:roi[2]] = local.astype(bool) & evidence
    if not np.array_equal(line, expected_line[y0:y1,x0:x1]): raise ValueError("centerline differs from selected R2 path")
    if int(line.sum()) != int(expected_line.sum()): raise ValueError("selected centerline was truncated by the export ROI")
    if np.any(line & ~tail): raise ValueError("supported centerline absent from tail mask")
    result = {"tail_pixels": int(tail.sum()), "head_pixels": int(head.sum()), "centerline_pixels": int(line.sum())}
    if v2:
        extra = {key: _binary(_read(directory/files[key]), key, shape) for key in _V2[6:]}
        if not np.array_equal(extra["head_evidence_mask"], raw_head): raise ValueError("head evidence differs from original label")
        if int(extra["head_evidence_mask"].sum()) != int((head_labels == head_id).sum()): raise ValueError("head evidence was truncated")
        all_counts = np.asarray(selected_centerline_counts)
        if all_counts.shape != head_labels.shape or not np.issubdtype(all_counts.dtype, np.integer) or np.any(all_counts < expected_line):
            raise ValueError("Invalid selected centerline counts")
        selected_ids = {int(value) for value in selected_head_ids}
        if head_id not in selected_ids or 0 in selected_ids:
            raise ValueError("Selected head inventory differs")
        counts = all_counts[y0:y1,x0:x1]
        foreign_pins = counts > line
        foreign_heads = np.isin(head_labels[y0:y1,x0:x1], list(selected_ids - {head_id}))
        if not np.array_equal(head, raw_head & ~foreign_pins): raise ValueError("logical head ownership mismatch")
        if np.any(tail & foreign_heads & ~line): raise ValueError("tail growth occupies foreign selected head")
        expected_uncertainty = (raw_head & foreign_pins) | (line & foreign_heads)
        if not np.array_equal(extra["ownership_uncertainty_mask"], expected_uncertainty): raise ValueError("uncertainty mask mismatch")
        priority = raw_head | (tail & ~foreign_heads)
        if not np.array_equal(extra["head_priority_instance_mask"], priority): raise ValueError("head priority mask mismatch")
        result.update({"head_evidence_pixels": int(raw_head.sum()), "pinned_tail_pixels_in_foreign_head": int((line & foreign_heads).sum())})
    return result
