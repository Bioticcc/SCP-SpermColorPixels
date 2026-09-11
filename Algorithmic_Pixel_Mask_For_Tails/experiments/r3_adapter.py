"""Adapt frozen R0/R1/R2 evidence into R3 logical-mask review assets."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from experiments.r1_adapter import restore_component_labels
from mask_reconstruction import CrossingRegion, Track, reconstruct_tracks
from mask_reconstruction.head_ownership import compose_head_and_tail


def _path(value: Any) -> Path:
    path = Path(str(value))
    if not path.is_file():
        raise ValueError(f"Frozen asset is missing: {path}")
    return path


def _write(path: Path, image: np.ndarray, root: Path) -> str:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite R3 export: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path)
    return path.relative_to(root).as_posix()


def _hypotheses(payload: dict) -> dict[str, dict]:
    return {str(row["id"]): row for row in payload.get("hypotheses", []) if isinstance(row, dict) and row.get("id") is not None}


def _solver_uncertainty(assignment: dict, head: str) -> dict:
    for group in assignment.get("groups", []):
        if head in {str(value) for value in group.get("head_ids", [])}:
            local = group.get("assignment", {})
            alternative = local.get("head_alternatives", {}).get(head, {})
            return {"solver_status": local.get("status"), "per_head_score_margin_uncalibrated": alternative.get("score_margin")}
    return {"solver_status": assignment.get("status"), "per_head_score_margin_uncalibrated": None}


def _component_by_tail(record: dict) -> dict[str, dict]:
    return {str(row.get("tail_id")): row for row in record.get("components", [])}


def _crossings(graph: dict, evidence: np.ndarray, tracks: list[Track], node_ids: dict[str, tuple[str, ...]], directory: Path, root: Path) -> tuple[list[CrossingRegion], list[dict]]:
    regions, metadata = [], []
    segments = graph.get("segments", [])
    for node in graph.get("nodes", []):
        if node.get("kind") != "junction":
            continue
        node_id = str(node.get("id")); involved = [track for track in tracks if node_id in node_ids.get(track.id, ())]
        pixels = node.get("pixels_xy") or ([node.get("xy")] if node.get("xy") else [])
        radii = [float(radius) for segment in segments if node_id in {str(segment.get("start")), str(segment.get("end"))}
                 for radius in segment.get("radii", []) if isinstance(radius, (int, float)) and math.isfinite(radius)]
        radius = max(1, int(math.ceil(float(np.median(radii))))) if radii else 1
        seed = np.zeros_like(evidence, np.uint8)
        for point in pixels:
            if isinstance(point, (list, tuple)) and len(point) >= 2:
                x, y = int(point[0]), int(point[1])
                if 0 <= x < evidence.shape[1] and 0 <= y < evidence.shape[0]: seed[y, x] = 1
        mask = cv2.dilate(seed, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1,) * 2)).astype(bool) & evidence
        track_ids = tuple(track.id for track in involved)
        region_file = _write(directory / f"crossing_{node_id}_mask.png", (mask * 255).astype(np.uint8), root)
        metadata.append({"id": node_id, "radius_px": radius, "track_ids": list(track_ids), "pixels_xy": pixels, "mask_file": region_file})
        if track_ids and mask.any(): regions.append(CrossingRegion(node_id, mask, track_ids))
    return regions, metadata


def _bbox(mask: np.ndarray, padding: int) -> tuple[int, int, int, int]:
    yy, xx = np.nonzero(mask)
    if not len(xx): raise ValueError("R3 instance has no selected tail/head pixels")
    return max(0, int(xx.min()) - padding), max(0, int(yy.min()) - padding), min(mask.shape[1], int(xx.max()) + 1 + padding), min(mask.shape[0], int(yy.max()) + 1 + padding)


def reconstruct_image(directory: Path, source_record: dict, r1_record: dict, r2_payload: dict, config: dict) -> dict:
    """Reconstruct selected non-null R2 tracks without changing selection or evidence."""
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    rgb = np.asarray(Image.open(_path(source_record["image"])).convert("RGB"))
    outputs = source_record.get("outputs", {})
    tail_mask = np.asarray(Image.open(_path(outputs["tail_mask"])).convert("L")) > 0
    head_mask = np.asarray(Image.open(_path(outputs["head_mask"])).convert("L")) > 0
    if rgb.shape[:2] != tail_mask.shape or tail_mask.shape != head_mask.shape:
        raise ValueError("Frozen RGB, tail mask, and head mask dimensions disagree")
    identity = source_record.get("relative_image")
    if not identity or r1_record.get("relative_image") != identity or r2_payload.get("relative_image") != identity:
        raise ValueError("Frozen R0, R1, and R2 relative-image identities must match")
    if any(value is not None and int(value) != expected for value, expected in ((r1_record.get("width"), rgb.shape[1]), (r1_record.get("height"), rgb.shape[0]), (r2_payload.get("width"), rgb.shape[1]), (r2_payload.get("height"), rgb.shape[0]))):
        raise ValueError("Frozen R0, R1, and R2 dimensions must match")
    tail_labels = restore_component_labels(tail_mask, source_record["tails"])
    head_count, head_labels, head_stats, _ = cv2.connectedComponentsWithStats(head_mask.astype(np.uint8), connectivity=8)
    if head_count - 1 != len(source_record["heads"]):
        raise ValueError("Head component inventory differs from frozen production metadata")
    by_id, assignment = _hypotheses(r2_payload), r2_payload.get("assignment", {})
    selected = assignment.get("selected_by_head", {})
    chosen = []
    for head, hypothesis_id in selected.items():
        item = by_id.get(str(hypothesis_id))
        if item is None or str(item.get("head_id")) != str(head):
            raise ValueError(f"Selected R2 choice is missing or belongs to another head: {head}")
        chosen.append((str(head), item))
    nonnull = [(head, item) for head, item in chosen if item.get("termination") != "null"]
    r1_components = _component_by_tail(r1_record)
    component_tracks: dict[str, list[Track]] = {}
    track_data: dict[str, tuple[str, dict, dict]] = {}; node_ids: dict[str, tuple[str, ...]] = {}
    for head, item in nonnull:
        metadata = item.get("metadata", {})
        tail_id = str(metadata.get("tail_id"))
        component = r1_components.get(tail_id)
        if component is None: raise ValueError(f"Selected R2 hypothesis lacks R1 component {tail_id}")
        roi = component.get("roi_xyxy")
        if not isinstance(roi, (list, tuple)) or len(roi) != 4 or not (0 <= int(roi[0]) < int(roi[2]) <= rgb.shape[1] and 0 <= int(roi[1]) < int(roi[3]) <= rgb.shape[0]):
            raise ValueError(f"R1 component ROI is outside source image: {tail_id}")
        if metadata.get("roi_xyxy") != list(roi):
            raise ValueError(f"R2 hypothesis ROI differs from frozen R1 component: {item['id']}")
        points = metadata.get("pixels_xy_unordered", []) if metadata.get("kind") == "baseline_control" else metadata.get("points_xy_ordered", [])
        track = Track(str(item["id"]), head, points, ordered=metadata.get("kind") != "baseline_control")
        node_ids[track.id] = tuple(str(value) for value in metadata.get("r1_route", {}).get("node_ids", ()))
        component_tracks.setdefault(tail_id, []).append(track); track_data[track.id] = (head, item, component)

    results, component_reports = {}, []
    reconstruction_config = config.get("reconstruction", {})
    for tail_id, tracks in component_tracks.items():
        component = r1_components[tail_id]; x0, y0, x1, y1 = (int(value) for value in component["roi_xyxy"])
        evidence = tail_labels[y0:y1, x0:x1] == int(tail_id)
        component_dir = directory / "components" / f"tail_{tail_id}"
        evidence_file = _write(component_dir / "evidence_mask.png", (evidence * 255).astype(np.uint8), directory)
        crossings, crossing_metadata = _crossings(component.get("graph", {}), evidence, tracks, node_ids, component_dir, directory)
        core = reconstruct_tracks(evidence, tracks, crossings, reconstruction_config)
        results.update({track_id: (core, x0, y0) for track_id in core["masks"]})
        component_reports.append({"tail_id": tail_id, "source_roi_xyxy": [x0, y0, x1, y1], "evidence_mask_file": evidence_file, "crossing_regions": crossing_metadata,
                                  "core_diagnostics": core["diagnostics"], "evidence_pixels": int(evidence.sum())})

    selected_heads = {int(head): str(item['id']) for head, item in nonnull}
    owned_head_pixels = np.isin(head_labels, list(selected_heads))
    pinned_counts = np.zeros(tail_mask.shape, np.uint16)
    conflicts = []
    for track_id, (core, x0, y0) in results.items():
        line = core['centerlines'][track_id]
        pinned_counts[y0:y0+line.shape[0], x0:x0+line.shape[1]] += line
        own_head = int(track_data[track_id][0])
        local_heads = head_labels[y0:y0+line.shape[0], x0:x0+line.shape[1]]
        for foreign_head in sorted((set(np.unique(local_heads[line])) & set(selected_heads)) - {own_head}):
            yy, xx = np.nonzero(line & (local_heads == foreign_head))
            conflicts.append({'reason': 'head_tail_evidence_conflict',
                'head_instance_id': selected_heads[foreign_head], 'tail_instance_id': track_id,
                'source_pixels_xy': np.column_stack((xx+x0, yy+y0)).tolist(),
                'default_owner': track_id, 'alternative_owner': selected_heads[foreign_head]})
    instances = []
    for ordinal, (track_id, (core, x0, y0)) in enumerate(sorted(results.items())):
        head, item, component = track_data[track_id]; head_label = int(head)
        if not 0 < head_label < head_count: raise ValueError(f"Selected head label is outside frozen head inventory: {head}")
        local_tail, local_line = core["masks"][track_id], core["centerlines"][track_id]
        tail_source = np.zeros_like(tail_mask); line_source = np.zeros_like(tail_mask)
        tail_source[y0:y0 + local_tail.shape[0], x0:x0 + local_tail.shape[1]] = local_tail
        line_source[y0:y0 + local_line.shape[0], x0:x0 + local_line.shape[1]] = local_line
        composition = compose_head_and_tail(tail_source, line_source, head_labels == head_label,
                                            owned_head_pixels & (head_labels != head_label), pinned_counts > line_source)
        tail_source, head_source = composition['tail'], composition['head']
        logical = tail_source | head_source
        bx0, by0, bx1, by1 = _bbox(logical | composition['head_evidence'], 4)
        crop = rgb[by0:by1, bx0:bx1].copy()
        if not np.array_equal(crop, rgb[by0:by1, bx0:bx1]): raise ValueError("Original crop preservation check failed")
        instance_dir = directory / "instances" / f"{ordinal:03d}_{track_id.replace(':', '_')}"
        files = {"original_crop": _write(instance_dir / "original_crop.png", crop, directory),
                 "tail_mask": _write(instance_dir / "tail_mask.png", (tail_source[by0:by1, bx0:bx1] * 255).astype(np.uint8), directory),
                 "head_mask": _write(instance_dir / "head_mask.png", (head_source[by0:by1, bx0:bx1] * 255).astype(np.uint8), directory),
                 "instance_mask": _write(instance_dir / "instance_mask.png", (logical[by0:by1, bx0:bx1] * 255).astype(np.uint8), directory),
                 "centerline": _write(instance_dir / "centerline.png", (line_source[by0:by1, bx0:bx1] * 255).astype(np.uint8), directory)}
        logical_crop = logical[by0:by1, bx0:bx1]
        highlighted = crop.copy()
        highlighted[logical_crop] = np.rint(.55 * highlighted[logical_crop] + .45 * np.array([0, 185, 232])).astype(np.uint8)
        files["highlighted_crop"] = _write(instance_dir / "highlighted_crop.png", highlighted, directory)
        for key, field in [('head_evidence_mask', 'head_evidence'), ('ownership_uncertainty_mask', 'uncertainty'),
                           ('head_priority_instance_mask', 'head_priority_instance')]:
            files[key] = _write(instance_dir/f'{key}.png', (composition[field][by0:by1, bx0:bx1]*255).astype(np.uint8), directory)
        radii = core["radii"][track_id]; supported = int(local_line.sum())
        metadata = item.get("metadata", {}); route = metadata.get("r1_route", {})
        r1_search = component.get("searches", {}).get(head, {})
        diagnostics = core["diagnostics"]
        head_crop, tail_crop = head_source[by0:by1, bx0:bx1], tail_source[by0:by1, bx0:bx1]
        distance = cv2.distanceTransform((~head_crop).astype(np.uint8), cv2.DIST_L2, 5)
        gap = float(distance[tail_crop].min()) if tail_crop.any() and head_crop.any() else None
        adjacent = bool((cv2.dilate(head_crop.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & tail_crop).any())
        union_components = int(cv2.connectedComponents(logical_crop.astype(np.uint8), connectivity=8)[0] - 1)
        uncertainty = {"r2_termination": item.get("termination"), **_solver_uncertainty(assignment, head),
                       'head_tail_evidence_conflicts': [entry for entry in conflicts if track_id in (entry['head_instance_id'], entry['tail_instance_id'])],
                       'logical_ownership_policy': 'Default: supported centerline priority. Alternative: detected head priority. This policy does not establish biological occlusion or identity.',
                       'detected_head_measurement_source': 'head_evidence_mask; logical head_mask may exclude explicitly uncertain boundary pixels',
                       "upstream_r1_truncated": bool(r1_search.get("diagnostics", {}).get("truncated")),
                       "unsupported_centerline_pixels": diagnostics["unsupported_centerline_pixels"][track_id],
                       "out_of_bounds_centerline_samples": diagnostics["out_of_bounds_centerline_samples"][track_id],
                       "width_unknown_samples": diagnostics["width_unknown_samples"][track_id],
                       "width_interpolated_samples": diagnostics["width_interpolated_samples"][track_id],
                       "border_truncated_width_samples": diagnostics["border_truncated_width_samples"][track_id],
                       "crossing_ids": [row["id"] for row in next(report for report in component_reports if report["tail_id"] == str(metadata.get("tail_id")))["crossing_regions"] if track_id in row["track_ids"]]}
        instances.append({"id": track_id, "head_id": head, "tail_id": str(metadata.get("tail_id")), "termination": item.get("termination"), "roi_xyxy": [bx0, by0, bx1, by1], "files": files,
                          "measurements": {"supported_centerline_pixels": supported, "unsupported_centerline_pixels": diagnostics["unsupported_centerline_pixels"][track_id],
                                           "radius_stats_px": {"min": min(radii) if radii else None, "median": float(np.median(radii)) if radii else None, "max": max(radii) if radii else None},
                                           "radius_points_xy_roi_local": core.get("radius_points_xy", {}).get(track_id, []),
                                           "diameter_stats_px_boundary_convention": "twice center-to-boundary radius; distance-transform radius uses a half-pixel boundary convention",
                                           "tail_pixels": int(tail_source.sum()), "head_pixels": int(head_source.sum()), "unselected_evidence_in_crop": int((tail_mask[by0:by1, bx0:bx1] & ~tail_source[by0:by1, bx0:bx1]).sum()),
                                           'head_evidence_pixels': int(composition['head_evidence'].sum()),
                                           'head_pixels_deferred_to_foreign_centerline': composition['head_pixels_deferred'],
                                           'tail_growth_pixels_removed_at_foreign_heads': composition['growth_pixels_removed'],
                                           'pinned_tail_pixels_in_foreign_head': composition['pinned_tail_pixels_in_foreign_head'],
                                           "head_tail_gap_diagnostic_px": gap, "head_tail_adjacent_8": adjacent, "logical_union_component_count": union_components}, "uncertainty": uncertainty})
    null_choices = [{"head_id": head, "termination": item.get("termination"), "selected_id": item["id"]} for head, item in chosen if item.get("termination") == "null"]
    return {"schema_version": "scp.r3.reconstruction.v2", "relative_image": source_record.get("relative_image"), "width": int(rgb.shape[1]), "height": int(rgb.shape[0]), "instances": instances, "null_choices": null_choices,
            'head_tail_evidence_conflicts': conflicts,
            "components": component_reports, "source_comparisons": {"production_statuses_changed": False, "scope": "Selected R2 paths are unchanged; logical masks are evidence-only exports."}}
