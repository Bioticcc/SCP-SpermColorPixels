"""Translate frozen R1 routes into inspectable R2 assignment hypotheses.

This module does not alter SCP candidates, masks, or statuses.  Its costs are
versioned engineering preferences for the bounded R2 solver, not probabilities
or biological length priors.
"""
from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np

from global_assignment import Hypothesis
from graph_construction import from_explicit_graph


_COST_DEFAULTS = {"null_cost": 1.5, "partial_cost": 0.85, "coverage_weight": 1.0}
_SCHEMA_VERSION = "r2.candidate-pool.v1"


def _controls(config: Mapping[str, Any]) -> dict[str, float]:
    if not isinstance(config, Mapping):
        raise TypeError("config must be a mapping")
    result = dict(_COST_DEFAULTS)
    for key in _COST_DEFAULTS:
        if key not in config:
            continue
        value = config[key]
        if isinstance(value, bool):
            raise ValueError(f"{key} must be a finite non-negative number")
        try:
            value = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{key} must be a finite non-negative number") from error
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{key} must be a finite non-negative number")
        result[key] = value
    return result


def _graph(graph: Any):
    if hasattr(graph, "segments") and hasattr(graph, "nodes") and hasattr(graph, "anchors"):
        return graph
    if not isinstance(graph, Mapping):
        raise TypeError("graph must be a SegmentGraph or serialized graph mapping")
    shape = tuple(graph.get("shape", (128, 128)))
    return from_explicit_graph(graph, shape=shape, anchors=graph.get("anchors"))


def _length(points_xy: Any) -> float:
    points = np.asarray(points_xy, dtype=float)
    if points.ndim != 2 or points.shape[1:] != (2,) or not np.isfinite(points).all():
        raise ValueError("R1 hypothesis points_xy must be finite Nx2 coordinates")
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum()) if len(points) > 1 else 0.0


def _endpoint_id(graph: Any, tail_id: str, record: Mapping[str, Any]) -> str | None:
    if record["termination"] not in {"endpoint", "boundary"} or not record["node_ids"]:
        return None
    node_id = str(record["node_ids"][-1])
    node = graph.nodes[node_id]
    kind = getattr(node, "kind", "ordinary")
    return f"{tail_id}:{node_id}" if kind in {"endpoint", "boundary", "boundary_port"} else None


def _nonnull_records(searches: Mapping[str, Mapping[str, Any]]) -> list[tuple[str, Mapping[str, Any]]]:
    rows = []
    for head_id, result in sorted(searches.items(), key=lambda item: str(item[0])):
        for record in result.get("hypotheses", ()):
            if record.get("termination") != "null":
                rows.append((str(head_id), record))
    return rows


def _r1_hypothesis(
    graph: Any,
    record: Mapping[str, Any],
    head_id: str,
    tail_id: str,
    roi_xyxy: list[int] | tuple[int, ...] | None,
    controls: Mapping[str, float],
    max_length: float,
    baseline_refs: list[Mapping[str, Any]],
) -> Hypothesis:
    route_id = str(record.get("id", "pending"))
    segment_ids = tuple(str(segment_id) for segment_id in record.get("segment_ids", ()))
    length = _length(record.get("points_xy", ()))
    costs = record.get("costs", {})
    direction = float(costs.get("direction", 0.0))
    curvature = float(costs.get("curvature", 0.0))
    evidence = float(costs.get("evidence", 0.0))
    if not all(math.isfinite(value) for value in (direction, curvature, evidence)):
        raise ValueError("R1 route costs must be finite")
    termination = str(record["termination"])
    coverage_deficit = 0.0 if max_length <= 0.0 else 1.0 - length / max_length
    termination_cost = controls["partial_cost"] if termination == "partial" else 0.0
    mean_transition_cost = (direction + curvature) / max(1, len(segment_ids) - 1)
    total = mean_transition_cost + evidence + controls["coverage_weight"] * coverage_deficit + termination_cost
    return Hypothesis(
        id=f"r2:{tail_id}:{head_id}:{route_id}",
        head_id=head_id,
        cost=float(total),
        exclusive_segments=tuple(f"{tail_id}:{segment_id}" for segment_id in segment_ids),
        endpoint_id=_endpoint_id(graph, tail_id, record),
        termination=termination,
        metadata={
            "kind": "r1_route",
            "tail_id": tail_id,
            "head_id": head_id,
            "roi_xyxy": list(roi_xyxy) if roi_xyxy is not None else None,
            "source_roi_offset_xy": list(roi_xyxy[:2]) if roi_xyxy is not None else None,
            "points_xy_ordered": list(record.get("points_xy", ())),
            "r1_route": dict(record),
            "r1_raw_score": record.get("score"),
            "baseline_comparison_refs": [dict(item) for item in baseline_refs],
            "cost_decomposition": {
                "direction": direction,
                "curvature": curvature,
                "mean_transition_cost": mean_transition_cost,
                "evidence": evidence,
                "path_arclength": length,
                "max_available_path_arclength": max_length,
                "coverage_deficit": coverage_deficit,
                "coverage_cost": controls["coverage_weight"] * coverage_deficit,
                "termination_cost": termination_cost,
                "total": total,
            },
        },
    )


def _pool_for_graph(graph: Any, searches: Mapping[str, Mapping[str, Any]], controls: Mapping[str, float], component_id: str, *, include_null: bool, roi_xyxy: list[int] | tuple[int, ...] | None = None, baseline_refs: list[Mapping[str, Any]] | None = None, max_lengths: Mapping[str, float] | None = None) -> list[Hypothesis]:
    graph = _graph(graph)
    tail_id = str(component_id)
    baseline_refs = [] if baseline_refs is None else baseline_refs
    rows = _nonnull_records(searches)
    local_max_lengths = {
        head_id: max((_length(record.get("points_xy", ())) for candidate_head, record in rows if candidate_head == head_id), default=0.0)
        for head_id in (str(key) for key in searches)
    }
    max_lengths = local_max_lengths if max_lengths is None else max_lengths
    hypotheses = [
        _r1_hypothesis(graph, record, head_id, tail_id, roi_xyxy, controls, max_lengths.get(head_id, 0.0), baseline_refs)
        for head_id, record in rows
    ]
    if include_null:
        hypotheses.extend(
            Hypothesis(f"r2:null:{head_id}", str(head_id), controls["null_cost"], termination="null",
                       metadata={"kind": "null", "head_id": str(head_id), "cost_decomposition": {"null_cost": controls["null_cost"]}})
            for head_id in sorted(searches, key=str)
        )
    return hypotheses


def build_graph_pool(graph: Any, searches: Mapping[str, Mapping[str, Any]], config: Mapping[str, Any], component_id: str = "fixture") -> list[Hypothesis]:
    """Build a standalone, solver-valid R2 pool from one R1 graph result."""
    return _pool_for_graph(graph, searches, _controls(config), component_id, include_null=True)


def build_candidate_pool(image_record: dict, config: dict) -> dict:
    """Build an image-level R2 pool from one saved R1 ``routes.json`` record."""
    controls = _controls(config)
    if not isinstance(image_record, Mapping) or not isinstance(image_record.get("components"), list):
        raise ValueError("image_record must contain R1 components")
    components = image_record["components"]
    anchored_components: dict[str, int] = {}
    parsed = []
    for component in components:
        graph = _graph(component["graph"])
        searches = component.get("searches", {})
        for head_id in graph.anchors:
            anchored_components[str(head_id)] = anchored_components.get(str(head_id), 0) + 1
        parsed.append((component, graph, searches))

    # Coverage is comparable only among routes for the same source-image head.
    # Do this before component conversion so a reused head has one shared scale.
    image_max_lengths: dict[str, float] = {}
    for _, _, searches in parsed:
        for head_id, record in _nonnull_records(searches):
            image_max_lengths[head_id] = max(image_max_lengths.get(head_id, 0.0), _length(record.get("points_xy", ())))

    hypotheses: list[Hypothesis] = []
    control_records = []
    missing_routes = []
    all_heads = set()
    for component, graph, searches in parsed:
        tail_id = str(component["tail_id"])
        all_heads.update(str(head_id) for head_id in graph.anchors)
        all_heads.update(str(head_id) for head_id in searches)
        baseline_refs = [dict(row) for row in component.get("baseline", ())]
        anchors = {str(head_id) for head_id in graph.anchors}
        junction_count = sum(getattr(node, "kind", "ordinary") == "junction" for node in graph.nodes.values())
        isolated_head = next(iter(anchors)) if len(anchors) == 1 else None
        matching_baselines = [row for row in baseline_refs if isolated_head is not None and str(row.get("head_id")) == isolated_head]
        isolated = bool(isolated_head and junction_count == 0 and len(baseline_refs) == 1 and len(matching_baselines) == 1 and anchored_components[isolated_head] == 1)
        if isolated:
            row = matching_baselines[0]
            hypotheses.append(Hypothesis(
                id=f"r2:{tail_id}:{isolated_head}:baseline_control:{row['crop_id']}", head_id=isolated_head, cost=0.0,
                exclusive_segments=tuple(f"{tail_id}:{segment_id}" for segment_id in graph.segments), termination="baseline_control",
                metadata={"kind": "baseline_control", "tail_id": tail_id, "head_id": isolated_head,
                          "roi_xyxy": list(component.get("roi_xyxy", ())), "crop_id": row["crop_id"],
                          "source_roi_offset_xy": list(component.get("roi_xyxy", ())[:2]),
                          "pixels_xy_unordered": list(row["pixels_xy"]), "baseline_comparison_refs": baseline_refs,
                          "preservation": "exact saved baseline raster; control, not an accuracy claim"},
            ))
            control_records.append({"tail_id": tail_id, "head_id": isolated_head, "crop_id": row["crop_id"],
                                    "r1_alternatives_retained_for_inspection": list(searches.get(isolated_head, {}).get("hypotheses", ()))})
        else:
            hypotheses.extend(_pool_for_graph(graph, searches, controls, tail_id, include_null=False,
                                               roi_xyxy=component.get("roi_xyxy"), baseline_refs=baseline_refs,
                                               max_lengths=image_max_lengths))
        for head_id in searches:
            paths = [record for candidate_head, record in _nonnull_records(searches) if candidate_head == str(head_id)]
            maximum = max((_length(path.get("points_xy", ())) for path in paths), default=0.0)
            baseline_sizes = [len(row.get("pixels_xy", ())) for row in baseline_refs if str(row.get("head_id")) == str(head_id)]
            if baseline_sizes and maximum < 0.5 * max(baseline_sizes):
                missing_routes.append({"tail_id": tail_id, "head_id": str(head_id), "max_r1_path_arclength": maximum,
                                       "baseline_pixel_count": max(baseline_sizes), "interpretation": "approximate missing-route diagnostic; baseline raster is not biological truth"})
    hypotheses.extend(Hypothesis(f"r2:null:{head_id}", head_id, controls["null_cost"], termination="null",
                                 metadata={"kind": "null", "head_id": head_id, "cost_decomposition": {"null_cost": controls["null_cost"]}})
                        for head_id in sorted(all_heads))
    return {"hypotheses": hypotheses,
            "controls": {"schema_version": _SCHEMA_VERSION, "cost_preferences": controls,
                         "normalization": "path coverage relative to the longest R1 route for the same head across image components; heuristic, not a typical-tail-length prior",
                         "isolated_control_policy": "one anchored head, no junctions, one matching saved baseline, and no other anchored component for that head"},
            "pool_diagnostics": {"component_count": len(components), "head_count": len(all_heads),
                                 "isolated_controls": control_records, "approximate_missing_route_heads": missing_routes}}
