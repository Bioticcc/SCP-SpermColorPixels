"""Synthetic R2 endpoint-competition geometry for joint-assignment tests.

This is a controlled geometric conflict, not biological ground truth.  It
deliberately makes independent route ranking choose the same right endpoint so
later R2 assignment can demonstrate its one-to-one completeness constraint.
"""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from .synthetic_fixtures import SyntheticFixture


_SHAPE = (128, 128)
_TAIL_RGB = (135, 82, 40)
_HEAD_RGB = (92, 60, 32)
_WIDTHS = {1, 3, 5}


def _edge(edge_id: str, start: str, end: str, points: list[tuple[int, int]]) -> dict[str, Any]:
    return {"id": edge_id, "start": start, "end": end, "points": [list(point) for point in points]}


def _path(instance_id: str, node_ids: list[str], edge_ids: list[str]) -> dict[str, Any]:
    return {"instance_id": instance_id, "node_ids": node_ids, "edge_ids": edge_ids}


def _scene() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    heads = [
        {"id": "h_upper", "center_xy": [14, 38], "radius_px": 7},
        {"id": "h_lower", "center_xy": [14, 90], "radius_px": 7},
    ]
    nodes = [
        {"id": "h_upper", "xy": [20, 38], "kind": "head_anchor"},
        {"id": "h_lower", "xy": [20, 90], "kind": "head_anchor"},
        {"id": "j", "xy": [64, 64], "kind": "junction"},
        {"id": "end_right", "xy": [112, 64], "kind": "endpoint"},
        {"id": "end_top", "xy": [64, 10], "kind": "endpoint"},
    ]
    edges = [
        _edge("upper_j", "h_upper", "j", [(20, 38), (37, 41), (53, 58), (64, 64)]),
        # Its final port is intentionally mostly rightward, so local route
        # ranking also favours the right endpoint over the upward endpoint.
        _edge("lower_j", "h_lower", "j", [(20, 90), (39, 87), (51, 72), (55, 65), (64, 64)]),
        _edge("j_right", "j", "end_right", [(64, 64), (88, 64), (112, 64)]),
        _edge("j_top", "j", "end_top", [(64, 64), (64, 37), (64, 10)]),
    ]
    paths = [
        _path("sperm_upper", ["h_upper", "j", "end_right"], ["upper_j", "j_right"]),
        _path("sperm_lower", ["h_lower", "j", "end_top"], ["lower_j", "j_top"]),
    ]
    return heads, nodes, edges, paths


def _render(
    heads: list[dict[str, Any]], edges: list[dict[str, Any]], paths: list[dict[str, Any]], seed: int, width_px: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    base = rng.integers(238, 244, size=(*_SHAPE, 1), dtype=np.uint8)
    rgb = np.repeat(base, 3, axis=2)
    tail_mask = np.zeros(_SHAPE, dtype=bool)
    edge_masks: dict[str, np.ndarray] = {}
    for edge in edges:
        mask = np.zeros(_SHAPE, dtype=np.uint8)
        cv2.polylines(mask, [np.asarray(edge["points"], dtype=np.int32)], False, 1, width_px, lineType=cv2.LINE_8)
        edge_masks[edge["id"]] = mask.astype(bool)
        tail_mask |= edge_masks[edge["id"]]
    head_u8 = np.zeros(_SHAPE, dtype=np.uint8)
    for head in heads:
        cv2.circle(head_u8, tuple(head["center_xy"]), int(head["radius_px"]), 1, thickness=-1, lineType=cv2.LINE_8)
    head_mask = head_u8.astype(bool)
    rgb[tail_mask] = _TAIL_RGB
    rgb[head_mask] = _HEAD_RGB
    instance_masks = {
        path["instance_id"]: np.logical_or.reduce([edge_masks[edge_id] for edge_id in path["edge_ids"]])
        for path in paths
    }
    return rgb, tail_mask, head_mask, instance_masks


def generate_competing_case(seed: int = 0, width_px: int = 3) -> SyntheticFixture:
    """Generate a deterministic two-head, one-endpoint local-ranking conflict."""
    if width_px not in _WIDTHS:
        raise ValueError("width_px must be one of 1, 3, or 5")
    heads, nodes, edges, paths = _scene()
    rgb, tail_mask, head_mask, instance_masks = _render(heads, edges, paths, seed, width_px)
    edge_by_id = {edge["id"]: edge for edge in edges}
    truth = {
        "schema_version": "r2.synthetic-fixture.v1",
        "name": "r2_competing_endpoint",
        "seed": int(seed),
        "width_px": int(width_px),
        "heads": heads,
        "instances": [
            {"id": path["instance_id"], "head_id": path["node_ids"][0],
             "centerline_xy": [point for index, edge_id in enumerate(path["edge_ids"])
                               for point in (edge_by_id[edge_id]["points"] if index == 0 else edge_by_id[edge_id]["points"][1:])]}
            for path in paths
        ],
        "graph": {"nodes": nodes, "edges": edges},
        "true_edge_paths": paths,
        "expected_behavior": {
            "determinate": True,
            "outcome": "requires_joint_endpoint_assignment",
            "expected_instance_count": 2,
            "local_conflict_endpoint": "end_right",
            "declared_smooth_pairing_assumption": "Synthetic assignment convention only; not real biological gold.",
        },
        "identity_solutions": [{"id": "declared", "paths": paths}],
    }
    return SyntheticFixture("r2_competing_endpoint", int(seed), rgb, tail_mask, head_mask, instance_masks, truth)
