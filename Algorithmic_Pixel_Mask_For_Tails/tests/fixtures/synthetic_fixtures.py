"""Deterministic graph and rendered fixtures for overlap-resolution work.

Truth intentionally describes logical tracks and graph topology rather than any
particular thinning result.  Rendered RGB and binary masks are only evidence
that an upstream image-processing stage may consume.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


_CANVAS_SHAPE = (128, 128)
_TAIL_RGB = (135, 82, 40)
_HEAD_RGB = (92, 60, 32)


@dataclass(frozen=True)
class SyntheticFixture:
    """One known synthetic scene, with evidence and independently declared truth."""

    name: str
    seed: int
    rgb: np.ndarray
    tail_mask: np.ndarray
    head_mask: np.ndarray
    instance_masks: dict[str, np.ndarray]
    truth: dict[str, Any]

    def to_truth_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable copy of the graph and identity truth."""
        return json.loads(json.dumps(self.truth, sort_keys=True))

    def export_truth(self, path: str | Path) -> None:
        """Write only declared truth; rendered arrays are deliberately excluded."""
        Path(path).write_text(json.dumps(self.to_truth_dict(), indent=2, sort_keys=True) + "\n")


def render_identity_solution(fixture: SyntheticFixture, solution_id: str) -> tuple[np.ndarray, np.ndarray]:
    """Render a declared identity solution from its actual graph edges.

    This deliberately renders route evidence from the solution rather than
    returning the fixture's cached evidence, so ambiguity tests can establish
    whether distinct identities are observationally equivalent.
    """
    solutions = {solution["id"]: solution for solution in fixture.truth["identity_solutions"]}
    try:
        solution = solutions[solution_id]
    except KeyError as exc:
        raise ValueError(f"Unknown identity solution {solution_id!r}") from exc
    edges = {edge["id"]: edge for edge in fixture.truth["graph"]["edges"]}
    selected_edge_ids = {edge_id for path in solution["paths"] for edge_id in path["edge_ids"]}
    rng = np.random.default_rng(fixture.seed)
    base = rng.integers(238, 244, size=(*fixture.rgb.shape[:2], 1), dtype=np.uint8)
    rgb = np.repeat(base, 3, axis=2)
    tail_mask = np.zeros(fixture.rgb.shape[:2], dtype=np.uint8)
    for edge_id in selected_edge_ids:
        cv2.polylines(tail_mask, [np.asarray(edges[edge_id]["points"], dtype=np.int32)], False, 1, thickness=3, lineType=cv2.LINE_8)
    head_mask = np.zeros(fixture.rgb.shape[:2], dtype=np.uint8)
    for head in fixture.truth["heads"]:
        cv2.circle(head_mask, tuple(head["center_xy"]), int(head["radius_px"]), 1, thickness=-1, lineType=cv2.LINE_8)
    rgb[tail_mask.astype(bool)] = _TAIL_RGB
    rgb[head_mask.astype(bool)] = _HEAD_RGB
    return rgb, tail_mask.astype(bool)


def case_names() -> tuple[str, ...]:
    return (
        "isolated_curve",
        "same_color_x",
        "curved_crossing",
        "competing_heads",
        "self_loop",
        "t_contact",
        "sharp_supported_bend",
        "indeterminate_identity",
    )


def load_case(name: str, seed: int = 0) -> SyntheticFixture:
    """Load a procedural case; retained as the stable harness-facing name."""
    return generate_case(name, seed=seed)


def _edge(edge_id: str, start: str, end: str, points: list[tuple[int, int]]) -> dict[str, Any]:
    return {"id": edge_id, "start": start, "end": end, "points": [list(point) for point in points]}


def _path(instance_id: str, node_ids: list[str], edge_ids: list[str]) -> dict[str, Any]:
    return {"instance_id": instance_id, "node_ids": node_ids, "edge_ids": edge_ids}


def _scene(name: str) -> dict[str, Any]:
    # Coordinates are x, y.  Each graph edge has explicit geometry so later
    # graph code can test routes without thinning a raster mask.
    scenes: dict[str, dict[str, Any]] = {
        "isolated_curve": {
            "heads": [("h1", (16, 99), 8)],
            "nodes": [("h1", (20, 96), "head_anchor"), ("b1", (42, 74), "ordinary"), ("b2", (72, 54), "ordinary"), ("e1", (108, 27), "endpoint")],
            "edges": [
                _edge("e_h_b1", "h1", "b1", [(20, 96), (29, 87), (42, 74)]),
                _edge("e_b1_b2", "b1", "b2", [(42, 74), (55, 64), (72, 54)]),
                _edge("e_b2_end", "b2", "e1", [(72, 54), (91, 42), (108, 27)]),
            ],
            "instances": ["sperm_1"],
            "paths": [_path("sperm_1", ["h1", "b1", "b2", "e1"], ["e_h_b1", "e_b1_b2", "e_b2_end"])],
            "expected": {"determinate": True, "outcome": "single_supported_route", "expected_instance_count": 1},
        },
        "same_color_x": {
            "heads": [("h_left", (14, 64), 7), ("h_top", (64, 14), 7)],
            "nodes": [("h_left", (20, 64), "head_anchor"), ("h_top", (64, 20), "head_anchor"), ("j", (64, 64), "junction"), ("end_right", (108, 64), "endpoint"), ("end_bottom", (64, 108), "endpoint")],
            "edges": [_edge("left_j", "h_left", "j", [(20, 64), (64, 64)]), _edge("top_j", "h_top", "j", [(64, 20), (64, 64)]), _edge("j_right", "j", "end_right", [(64, 64), (108, 64)]), _edge("j_bottom", "j", "end_bottom", [(64, 64), (64, 108)])],
            "instances": ["sperm_horizontal", "sperm_vertical"],
            "paths": [_path("sperm_horizontal", ["h_left", "j", "end_right"], ["left_j", "j_right"]), _path("sperm_vertical", ["h_top", "j", "end_bottom"], ["top_j", "j_bottom"])],
            "expected": {"determinate": True, "outcome": "straight_continuations", "expected_instance_count": 2, "assumption": "smooth_straight_continuations_are_preferred"},
        },
        "curved_crossing": {
            "heads": [("h_left", (12, 79), 7), ("h_top", (58, 12), 7)],
            "nodes": [("h_left", (18, 78), "head_anchor"), ("h_top", (59, 18), "head_anchor"), ("j", (63, 63), "junction"), ("end_right", (112, 42), "endpoint"), ("end_bottom", (82, 112), "endpoint")],
            "edges": [_edge("left_j", "h_left", "j", [(18, 78), (38, 80), (53, 73), (63, 63)]), _edge("top_j", "h_top", "j", [(59, 18), (54, 37), (56, 53), (63, 63)]), _edge("j_right", "j", "end_right", [(63, 63), (77, 51), (95, 43), (112, 42)]), _edge("j_bottom", "j", "end_bottom", [(63, 63), (68, 80), (75, 98), (82, 112)])],
            "instances": ["sperm_curve_a", "sperm_curve_b"],
            "paths": [_path("sperm_curve_a", ["h_left", "j", "end_right"], ["left_j", "j_right"]), _path("sperm_curve_b", ["h_top", "j", "end_bottom"], ["top_j", "j_bottom"])],
            "expected": {"determinate": True, "outcome": "curvature_consistent_continuations", "expected_instance_count": 2},
        },
        "competing_heads": {
            "heads": [("h_left", (14, 55), 7), ("h_bottom", (62, 114), 7)],
            "nodes": [("h_left", (20, 55), "head_anchor"), ("h_bottom", (62, 108), "head_anchor"), ("j", (62, 62), "junction"), ("end_right", (110, 62), "endpoint"), ("end_top", (62, 18), "endpoint")],
            "edges": [_edge("left_j", "h_left", "j", [(20, 55), (43, 58), (62, 62)]), _edge("bottom_j", "h_bottom", "j", [(62, 108), (62, 62)]), _edge("j_right", "j", "end_right", [(62, 62), (110, 62)]), _edge("j_top", "j", "end_top", [(62, 62), (62, 18)])],
            "instances": ["sperm_left", "sperm_bottom"],
            "paths": [_path("sperm_left", ["h_left", "j", "end_right"], ["left_j", "j_right"]), _path("sperm_bottom", ["h_bottom", "j", "end_top"], ["bottom_j", "j_top"])],
            "expected": {"determinate": True, "outcome": "requires_global_assignment", "expected_instance_count": 2, "conflicting_edge_ids": ["j_right", "j_top"]},
        },
        "self_loop": {
            "heads": [("h1", (15, 64), 7)],
            "nodes": [("h1", (21, 64), "head_anchor"), ("j", (48, 64), "junction"), ("loop_top", (71, 34), "ordinary"), ("loop_right", (96, 64), "ordinary"), ("end", (23, 105), "endpoint")],
            "edges": [_edge("head_j", "h1", "j", [(21, 64), (48, 64)]), _edge("loop_up", "j", "loop_top", [(48, 64), (55, 43), (71, 34)]), _edge("loop_across", "loop_top", "loop_right", [(71, 34), (88, 39), (96, 64)]), _edge("loop_return", "loop_right", "j", [(96, 64), (81, 82), (62, 81), (48, 64)]), _edge("exit", "j", "end", [(48, 64), (38, 83), (23, 105)])],
            "instances": ["sperm_loop"],
            "paths": [_path("sperm_loop", ["h1", "j", "loop_top", "loop_right", "j", "end"], ["head_j", "loop_up", "loop_across", "loop_return", "exit"])],
            "expected": {"determinate": True, "outcome": "junction_revisit_allowed", "expected_instance_count": 1},
        },
        "t_contact": {
            "heads": [("h_left", (14, 64), 7), ("h_top", (64, 14), 7)],
            "nodes": [("h_left", (20, 64), "head_anchor"), ("h_top", (64, 20), "head_anchor"), ("j", (64, 64), "junction"), ("end_right", (110, 64), "endpoint")],
            "edges": [_edge("left_j", "h_left", "j", [(20, 64), (64, 64)]), _edge("top_j", "h_top", "j", [(64, 20), (64, 64)]), _edge("j_right", "j", "end_right", [(64, 64), (110, 64)])],
            "instances": ["sperm_main", "contact_fragment"],
            "paths": [_path("sperm_main", ["h_left", "j", "end_right"], ["left_j", "j_right"]), _path("contact_fragment", ["h_top", "j"], ["top_j"])],
            "expected": {"determinate": True, "outcome": "t_contact_no_forced_continuation", "expected_instance_count": 1, "partial_instance_ids": ["contact_fragment"]},
        },
        "sharp_supported_bend": {
            "heads": [("h1", (15, 96), 7)],
            "nodes": [("h1", (21, 93), "head_anchor"), ("bend", (61, 65), "ordinary"), ("end", (103, 92), "endpoint")],
            "edges": [_edge("head_bend", "h1", "bend", [(21, 93), (42, 76), (61, 65)]), _edge("bend_end", "bend", "end", [(61, 65), (66, 77), (83, 88), (103, 92)])],
            "instances": ["sperm_bend"],
            "paths": [_path("sperm_bend", ["h1", "bend", "end"], ["head_bend", "bend_end"])],
            "expected": {"determinate": True, "outcome": "supported_sharp_bend_retained", "expected_instance_count": 1},
        },
        "indeterminate_identity": {
            "heads": [("h_upper", (14, 46), 7), ("h_lower", (14, 82), 7)],
            "nodes": [("h_upper", (20, 46), "head_anchor"), ("h_lower", (20, 82), "head_anchor"), ("merge", (44, 64), "junction"), ("corridor_a", (60, 64), "shared_corridor"), ("corridor_b", (78, 64), "shared_corridor"), ("split", (92, 64), "junction"), ("end_upper", (112, 46), "endpoint"), ("end_lower", (112, 82), "endpoint")],
            "edges": [_edge("upper_merge", "h_upper", "merge", [(20, 46), (32, 51), (44, 64)]), _edge("lower_merge", "h_lower", "merge", [(20, 82), (32, 77), (44, 64)]), _edge("corridor_1", "merge", "corridor_a", [(44, 64), (60, 64)]), _edge("corridor_2", "corridor_a", "corridor_b", [(60, 64), (78, 64)]), _edge("corridor_3", "corridor_b", "split", [(78, 64), (92, 64)]), _edge("split_upper", "split", "end_upper", [(92, 64), (102, 56), (112, 46)]), _edge("split_lower", "split", "end_lower", [(92, 64), (102, 72), (112, 82)])],
            "instances": ["identity_a", "identity_b"],
            "paths": [_path("identity_a", ["h_upper", "merge", "corridor_a", "corridor_b", "split", "end_upper"], ["upper_merge", "corridor_1", "corridor_2", "corridor_3", "split_upper"]), _path("identity_b", ["h_lower", "merge", "corridor_a", "corridor_b", "split", "end_lower"], ["lower_merge", "corridor_1", "corridor_2", "corridor_3", "split_lower"])],
            "identity_solutions": [
                {"id": "preserved_order", "paths": [_path("identity_a", ["h_upper", "merge", "corridor_a", "corridor_b", "split", "end_upper"], ["upper_merge", "corridor_1", "corridor_2", "corridor_3", "split_upper"]), _path("identity_b", ["h_lower", "merge", "corridor_a", "corridor_b", "split", "end_lower"], ["lower_merge", "corridor_1", "corridor_2", "corridor_3", "split_lower"])]},
                {"id": "swapped_order", "paths": [_path("identity_a", ["h_upper", "merge", "corridor_a", "corridor_b", "split", "end_lower"], ["upper_merge", "corridor_1", "corridor_2", "corridor_3", "split_lower"]), _path("identity_b", ["h_lower", "merge", "corridor_a", "corridor_b", "split", "end_upper"], ["lower_merge", "corridor_1", "corridor_2", "corridor_3", "split_upper"])]},
            ],
            "expected": {"determinate": False, "outcome": "abstain_with_identity_alternatives", "expected_instance_count": 2, "shared_corridor_edge_ids": ["corridor_1", "corridor_2", "corridor_3"]},
        },
    }
    try:
        return scenes[name]
    except KeyError as exc:
        raise ValueError(f"Unknown synthetic fixture {name!r}; choose from {case_names()}") from exc


def _render(scene: dict[str, Any], seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    base = rng.integers(238, 244, size=(*_CANVAS_SHAPE, 1), dtype=np.uint8)
    rgb = np.repeat(base, 3, axis=2)
    tail_mask = np.zeros(_CANVAS_SHAPE, dtype=bool)
    head_mask_u8 = np.zeros(_CANVAS_SHAPE, dtype=np.uint8)
    edge_masks: dict[str, np.ndarray] = {}
    for edge in scene["edges"]:
        mask = np.zeros(_CANVAS_SHAPE, dtype=np.uint8)
        cv2.polylines(mask, [np.asarray(edge["points"], dtype=np.int32)], False, 1, thickness=3, lineType=cv2.LINE_8)
        edge_masks[edge["id"]] = mask.astype(bool)
        tail_mask |= edge_masks[edge["id"]]
    for _, center, radius in scene["heads"]:
        cv2.circle(head_mask_u8, center, radius, 1, thickness=-1, lineType=cv2.LINE_8)
    head_mask = head_mask_u8.astype(bool)
    rgb[tail_mask] = _TAIL_RGB
    rgb[head_mask] = _HEAD_RGB
    instance_masks: dict[str, np.ndarray] = {}
    for path in scene["paths"]:
        mask = np.zeros(_CANVAS_SHAPE, dtype=bool)
        for edge_id in path["edge_ids"]:
            mask |= edge_masks[edge_id]
        instance_masks[path["instance_id"]] = mask
    return rgb, tail_mask, head_mask, instance_masks


def _centerline_for_path(path: dict[str, Any], edges_by_id: dict[str, dict[str, Any]]) -> list[list[int]]:
    points: list[list[int]] = []
    for edge_id in path["edge_ids"]:
        edge_points = edges_by_id[edge_id]["points"]
        points.extend(edge_points if not points else edge_points[1:])
    return points


def generate_case(name: str, seed: int = 0) -> SyntheticFixture:
    """Generate deterministic evidence and separately declared graph truth."""
    scene = _scene(name)
    rgb, tail_mask, head_mask, instance_masks = _render(scene, seed)
    edges_by_id = {edge["id"]: edge for edge in scene["edges"]}
    centerlines = {
        path["instance_id"]: _centerline_for_path(path, edges_by_id)
        for path in scene["paths"]
    }
    truth = {
        "schema_version": "r0.synthetic-fixture.v1",
        "name": name,
        "seed": int(seed),
        "heads": [{"id": head_id, "center_xy": list(center), "radius_px": radius} for head_id, center, radius in scene["heads"]],
        "instances": [
            {
                "id": instance_id,
                "head_id": next(path["node_ids"][0] for path in scene["paths"] if path["instance_id"] == instance_id),
                "centerline_xy": centerlines.get(instance_id, []),
            }
            for instance_id in scene["instances"]
        ],
        "graph": {
            "nodes": [{"id": node_id, "xy": list(xy), "kind": kind} for node_id, xy, kind in scene["nodes"]],
            "edges": scene["edges"],
        },
        "true_edge_paths": scene["paths"],
        "expected_behavior": scene["expected"],
        "identity_solutions": scene.get("identity_solutions", [{"id": "declared", "paths": scene["paths"]}]),
    }
    return SyntheticFixture(name, int(seed), rgb, tail_mask, head_mask, instance_masks, truth)
