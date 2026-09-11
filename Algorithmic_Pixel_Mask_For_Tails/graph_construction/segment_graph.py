"""Build a compressed branch/segment graph from a one-pixel skeleton.

Coordinates in this module are image-local ``(x, y)`` pairs.  Junction pixels
are deliberately represented by one node and boundary ports, rather than by
invented chords through the centre of a thick skeleton junction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Mapping

import cv2
import numpy as np


XY = tuple[float, float]
PixelXY = tuple[int, int]


@dataclass(frozen=True)
class Node:
    id: str
    xy: XY
    kind: str = "ordinary"
    pixels_xy: tuple[PixelXY, ...] = ()


@dataclass(frozen=True)
class Segment:
    id: str
    start: str
    end: str
    points_xy: tuple[XY, ...]
    radii: tuple[float, ...] = ()
    evidence: float = 1.0


@dataclass(frozen=True)
class SegmentGraph:
    nodes: dict[str, Node]
    segments: dict[str, Segment]
    anchors: dict[str, str]
    shape: tuple[int, int]
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def outgoing(self, node_id: str) -> list[tuple[str, bool]]:
        """Return incident segments in deterministic ID/direction order."""
        if node_id not in self.nodes:
            raise KeyError(node_id)
        result: list[tuple[str, bool]] = []
        for segment_id, segment in self.segments.items():
            if segment.start == node_id:
                result.append((segment_id, True))
            if segment.end == node_id:
                # A self-loop intentionally has both travel directions.
                result.append((segment_id, False))
        return sorted(result, key=lambda item: (item[0], not item[1]))

    def points(self, segment_id: str, forward: bool = True) -> np.ndarray:
        segment = self.segments[segment_id]
        points = np.asarray(segment.points_xy, dtype=np.float32).reshape((-1, 2))
        return points if forward else points[::-1].copy()

    def destination(self, segment_id: str, forward: bool) -> str:
        segment = self.segments[segment_id]
        return segment.end if forward else segment.start

    def to_dict(self) -> dict[str, Any]:
        return {
            "shape": [int(self.shape[0]), int(self.shape[1])],
            "nodes": [
                {
                    "id": node.id,
                    "xy": [float(node.xy[0]), float(node.xy[1])],
                    "kind": node.kind,
                    "pixels_xy": [[int(x), int(y)] for x, y in node.pixels_xy],
                }
                for _, node in sorted(self.nodes.items())
            ],
            "segments": [
                {
                    "id": segment.id,
                    "start": segment.start,
                    "end": segment.end,
                    "points_xy": [[float(x), float(y)] for x, y in segment.points_xy],
                    "radii": [float(radius) for radius in segment.radii],
                    "evidence": float(segment.evidence),
                }
                for _, segment in sorted(self.segments.items())
            ],
            "anchors": dict(sorted(self.anchors.items())),
            "diagnostics": _json_value(self.diagnostics),
        }


def _json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _finite_pair(value: Any, field_name: str) -> XY:
    """Read a two-coordinate geometry field without silently coercing it."""
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{field_name} must contain exactly two coordinates")
    try:
        xy = (float(value[0]), float(value[1]))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field_name} must contain numeric coordinates") from error
    if not all(math.isfinite(item) for item in xy):
        raise ValueError(f"{field_name} coordinates must be finite")
    return xy


def _pixel_pair(value: Any, field_name: str) -> PixelXY:
    xy = _finite_pair(value, field_name)
    if not all(item.is_integer() for item in xy):
        raise ValueError(f"{field_name} coordinates must be integers")
    return (int(xy[0]), int(xy[1]))


def _shape_pair(value: Any) -> tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("shape must contain exactly height and width")
    try:
        numeric_shape = tuple(float(item) for item in value)
        shape = tuple(int(item) for item in numeric_shape)
    except (TypeError, ValueError) as error:
        raise ValueError("shape dimensions must be integers") from error
    if (not all(math.isfinite(item) for item in numeric_shape)
            or numeric_shape != tuple(float(item) for item in shape)
            or min(shape) <= 0):
        raise ValueError("shape dimensions must be positive integers")
    return shape  # type: ignore[return-value]


def from_explicit_graph(
    graph_dict: Mapping[str, Any],
    shape: tuple[int, int] = (128, 128),
    anchors: Mapping[str, str] | None = None,
) -> SegmentGraph:
    """Create a graph fixture from geometry only; identity truth is ignored."""
    checked_shape = _shape_pair(shape)
    raw_nodes = graph_dict.get("nodes", [])
    node_items = raw_nodes.values() if isinstance(raw_nodes, Mapping) else raw_nodes
    nodes: dict[str, Node] = {}
    for raw in node_items:
        if not isinstance(raw, Mapping) or "id" not in raw or "xy" not in raw:
            raise ValueError("Each node must provide id and xy")
        node_id = str(raw["id"])
        if not node_id or node_id in nodes:
            raise ValueError(f"Duplicate or empty node id: {node_id!r}")
        xy = _finite_pair(raw["xy"], f"Node {node_id} xy")
        nodes[node_id] = Node(
            id=node_id,
            xy=xy,
            kind=str(raw.get("kind", "ordinary")),
            pixels_xy=tuple(_pixel_pair(p, f"Node {node_id} pixels_xy") for p in raw.get("pixels_xy", ())),
        )

    raw_segments = graph_dict.get("segments", graph_dict.get("edges", []))
    segment_items = raw_segments.values() if isinstance(raw_segments, Mapping) else raw_segments
    segments: dict[str, Segment] = {}
    for index, raw in enumerate(segment_items):
        if not isinstance(raw, Mapping) or "start" not in raw or "end" not in raw:
            raise ValueError("Each segment must provide start and end")
        segment_id = str(raw.get("id", f"segment_{index:03d}"))
        if not segment_id or segment_id in segments:
            raise ValueError(f"Duplicate or empty segment id: {segment_id!r}")
        start, end = str(raw["start"]), str(raw["end"])
        if start not in nodes or end not in nodes:
            raise ValueError(f"Segment {segment_id} refers to an unknown node")
        points = tuple(_finite_pair(p, f"Segment {segment_id} points_xy") for p in raw.get("points_xy", raw.get("points", ())))
        if not points:
            points = (nodes[start].xy, nodes[end].xy)
        radii = tuple(float(radius) for radius in raw.get("radii", ()))
        if radii and len(radii) != len(points):
            raise ValueError(f"Segment {segment_id} radii must be empty or match points_xy")
        if not all(math.isfinite(radius) and radius >= 0.0 for radius in radii):
            raise ValueError(f"Segment {segment_id} radii must be finite and non-negative")
        try:
            segment_evidence = float(raw.get("evidence", 1.0))
        except (TypeError, ValueError) as error:
            raise ValueError(f"Segment {segment_id} evidence must be numeric") from error
        if not math.isfinite(segment_evidence):
            raise ValueError(f"Segment {segment_id} evidence must be finite")
        segments[segment_id] = Segment(segment_id, start, end, points, radii, segment_evidence)

    chosen_anchors = dict(anchors) if anchors is not None else {
        node_id: node_id for node_id, node in nodes.items() if node.kind == "head_anchor"
    }
    for anchor_id, node_id in chosen_anchors.items():
        if not str(anchor_id):
            raise ValueError("Anchor ids must be non-empty")
        if node_id not in nodes:
            raise ValueError(f"Anchor {anchor_id} refers to an unknown node {node_id}")
    return SegmentGraph(nodes, segments, chosen_anchors, checked_shape, dict(graph_dict.get("diagnostics", {})))


def _neighbors(points: set[PixelXY]) -> dict[PixelXY, set[PixelXY]]:
    """8-neighbour graph without redundant diagonal triangle shortcuts."""
    result: dict[PixelXY, set[PixelXY]] = {point: set() for point in points}
    for x, y in points:
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                other = (x + dx, y + dy)
                if other not in points:
                    continue
                if dx and dy and ((x + dx, y) in points or (x, y + dy) in points):
                    continue
                result[(x, y)].add(other)
    return result


def _components(points: set[PixelXY], neighbors: Mapping[PixelXY, set[PixelXY]]) -> list[set[PixelXY]]:
    unseen = set(points)
    found: list[set[PixelXY]] = []
    while unseen:
        root = min(unseen, key=lambda point: (point[1], point[0]))
        component = {root}
        pending = [root]
        unseen.remove(root)
        while pending:
            current = pending.pop()
            for neighbor in neighbors[current]:
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    component.add(neighbor)
                    pending.append(neighbor)
        found.append(component)
    return found


def build_segment_graph(
    skeleton: np.ndarray,
    tail_mask: np.ndarray | None = None,
    anchors_xy: Mapping[str, tuple[float, float]] | None = None,
    evidence: np.ndarray | None = None,
) -> SegmentGraph:
    """Compress skeleton branches into deterministic nodes and segments.

    Adjacent degree-three-or-more pixels form one junction node.  Segment
    geometry stops at its exterior ports, so a later transition module can
    decide which ports connect without fabricating a centreline chord.
    """
    if skeleton.ndim != 2:
        raise ValueError("skeleton must be a two-dimensional array")
    shape = (int(skeleton.shape[0]), int(skeleton.shape[1]))
    points: set[PixelXY] = {(int(x), int(y)) for y, x in np.argwhere(skeleton > 0)}
    diagnostics: dict[str, Any] = {
        "input_skeleton_pixels": len(points),
        "dropped_skeleton_pixels": 0,
        "unrepresented_skeleton_pixels": 0,
        "anchor_snaps": {},
        "anchor_collisions": [],
    }
    if not points:
        return SegmentGraph({}, {}, {}, shape, diagnostics)

    neighbors = _neighbors(points)
    junction_pixels = {point for point in points if len(neighbors[point]) >= 3}
    junction_components = _components(junction_pixels, {
        point: {other for other in neighbors[point] if other in junction_pixels}
        for point in junction_pixels
    }) if junction_pixels else []

    # The extractor receives a one-pixel skeleton.  A large fraction of a
    # component classified as one connected degree-three core is therefore a
    # dense/unthinned raster, not a crossing neighbourhood.  Compressing it
    # would silently turn long tracks into one giant junction and make later
    # continuation scoring impossible.  Fail at this representation boundary
    # so the caller can thin or repair the mask first.
    skeleton_components = _components(points, neighbors)
    for component in junction_components:
        owner = next(source for source in skeleton_components if component <= source)
        if len(component) > 64 and len(component) / len(owner) > 0.5:
            raise ValueError(
                "Skeleton has an oversized dense junction core; expected a one-pixel thinning "
                "before graph construction"
            )

    nodes: dict[str, Node] = {}
    point_node: dict[PixelXY, str] = {}
    for index, component in enumerate(junction_components, 1):
        pixels = tuple(sorted(component, key=lambda point: (point[1], point[0])))
        node_id = f"junction_{index:03d}"
        xy = (float(np.mean([point[0] for point in pixels])), float(np.mean([point[1] for point in pixels])))
        nodes[node_id] = Node(node_id, xy, "junction", pixels)
        point_node.update({point: node_id for point in pixels})

    endpoint_points = sorted((point for point in points if len(neighbors[point]) <= 1 and point not in point_node), key=lambda point: (point[1], point[0]))
    for index, point in enumerate(endpoint_points, 1):
        node_id = f"endpoint_{index:03d}"
        is_boundary = point[0] in (0, shape[1] - 1) or point[1] in (0, shape[0] - 1)
        nodes[node_id] = Node(
            node_id,
            (float(point[0]), float(point[1])),
            "boundary_port" if is_boundary else "endpoint",
            (point,),
        )
        point_node[point] = node_id

    anchors: dict[str, str] = {}
    if anchors_xy:
        available = sorted(points, key=lambda point: (point[1], point[0]))
        snapped_nodes: dict[PixelXY, str] = {}
        claimed_nodes: dict[str, str] = {}
        for anchor_id, requested in sorted(anchors_xy.items(), key=lambda item: str(item[0])):
            anchor_key = str(anchor_id)
            if not anchor_key:
                raise ValueError("Anchor ids must be non-empty")
            ax, ay = _finite_pair(requested, f"Anchor {anchor_key}")
            snapped = min(available, key=lambda point: ((point[0] - ax) ** 2 + (point[1] - ay) ** 2, point[1], point[0]))
            distance = float(np.hypot(snapped[0] - ax, snapped[1] - ay))
            if snapped in snapped_nodes:
                node_id = snapped_nodes[snapped]
                diagnostics["anchor_collisions"].append({
                    "anchor_id": anchor_key,
                    "node_id": node_id,
                    "with_anchor_id": claimed_nodes[node_id],
                    "reason": "shared_snapped_pixel",
                })
            elif snapped in point_node:
                node_id = point_node[snapped]
                if node_id in claimed_nodes:
                    diagnostics["anchor_collisions"].append({
                        "anchor_id": anchor_key,
                        "node_id": node_id,
                        "with_anchor_id": claimed_nodes[node_id],
                        "reason": "shared_structural_node",
                    })
                else:
                    diagnostics.setdefault("anchor_on_node", []).append({
                        "anchor_id": anchor_key,
                        "node_id": node_id,
                        "node_kind": nodes[node_id].kind,
                    })
            else:
                node_id = f"anchor_{len([n for n in nodes if n.startswith('anchor_')]) + 1:03d}"
                nodes[node_id] = Node(node_id, (float(snapped[0]), float(snapped[1])), "head_anchor", (snapped,))
                point_node[snapped] = node_id
                snapped_nodes[snapped] = node_id
            claimed_nodes.setdefault(node_id, anchor_key)
            anchors[anchor_key] = node_id
            diagnostics["anchor_snaps"][anchor_key] = {
                "requested_xy": [ax, ay], "snapped_xy": [int(snapped[0]), int(snapped[1])], "distance_px": distance,
            }

    # A degree-two-only component is a loop.  Give it an explicit node so it
    # remains traversable and its self-segment has both directions.
    for component in skeleton_components:
        if not any(point in point_node for point in component):
            loop_point = min(component, key=lambda point: (point[1], point[0]))
            node_id = f"loop_{len([n for n in nodes if n.startswith('loop_')]) + 1:03d}"
            nodes[node_id] = Node(node_id, (float(loop_point[0]), float(loop_point[1])), "loop", (loop_point,))
            point_node[loop_point] = node_id

    distance_source = (tail_mask > 0).astype(np.uint8) if tail_mask is not None else (skeleton > 0).astype(np.uint8)
    if distance_source.shape != skeleton.shape:
        raise ValueError("tail_mask must match skeleton shape")
    distances = cv2.distanceTransform(distance_source, cv2.DIST_L2, 5)
    if evidence is not None and evidence.shape != skeleton.shape:
        raise ValueError("evidence must match skeleton shape")

    segments: dict[str, Segment] = {}
    used_edges: set[frozenset[PixelXY]] = set()
    all_edges = {
        frozenset((point, other))
        for point, adjacent in neighbors.items()
        for other in adjacent
    }

    def add_segment(start_id: str, end_id: str, trace: list[PixelXY]) -> None:
        if not trace:
            return
        segment_id = f"segment_{len(segments) + 1:03d}"
        geometry = tuple((float(x), float(y)) for x, y in trace)
        radii = tuple(float(distances[y, x]) for x, y in trace)
        score = 1.0 if evidence is None else float(np.mean([evidence[y, x] for x, y in trace]))
        segments[segment_id] = Segment(segment_id, start_id, end_id, geometry, radii, score)

    # Start at every special-node boundary.  A junction contributes only its
    # outward boundary edges; its internal pixels never become a chord.
    for start_point, start_id in sorted(point_node.items(), key=lambda item: (item[1], item[0][1], item[0][0])):
        for first in sorted(neighbors[start_point], key=lambda point: (point[1], point[0])):
            if point_node.get(first) == start_id:
                continue
            edge = frozenset((start_point, first))
            if edge in used_edges:
                continue
            used_edges.add(edge)
            trace = [] if nodes[start_id].kind == "junction" else [start_point]
            previous, current = start_point, first
            while True:
                if current in point_node:
                    end_id = point_node[current]
                    if nodes[end_id].kind != "junction" or end_id == start_id:
                        trace.append(current)
                    add_segment(start_id, end_id, trace)
                    break
                trace.append(current)
                next_points = [point for point in neighbors[current] if point != previous]
                if not next_points:
                    diagnostics["dropped_skeleton_pixels"] += 1
                    break
                next_point = min(next_points, key=lambda point: (point[1], point[0]))
                next_edge = frozenset((current, next_point))
                if next_edge in used_edges:
                    # This is the return to an explicit loop node or a segment
                    # already represented from its other endpoint.
                    if next_point in point_node:
                        add_segment(start_id, point_node[next_point], trace)
                    break
                used_edges.add(next_edge)
                previous, current = current, next_point

    represented = set(point_node)
    for segment in segments.values():
        represented.update((int(x), int(y)) for x, y in segment.points_xy)
    diagnostics["segment_count"] = len(segments)
    diagnostics["node_count"] = len(nodes)
    diagnostics["junction_count"] = len(junction_components)
    diagnostics["unrepresented_skeleton_pixels"] = len(points - represented)
    internal_node_edges = {
        edge for edge in all_edges
        if len({point_node.get(point) for point in edge}) == 1
        and next(iter(edge)) in point_node
    }
    diagnostics["input_skeleton_edges"] = len(all_edges)
    diagnostics["represented_skeleton_edges"] = len(used_edges | internal_node_edges)
    diagnostics["unrepresented_skeleton_edges"] = len(all_edges - used_edges - internal_node_edges)
    return SegmentGraph(nodes, segments, anchors, shape, diagnostics)
