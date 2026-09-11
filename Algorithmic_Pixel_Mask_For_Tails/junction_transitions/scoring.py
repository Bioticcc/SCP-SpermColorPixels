"""Direction-aware, soft costs for continuations through junction ports."""
from __future__ import annotations

import math
from typing import Any

import numpy as np


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm else np.zeros(2, dtype=float)


def _radii(graph: Any, segment_id: str, forward: bool, count: int) -> np.ndarray:
    values = np.asarray(getattr(graph.segments[segment_id], "radii", ()), dtype=float)
    if values.size != count:
        return np.ones(count, dtype=float)
    return values if forward else values[::-1].copy()


def _port_tangents(graph: Any, port: tuple[str, bool], *, arriving: bool) -> list[np.ndarray]:
    """Fit tangents at local-radius arclengths instead of vertex indices."""
    segment_id, forward = port
    points = np.asarray(graph.points(segment_id, forward), dtype=float)
    if len(points) < 2:
        return [np.zeros(2)] * 3
    radii = _radii(graph, segment_id, forward, len(points))
    if arriving:
        points, radii = points[::-1].copy(), radii[::-1].copy()
    lengths = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))))
    radius = max(1.0, float(np.median(radii[: min(3, len(radii))])))
    tangents = []
    for multiplier in (1.5, 3.0, 6.0):
        target = min(float(lengths[-1]), multiplier * radius)
        index = max(1, min(int(np.searchsorted(lengths, target, side="left")), len(points) - 1))
        lo, hi = lengths[index - 1], lengths[index]
        fraction = 1.0 if hi <= lo else (target - lo) / (hi - lo)
        tangent = _unit(points[index - 1] + fraction * (points[index] - points[index - 1]) - points[0])
        # Reversing an arriving segment puts its port at index zero; negate
        # its outward ray to recover the direction of travel into the node.
        tangents.append(-tangent if arriving else tangent)
    return tangents


def _angle(a: np.ndarray, b: np.ndarray) -> float:
    if not np.linalg.norm(a) or not np.linalg.norm(b):
        return math.pi / 2
    return math.acos(float(np.clip(np.dot(a, b), -1.0, 1.0)))


def _signed_bend(tangents: list[np.ndarray]) -> float:
    """Signed port-local bend; reflecting the image flips both signs."""
    a, b = tangents[0], tangents[-1]
    if not np.linalg.norm(a) or not np.linalg.norm(b):
        return 0.0
    return float(math.atan2(a[0] * b[1] - a[1] * b[0], np.dot(a, b)) / math.pi)


def transition_cost(graph: Any, node_id: str, incoming: tuple[str, bool], outgoing: tuple[str, bool]) -> dict[str, float]:
    """Return continuous direction and curvature costs for a port transition."""
    del node_id
    before = _port_tangents(graph, incoming, arriving=True)
    after = _port_tangents(graph, outgoing, arriving=False)
    direction = float(np.mean([_angle(a, b) / math.pi for a, b in zip(before, after)]))
    # The signed estimates must agree across a continuation.  A mirror flips
    # both signs and therefore leaves this consistency penalty unchanged.
    curvature = abs(_signed_bend(before) - _signed_bend(after))
    return {"direction": direction, "curvature": curvature, "evidence": 0.0, "total": direction + 0.35 * curvature}


def enumerate_pairings(graph: Any, node_id: str) -> list[dict[str, Any]]:
    """List non-backtracking pairings; incoming ports are reversed outgoing ports."""
    outgoing = list(graph.outgoing(node_id))
    incoming = [(segment_id, not forward) for segment_id, forward in outgoing]
    result = []
    for before in incoming:
        for after in outgoing:
            if before[0] == after[0]:
                continue
            result.append({"node_id": node_id, "incoming": before, "outgoing": after,
                           "costs": transition_cost(graph, node_id, before, after)})
    return sorted(result, key=lambda item: (item["costs"]["total"], item["incoming"], item["outgoing"]))
