"""Bounded deterministic top-K directed route search over a segment graph."""
from __future__ import annotations

import heapq
import math
from typing import Any

import numpy as np

try:
    from junction_transitions import enumerate_pairings, transition_cost
except ImportError:  # pragma: no cover
    from ..junction_transitions import enumerate_pairings, transition_cost


def _kind(graph: Any, node_id: str) -> str:
    node = graph.nodes[node_id]
    return getattr(node, "kind", None) or (node.get("kind", "ordinary") if isinstance(node, dict) else "ordinary")


def _junction_bridge(node: Any, start: np.ndarray, end: np.ndarray) -> list[list[float]]:
    """Find an 8-connected bridge entirely on stated node support."""
    pixels = getattr(node, "pixels_xy", ())
    if not pixels:
        return []
    as_point = lambda p: (int(round(float(p[0]))), int(round(float(p[1]))))
    source, target = as_point(start), as_point(end)
    allowed = {as_point(p) for p in pixels} | {source, target}
    pending, parent = [source], {source: None}
    while pending:
        point = pending.pop(0)
        if point == target:
            route = []
            while point is not None:
                route.append([float(point[0]), float(point[1])])
                point = parent[point]
            return route[::-1]
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                candidate = (point[0] + dx, point[1] + dy)
                if (dx or dy) and candidate in allowed and candidate not in parent:
                    parent[candidate] = point
                    pending.append(candidate)
    return []


def _points(
    graph: Any,
    segment_id: str,
    forward: bool,
    cache: dict[tuple[str, bool], np.ndarray] | None = None,
) -> np.ndarray:
    key = (segment_id, forward)
    if cache is not None and key in cache:
        return cache[key]
    value = np.asarray(graph.points(segment_id, forward), dtype=float)
    if cache is not None:
        cache[key] = value
    return value


def _bridge_support(
    graph: Any,
    node_id: str,
    incoming: tuple[str, bool],
    outgoing: tuple[str, bool],
    points_cache: dict[tuple[str, bool], np.ndarray],
    bridge_cache: dict[tuple[str, tuple[str, bool], tuple[str, bool]], tuple[bool, np.ndarray]],
) -> tuple[bool, np.ndarray]:
    """Cache the exact bridge check shared by terminal validation and joining."""
    key = (node_id, incoming, outgoing)
    cached = bridge_cache.get(key)
    if cached is not None:
        return cached
    before = _points(graph, *incoming, points_cache)
    after = _points(graph, *outgoing, points_cache)
    bridge = _junction_bridge(graph.nodes.get(node_id), before[-1], after[0]) if node_id in graph.nodes else []
    if bridge:
        result = (True, np.asarray(bridge, dtype=float))
    else:
        # Match the original list equality test, which deliberately permits an
        # exact port match without inserting a junction bridge.
        result = (bool(before[-1].tolist() == after[0].tolist()), np.empty((0, 2), dtype=float))
    bridge_cache[key] = result
    return result


def _join(
    graph: Any,
    segments: list[tuple[str, bool]],
    points_cache: dict[tuple[str, bool], np.ndarray] | None = None,
    bridge_cache: dict[tuple[str, tuple[str, bool], tuple[str, bool]], tuple[bool, np.ndarray]] | None = None,
) -> np.ndarray | None:
    points_cache = {} if points_cache is None else points_cache
    bridge_cache = {} if bridge_cache is None else bridge_cache
    arrays = [_points(graph, segment_id, forward, points_cache) for segment_id, forward in segments]
    if not arrays:
        return np.empty((0, 2), dtype=float)
    output = arrays[0].tolist()
    for index, array in enumerate(arrays[1:], 1):
        node_id = graph.destination(*segments[index - 1])
        supported, bridge = _bridge_support(
            graph, node_id, segments[index - 1], segments[index], points_cache, bridge_cache,
        )
        if not supported:
            # Concatenating these points would make downstream polyline raster
            # code fabricate an unsupported chord through the junction.
            return None
        if len(bridge):
            output.extend(bridge[1:].tolist())
        values = array.tolist()
        output.extend(values[1:] if output and values and values[0] == output[-1] else values)
    return np.asarray(output, dtype=float)


def _null(head_id: str) -> dict[str, Any]:
    return {"id": f"{head_id}:null", "head_id": head_id, "segment_ids": [], "directions": [], "node_ids": [], "points_xy": [], "termination": "null", "costs": {"direction": 0.0, "curvature": 0.0, "evidence": 0.0, "total": 0.0}, "score": 0.0}


def _record(graph: Any, head_id: str, route: tuple[tuple[str, bool], ...], nodes: tuple[str, ...], termination: str, costs: dict[str, float], points_cache: dict[tuple[str, bool], np.ndarray] | None = None, bridge_cache: dict[tuple[str, tuple[str, bool], tuple[str, bool]], tuple[bool, np.ndarray]] | None = None) -> dict[str, Any] | None:
    points = _join(graph, list(route), points_cache, bridge_cache)
    if points is None:
        return None
    evidence = costs["evidence_sum"] / costs["evidence_length"] if costs["evidence_length"] else 0.0
    detailed = {"direction": float(costs["direction"]), "curvature": float(costs["curvature"]),
                "evidence": float(evidence), "termination": float(costs["termination"])}
    detailed["total"] = float(sum(detailed.values()))
    return {"id": "pending", "head_id": head_id, "segment_ids": [item[0] for item in route], "directions": [bool(item[1]) for item in route], "node_ids": list(nodes), "points_xy": points.tolist(), "termination": termination, "costs": detailed, "score": detailed["total"]}


def _segment_length(graph: Any, segment_id: str, forward: bool, points_cache: dict[tuple[str, bool], np.ndarray] | None = None, length_cache: dict[tuple[str, bool], float] | None = None) -> float:
    key = (segment_id, forward)
    if length_cache is not None and key in length_cache:
        return length_cache[key]
    points = _points(graph, segment_id, forward, points_cache)
    value = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum()) if len(points) > 1 else 1.0
    if length_cache is not None:
        length_cache[key] = value
    return value


def search_hypotheses(graph: Any, head_id: str, k: int = 5, max_expansions: int = 20000) -> dict[str, Any]:
    """Return at most K non-null candidates, ranked among bounded explored states.

    K=5 is the normal setting and K=10 is the diagnostic setting.  If a partial
    exists, it reserves one of K slots; this preserves all canonical determinate
    fixture paths while keeping T contacts and uncertainty visible.
    """
    if head_id not in graph.anchors:
        raise KeyError(f"unknown head {head_id!r}")
    if k < 1 or max_expansions < 1:
        raise ValueError("k and max_expansions must be positive")
    for segment_id, segment in graph.segments.items():
        evidence = float(getattr(segment, "evidence", 1.0))
        if not math.isfinite(evidence) or not 0.0 <= evidence <= 1.0:
            raise ValueError(f"segment {segment_id!r} evidence must be finite and within [0, 1]")
    start, anchor_nodes = graph.anchors[head_id], set(graph.anchors.values())
    initial = {"direction": 0.0, "curvature": 0.0, "evidence_sum": 0.0, "evidence_length": 0.0, "termination": 0.0}
    frontier = [(0.0, (start, ()), start, (), (start,), frozenset(), initial, True)]
    terminal: list[dict[str, Any]] = []
    expansions = repeated = frontier_pruned = partial_stops = invalid_bridges = 0
    frontier_limit = max(64, min(4096, max_expansions))
    transition_cache: dict[tuple[str, tuple[str, bool], tuple[str, bool]], dict[str, float]] = {}
    points_cache: dict[tuple[str, bool], np.ndarray] = {}
    length_cache: dict[tuple[str, bool], float] = {}
    bridge_cache: dict[tuple[str, tuple[str, bool], tuple[str, bool]], tuple[bool, np.ndarray]] = {}

    def add_terminal(route, nodes, termination, costs, bridges_supported) -> None:
        nonlocal invalid_bridges
        if not bridges_supported:
            invalid_bridges += 1
            return
        evidence = costs["evidence_sum"] / costs["evidence_length"] if costs["evidence_length"] else 0.0
        detailed = {"direction": float(costs["direction"]), "curvature": float(costs["curvature"]),
                    "evidence": float(evidence), "termination": float(costs["termination"])}
        detailed["total"] = float(sum(detailed.values()))
        # Keep only route identifiers and scalar costs while searching.  Full
        # polylines are expensive for dense graphs and only K routes survive.
        terminal.append({"route": route, "nodes": nodes, "termination": termination,
                         "costs": detailed, "score": detailed["total"]})

    def score(costs: dict[str, float]) -> float:
        evidence = costs["evidence_sum"] / costs["evidence_length"] if costs["evidence_length"] else 0.0
        return float(costs["direction"] + costs["curvature"] + evidence + costs["termination"])
    while frontier and expansions < max_expansions:
        _, _, node, route, nodes, used, costs, bridges_supported = heapq.heappop(frontier)
        expansions += 1
        kind, foreign = _kind(graph, node), node in anchor_nodes and node != start
        if route and (foreign or node == start or kind in {"boundary", "boundary_port", "endpoint"}):
            termination = "partial" if (foreign or node == start) else ("boundary" if kind in {"boundary", "boundary_port"} else "endpoint")
            terminal_costs = dict(costs)
            terminal_costs["termination"] += 0.25 if termination == "partial" else 0.0
            add_terminal(route, nodes, termination, terminal_costs, bridges_supported)
            if foreign or node == start or kind in {"boundary", "boundary_port", "endpoint"}:
                continue
        if route and kind == "junction":
            terminal_costs = dict(costs); terminal_costs["termination"] += 0.25
            add_terminal(route, nodes, "partial", terminal_costs, bridges_supported); partial_stops += 1
        options = graph.outgoing(node)
        if not options and route:
            terminal_costs = dict(costs); terminal_costs["termination"] += 0.25
            add_terminal(route, nodes, "partial", terminal_costs, bridges_supported)
        for segment_id, forward in options:
            if segment_id in used:
                repeated += 1
                continue
            next_costs = dict(costs)
            if route:
                cache_key = (node, route[-1], (segment_id, forward))
                transition = transition_cache.get(cache_key)
                if transition is None:
                    transition = transition_cost(graph, node, route[-1], (segment_id, forward))
                    transition_cache[cache_key] = transition
                transition_supported, _ = _bridge_support(
                    graph, node, route[-1], (segment_id, forward), points_cache, bridge_cache,
                )
                next_costs["direction"] += transition["direction"]
                next_costs["curvature"] += 0.35 * transition["curvature"]
            else:
                transition_supported = True
            length = _segment_length(graph, segment_id, forward, points_cache, length_cache)
            next_costs["evidence_sum"] += length * -math.log(max(float(getattr(graph.segments[segment_id], "evidence", 1.0)), 1e-6))
            next_costs["evidence_length"] += length
            route2 = route + ((segment_id, forward),)
            destination = graph.destination(segment_id, forward)
            entry = (score(next_costs), (destination, tuple((sid, int(direction)) for sid, direction in route2)), destination, route2, nodes + (destination,), used | {segment_id}, next_costs, bridges_supported and transition_supported)
            if len(frontier) < frontier_limit:
                heapq.heappush(frontier, entry)
            else:
                worst = max(range(len(frontier)), key=lambda index: (frontier[index][0], frontier[index][1]))
                if (entry[0], entry[1]) < (frontier[worst][0], frontier[worst][1]):
                    frontier[worst] = entry; heapq.heapify(frontier)
                frontier_pruned += 1
    terminal.sort(key=lambda item: (item["score"], item["termination"], tuple(segment_id for segment_id, _ in item["route"]), tuple(forward for _, forward in item["route"])))
    unique, seen = [], set()
    for item in terminal:
        key = (tuple(segment_id for segment_id, _ in item["route"]), tuple(forward for _, forward in item["route"]), item["termination"])
        if key not in seen:
            seen.add(key); unique.append(item)
    full = [item for item in unique if item["termination"] != "partial"]
    partials = [item for item in unique if item["termination"] == "partial"]
    if k == 1:
        selected = full[:1] if full else partials[:1]
    else:
        selected_full = full[:k - 1]
        # Reserve the first partial, then use otherwise-empty K slots for
        # additional distinct partial endings (for example a returned anchor).
        selected = selected_full + partials[: k - len(selected_full)] if partials else full[:k]
    selected.sort(key=lambda item: (item["score"], item["termination"], tuple(segment_id for segment_id, _ in item["route"]), tuple(forward for _, forward in item["route"])))
    materialized = []
    for index, item in enumerate(selected, 1):
        route = _record(graph, head_id, item["route"], item["nodes"], item["termination"], {
            "direction": item["costs"]["direction"], "curvature": item["costs"]["curvature"],
            "evidence_sum": item["costs"]["evidence"], "evidence_length": 1.0,
            "termination": item["costs"]["termination"],
        }, points_cache, bridge_cache)
        if route is None:  # Defensive: terminal validation above used this same cached bridge evidence.
            raise RuntimeError("cached terminal bridge validation changed during search")
        route["id"] = f"{head_id}:h{index}"
        materialized.append(route)
    pairings = [pairing for node_id in sorted(graph.nodes) if _kind(graph, node_id) == "junction" for pairing in enumerate_pairings(graph, node_id)]
    diagnostics = {"expansions": expansions, "max_expansions": max_expansions, "frontier_limit": frontier_limit, "pruned_repeated_edge": repeated, "pruned_frontier": frontier_pruned, "partial_stops": partial_stops, "invalid_unsupported_bridges": invalid_bridges, "transition_cache_size": len(transition_cache), "terminal_count": len(terminal), "unique_terminal_count": len(unique), "returned_nonnull_count": len(selected), "truncated": bool(frontier) or bool(frontier_pruned), "search_complete": not frontier and not frontier_pruned, "ranking_scope": "top_k_among_explored_candidates"}
    return {"head_id": head_id, "hypotheses": [_null(head_id)] + materialized, "diagnostics": diagnostics, "junction_pairings": pairings}
