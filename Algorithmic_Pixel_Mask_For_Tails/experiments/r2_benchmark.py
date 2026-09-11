"""Bounded R2 joint-assignment benchmark on constructed SCP geometry only."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw

from experiments.benchmark import centerline_measurements
from experiments.r1_benchmark import _rendered_graph
from experiments.r2_candidates import build_graph_pool
from global_assignment import Hypothesis, solve_with_margins
from graph_construction import from_explicit_graph
from path_hypotheses import search_hypotheses
from tests.fixtures import case_names, generate_case
from tests.fixtures.r2_competing import generate_competing_case


_DEFAULTS = {
    "k": 5,
    "search_max_expansions": 20000,
    "solver_max_expansions": 100000,
    "fixture_seed": 0,
    "null_cost": 1.5,
    "partial_cost": 0.85,
    "coverage_weight": 1.0,
}
_PALETTE = [(0, 170, 220), (235, 80, 120), (80, 170, 40), (170, 70, 230)]


def _json(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Hypothesis):
        return asdict(value)
    if isinstance(value, dict):
        return {str(key): _json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json(item) for item in value]
    return value


def _search(graph: Any, settings: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(head_id): search_hypotheses(graph, head_id, k=int(settings["k"]), max_expansions=int(settings["search_max_expansions"]))
        for head_id in sorted(graph.anchors)
    }


def _nonnull(search: dict[str, Any]) -> list[dict[str, Any]]:
    return [item for item in search["hypotheses"] if item["termination"] != "null"]


def _unary_choices(pool: list[Hypothesis]) -> dict[str, str]:
    heads = sorted({item.head_id for item in pool})
    return {
        head: min((item for item in pool if item.head_id == head), key=lambda item: (item.cost, item.id)).id
        for head in heads
    }


def _conflicts(selected: dict[str, str], by_id: dict[str, Hypothesis]) -> dict[str, list[str]]:
    segments: dict[str, list[str]] = {}
    endpoints: dict[str, list[str]] = {}
    for head_id, hypothesis_id in selected.items():
        item = by_id[hypothesis_id]
        for segment in item.exclusive_segments:
            segments.setdefault(segment, []).append(head_id)
        if item.endpoint_id is not None:
            endpoints.setdefault(item.endpoint_id, []).append(head_id)
    return {
        "ordinary_segments": sorted(key for key, heads in segments.items() if len(heads) > 1),
        "endpoints": sorted(key for key, heads in endpoints.items() if len(heads) > 1),
    }


def _route_for_choice(hypothesis_id: str | None, by_id: dict[str, Hypothesis]) -> dict[str, Any] | None:
    if hypothesis_id is None:
        return None
    item = by_id[hypothesis_id]
    return item.metadata.get("r1_route") if item.metadata.get("kind") == "r1_route" else None


def _raster(points: list[list[float]], shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, np.uint8)
    if len(points) > 1:
        cv2.polylines(mask, [np.asarray(points, np.int32)], False, 1, 1, lineType=cv2.LINE_8)
    elif points:
        x, y = map(int, points[0])
        if 0 <= x < shape[1] and 0 <= y < shape[0]:
            mask[y, x] = 1
    return mask


def _selected_metrics(case: Any, selected: dict[str, str], by_id: dict[str, Hypothesis]) -> list[dict[str, Any]] | None:
    if not case.truth["expected_behavior"]["determinate"]:
        return None
    partial = set(case.truth["expected_behavior"].get("partial_instance_ids", ()))
    rows = []
    for instance in case.truth["instances"]:
        hypothesis_id = selected.get(instance["head_id"])
        route = _route_for_choice(hypothesis_id, by_id)
        points = route.get("points_xy", []) if route is not None else []
        # An explicit null or missing proposal is an empty prediction.  It
        # remains visible in availability counts but contributes zero recall
        # and F1 to the all-instance measurement.
        route_available = bool(points)
        metric = centerline_measurements(_raster(points, case.tail_mask.shape), instance["centerline_xy"])
        selected_item = by_id.get(hypothesis_id) if hypothesis_id is not None else None
        rows.append({"instance_id": instance["id"], "head_id": instance["head_id"],
                     "expected_status": "partial" if instance["id"] in partial else "full",
                     "selected_hypothesis_id": hypothesis_id,
                     "selected_route_available": route_available,
                     "selected_null_or_missing": selected_item is None or selected_item.is_null,
                     "centerline": metric})
    return rows


def _metric_summary(reports: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate rendered all-instance metrics and labeled conditional means."""
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for record in reports:
        if record["mode"] != "rendered":
            continue
        for selection, rows in (record.get("selection_centerline_metrics") or {}).items():
            if rows is None:
                continue
            for row in rows:
                grouped.setdefault(selection, {}).setdefault(row["expected_status"], []).append(row)
    result: dict[str, Any] = {}
    for selection, statuses in grouped.items():
        result[selection] = {}
        for status, rows in statuses.items():
            all_metrics = [row["centerline"] for row in rows]
            available = [row["centerline"] for row in rows if row["selected_route_available"]]
            result[selection][status] = {
                "declared_instances": len(rows),
                "selected_route_available": len(available),
                "number_null_or_missing_selections": sum(bool(row["selected_null_or_missing"]) for row in rows),
                "all_instance_mean_precision": float(np.mean([item["precision"] for item in all_metrics])),
                "all_instance_mean_recall": float(np.mean([item["recall"] for item in all_metrics])),
                "all_instance_mean_f1": float(np.mean([item["f1"] for item in all_metrics])),
                "conditional_on_route_mean_precision": float(np.mean([item["precision"] for item in available])) if available else None,
                "conditional_on_route_mean_recall": float(np.mean([item["recall"] for item in available])) if available else None,
                "conditional_on_route_mean_f1": float(np.mean([item["f1"] for item in available])) if available else None,
            }
    return result


def _draw_routes(base: np.ndarray, selected: dict[str, str], by_id: dict[str, Hypothesis]) -> np.ndarray:
    image = base.copy()
    for index, head_id in enumerate(sorted(selected)):
        route = _route_for_choice(selected[head_id], by_id)
        if route and route.get("points_xy"):
            cv2.polylines(image, [np.asarray(route["points_xy"], np.int32)], False, _PALETTE[index % len(_PALETTE)], 2)
    return image


def _truth_image(case: Any, solution: dict[str, Any] | None = None) -> np.ndarray:
    image = case.rgb.copy()
    if solution is None:
        paths = case.truth.get("true_edge_paths", ())
    else:
        paths = solution["paths"]
    edges = {edge["id"]: edge for edge in case.truth["graph"]["edges"]}
    head_indices = {head: index for index, head in enumerate(sorted(row["id"] for row in case.truth["heads"]))}
    for path in paths:
        index = head_indices[path["node_ids"][0]]
        for edge_id in path["edge_ids"]:
            cv2.polylines(image, [np.asarray(edges[edge_id]["points"], np.int32)], False, _PALETTE[index % len(_PALETTE)], 2)
    return image


def _write_panel(path: Path, case: Any, r1_rank1: dict[str, str], unary: dict[str, str], joint: dict[str, str], by_id: dict[str, Hypothesis]) -> None:
    panels: list[tuple[str, np.ndarray]] = [("Source", case.rgb)]
    if case.truth["expected_behavior"]["determinate"]:
        panels.append(("Declared geometry", _truth_image(case)))
    else:
        for option in case.truth["identity_solutions"]:
            panels.append((f"Identity option {option['id']}", _truth_image(case, option)))
    panels.extend([
        ("R1 rank-1", _draw_routes(case.rgb, r1_rank1, by_id)),
        ("R2 unary independent", _draw_routes(case.rgb, unary, by_id)),
        ("R2 joint assignment", _draw_routes(case.rgb, joint, by_id)),
    ])
    width, height = case.rgb.shape[1], case.rgb.shape[0]
    output = Image.new("RGB", (width * len(panels), height + 24), "white")
    draw = ImageDraw.Draw(output)
    for index, (label, pixels) in enumerate(panels):
        draw.text((index * width + 2, 3), label, fill="black")
        output.paste(Image.fromarray(pixels), (index * width, 24))
    output.save(path)


def _write_centerline_masks(directory: Path, case: Any, selections: dict[str, dict[str, str]], by_id: dict[str, Hypothesis]) -> None:
    for label, selected in selections.items():
        mask = np.zeros(case.tail_mask.shape, np.uint8)
        for hypothesis_id in selected.values():
            route = _route_for_choice(hypothesis_id, by_id)
            if route:
                mask |= _raster(route.get("points_xy", []), case.tail_mask.shape)
        Image.fromarray(mask * 255).save(directory / f"{label}_centerlines.png")


def _one_case(case: Any, label: str, mode: str, directory: Path, settings: dict[str, Any]) -> dict[str, Any]:
    if mode == "explicit":
        graph = from_explicit_graph(case.truth["graph"], shape=case.tail_mask.shape)
    elif mode == "rendered":
        graph, _skeleton, _anchors = _rendered_graph(case)
    else:  # pragma: no cover - internal caller only
        raise ValueError(mode)
    searches = _search(graph, settings)
    pool = build_graph_pool(graph, searches, settings, component_id=label)
    by_id = {item.id: item for item in pool}
    unary = _unary_choices(pool)
    r1_rank1 = {
        head_id: next((item.id for item in pool if item.head_id == head_id and item.metadata.get("r1_route", {}).get("id") == _nonnull(search)[0]["id"]),
                      next(item.id for item in pool if item.head_id == head_id and item.is_null))
        for head_id, search in searches.items()
    }
    joint_result = solve_with_margins(pool, max_expansions=int(settings["solver_max_expansions"]))
    joint = joint_result["solutions"][0]["selected_by_head"] if joint_result["solutions"] else {}
    selections = {"r1_rank1": r1_rank1, "independent": unary, "joint": joint}
    selection_metrics = {
        selection: _selected_metrics(case, selected, by_id)
        for selection, selected in selections.items()
    } if mode == "rendered" else None
    declared_available = None
    if mode == "explicit" and case.truth["expected_behavior"]["determinate"]:
        declared_available = {
            path["node_ids"][0]: any(
                item.metadata.get("r1_route", {}).get("segment_ids") == path["edge_ids"]
                for item in pool if item.head_id == path["node_ids"][0]
            )
            for path in case.truth["true_edge_paths"]
        }
    unmodeled_corridors = [
        segment.id for segment in graph.segments.values()
        if graph.nodes[segment.start].kind == "junction" and graph.nodes[segment.end].kind == "junction"
    ]
    _write_panel(directory / "comparison.png", case, r1_rank1, unary, joint, by_id)
    _write_centerline_masks(directory, case, selections, by_id)
    record = {
        "name": label,
        "mode": mode,
        "expected_behavior": case.truth["expected_behavior"],
        "r1_rank1": r1_rank1,
        "independent": {"selected_by_head": unary, "conflicts": _conflicts(unary, by_id), "cost": sum(by_id[item].cost for item in unary.values())},
        "joint": {"selected_by_head": joint, "conflicts": _conflicts(joint, by_id), "result": joint_result},
        "selected_centerline_metrics": selection_metrics["joint"] if selection_metrics is not None else None,
        "selection_centerline_metrics": selection_metrics,
        "declared_path_available": declared_available,
        "hypotheses": pool,
        "searches": searches,
        "capacity_note": "Ordinary segments are exclusive. Crossing-center pixels are absent from this resource model; extended overlap corridors remain unmodeled until R5.",
        "unmodeled_extended_corridor_segments": unmodeled_corridors,
        "indeterminate_note": ("Multiple declared identity options are shown without a unique correctness claim; joint selection is an uncalibrated proposal." if not case.truth["expected_behavior"]["determinate"] else None),
        "indeterminate_identity_options": case.truth["identity_solutions"] if not case.truth["expected_behavior"]["determinate"] else None,
    }
    (directory / "assignment.json").write_text(json.dumps(_json(record), indent=2, sort_keys=True) + "\n")
    return _json(record)


def run_r2_fixtures(output_root: Path, config: dict) -> dict:
    """Write R2 fixture panels/records, refusing to overwrite fixture output."""
    settings = {**_DEFAULTS, **config}
    output_root = Path(output_root)
    fixtures_root = output_root / "fixtures"
    output_root.mkdir(parents=True, exist_ok=True)
    if fixtures_root.exists() or (output_root / "fixture_report.json").exists():
        raise FileExistsError(f"Refusing to overwrite R2 fixture artifacts: {output_root}")
    fixtures_root.mkdir()
    reports = []
    for name in case_names():
        case = generate_case(name, seed=int(settings["fixture_seed"]))
        directory = fixtures_root / name
        directory.mkdir()
        Image.fromarray(case.rgb).save(directory / "source.png")
        case.export_truth(directory / "truth.json")
        reports.append(_one_case(case, name, "explicit", directory, settings))
        rendered = directory / "rendered"
        rendered.mkdir()
        reports.append(_one_case(case, f"{name}_rendered", "rendered", rendered, settings))
    for width in (1, 3, 5):
        case = generate_competing_case(seed=int(settings["fixture_seed"]), width_px=width)
        label = f"r2_competing_endpoint_width_{width}"
        directory = fixtures_root / label
        directory.mkdir()
        Image.fromarray(case.rgb).save(directory / "source.png")
        case.export_truth(directory / "truth.json")
        reports.append(_one_case(case, label, "explicit", directory, settings))
        rendered = directory / "rendered"
        rendered.mkdir()
        reports.append(_one_case(case, f"{label}_rendered", "rendered", rendered, settings))
    report = {
        "schema_version": "r2.fixture-benchmark.v1",
        "config": _json(settings),
        "claims": "Constructed geometry benchmark only. Costs and margins are uncalibrated score differences, not probabilities or biological correctness estimates.",
        "cases": reports,
        "rendered_centerline_summary": _metric_summary(reports),
    }
    (output_root / "fixture_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report
