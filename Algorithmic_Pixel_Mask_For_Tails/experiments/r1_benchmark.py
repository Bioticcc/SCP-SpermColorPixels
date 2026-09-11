"""Synthetic R1 route benchmark, kept separate from production SCP decisions.

The explicit-graph pass tests route search without rasterization.  The rendered
pass intentionally adds thinning and anchor snapping, so failures there are
reported as representation/evidence failures rather than graph-search claims.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
from PIL import Image, ImageDraw

import algorithmic_tail_mask as atm
from experiments.benchmark import centerline_measurements
from graph_construction import build_segment_graph, from_explicit_graph
from path_hypotheses import search_hypotheses
from tests.fixtures import case_names, generate_case


_DEFAULTS = {"k": 5, "diagnostic_k": 10, "max_expansions": 20000, "fixture_seed": 0}
_TIE_TOLERANCE = 1e-6
_PALETTE = [(0, 170, 220), (235, 80, 120), (80, 170, 40), (170, 70, 230)]


def _json(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json(item) for item in value]
    return value


def _route_key(hypothesis: dict[str, Any]) -> tuple[tuple[str, ...], tuple[bool, ...], str]:
    return (tuple(hypothesis["segment_ids"]), tuple(bool(x) for x in hypothesis["directions"]), hypothesis["termination"])


def _non_null(result: dict[str, Any]) -> list[dict[str, Any]]:
    return [row for row in result["hypotheses"] if row["termination"] != "null"]


def _expected_paths(case, head_id: str) -> list[dict[str, Any]]:
    """Use only declared evaluator truth, including each indeterminate option."""
    solutions = case.truth["identity_solutions"] if not case.truth["expected_behavior"]["determinate"] else [{"paths": case.truth["true_edge_paths"]}]
    return [path for solution in solutions for path in solution["paths"] if path["node_ids"][0] == head_id]


def _route_present(hypotheses: list[dict[str, Any]], truth_paths: list[dict[str, Any]], partial: bool = False) -> tuple[bool, int | None]:
    expected = {tuple(path["edge_ids"]) for path in truth_paths}
    for rank, hypothesis in enumerate(hypotheses, 1):
        route = tuple(hypothesis["segment_ids"])
        if route in expected and (not partial or hypothesis["termination"] == "partial"):
            return True, rank
    return False, None


def _raster(points: list, shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, np.uint8)
    if len(points) >= 2:
        cv2.polylines(mask, [np.asarray(points, np.int32)], False, 1, 1, lineType=cv2.LINE_8)
    elif len(points) == 1:
        x, y = (int(points[0][0]), int(points[0][1]))
        if 0 <= x < shape[1] and 0 <= y < shape[0]:
            mask[y, x] = 1
    return mask


def _head_masks(case) -> dict[str, np.ndarray]:
    masks = {}
    for head in case.truth["heads"]:
        mask = np.zeros(case.tail_mask.shape, np.uint8)
        cv2.circle(mask, tuple(head["center_xy"]), int(head["radius_px"]), 1, -1, lineType=cv2.LINE_8)
        masks[head["id"]] = mask.astype(bool)
    return masks


def _rendered_graph(case):
    skeleton = atm.skeletonize_binary(case.tail_mask.astype(np.uint8) * 255)
    coords = np.argwhere(skeleton > 0)
    anchors: dict[str, tuple[int, int]] = {}
    anchor_diagnostics = {}
    for head_id, head_mask in _head_masks(case).items():
        _, distance, xy = atm.nearest_skeleton_anchor(coords, head_mask)
        if xy is not None:
            anchors[head_id] = (int(xy[0]), int(xy[1]))
        anchor_diagnostics[head_id] = {"xy": xy, "distance_px": distance}
    graph = build_segment_graph(skeleton, tail_mask=case.tail_mask, anchors_xy=anchors)
    return graph, skeleton, anchor_diagnostics


def _search(graph, config: dict) -> dict[str, dict[str, Any]]:
    return {head_id: {
        "k5": search_hypotheses(graph, head_id, k=config["k"], max_expansions=config["max_expansions"]),
        "k10": search_hypotheses(graph, head_id, k=config["diagnostic_k"], max_expansions=config["max_expansions"]),
    } for head_id in sorted(graph.anchors)}


def _head_rows(case, searches: dict[str, dict[str, Any]], shape: tuple[int, int], rendered: bool,
               metric_point_transform: Callable[[float, float], tuple[float, float]] | None = None,
               metric_shape: tuple[int, int] | None = None) -> list[dict[str, Any]]:
    partial_ids = set(case.truth["expected_behavior"].get("partial_instance_ids", []))
    rows = []
    for instance in case.truth["instances"]:
        head_id, instance_id = instance["head_id"], instance["id"]
        expected = _expected_paths(case, head_id)
        partial = instance_id in partial_ids
        entry: dict[str, Any] = {"instance_id": instance_id, "head_id": head_id,
                                 "expected_status": "partial" if partial else "full",
                                 "route_available": head_id in searches}
        if head_id not in searches:
            entry.update({"top_k_contains_truth": False, "truth_rank": None,
                          "failure": "missing_anchor_or_route"})
            rows.append(entry); continue
        for label in ("k5", "k10"):
            hypotheses = _non_null(searches[head_id][label])
            if not rendered:
                present, rank = _route_present(hypotheses, expected, partial=partial)
                entry[f"{label}_contains_declared_route"] = present
                entry[f"{label}_declared_route_rank"] = rank
            entry[f"{label}_termination"] = hypotheses[0]["termination"] if hypotheses else None
        top = _non_null(searches[head_id]["k5"])
        entry["top1"] = top[0] if top else None
        if rendered and top and case.truth["expected_behavior"]["determinate"]:
            transform = metric_point_transform or (lambda x, y: (x, y))
            measurement_shape = metric_shape or shape
            top_metrics = centerline_measurements(_raster([transform(*p) for p in top[0]["points_xy"]], measurement_shape), instance["centerline_xy"])
            candidate_metrics = [centerline_measurements(_raster([transform(*p) for p in h["points_xy"]], measurement_shape), instance["centerline_xy"]) for h in top]
            oracle_index = max(range(len(candidate_metrics)), key=lambda i: candidate_metrics[i]["f1"])
            entry["top1_centerline"] = top_metrics
            entry["oracle_centerline"] = candidate_metrics[oracle_index]
            entry["oracle_rank"] = oracle_index + 1
        # A T-contact's deliberately partial branch is a route-semantics check,
        # never a missing-full-tail failure.
        entry["passes_expected_semantics"] = bool(entry.get("k5_contains_declared_route")) if not rendered else None
        rows.append(entry)
    return rows


def _draw_paths(base: np.ndarray, rows: list[dict[str, Any]], field: str) -> np.ndarray:
    image = base.copy()
    for index, row in enumerate(rows):
        route = row.get(field)
        if route and route.get("points_xy"):
            cv2.polylines(image, [np.asarray(route["points_xy"], np.int32)], False, _PALETTE[index % len(_PALETTE)], 2)
    return image


def _truth_preview(case) -> np.ndarray:
    image = case.rgb.copy()
    for index, instance in enumerate(case.truth["instances"]):
        cv2.polylines(image, [np.asarray(instance["centerline_xy"], np.int32)], False, _PALETTE[index % len(_PALETTE)], 2)
    return image


def _identity_preview(case, solution: dict[str, Any]) -> np.ndarray:
    """Show one declared identity option without choosing it as recoverable truth."""
    image = case.rgb.copy()
    edges = {edge["id"]: edge for edge in case.truth["graph"]["edges"]}
    for index, path in enumerate(solution["paths"]):
        for edge_id in path["edge_ids"]:
            cv2.polylines(image, [np.asarray(edges[edge_id]["points"], np.int32)], False,
                          _PALETTE[index % len(_PALETTE)], 2)
    return image


def _alternative_preview(case, rows: list[dict[str, Any]]) -> np.ndarray:
    image = case.rgb.copy()
    color_index = 0
    for row in rows:
        for hypothesis in _non_null(row.get("_search", {}).get("k5", {"hypotheses": []}))[:2]:
            if hypothesis["points_xy"]:
                cv2.polylines(image, [np.asarray(hypothesis["points_xy"], np.int32)], False,
                              _PALETTE[color_index % len(_PALETTE)], 1)
                color_index += 1
    return image


def _baseline_preview(case, baseline_root: str | Path | None) -> np.ndarray:
    image = case.rgb.copy()
    if not baseline_root:
        return image
    root = Path(baseline_root)
    directories = [root / "review" / "fixtures" / case.name, root / "fixtures" / case.name]
    for index, _head in enumerate(case.truth["heads"], 1):
        candidate = next((directory / f"baseline_head_{index}.png" for directory in directories
                          if (directory / f"baseline_head_{index}.png").is_file()), None)
        if candidate is not None:
            mask = np.asarray(Image.open(candidate).convert("L")) > 0
            if mask.shape == image.shape[:2]:
                image[mask] = _PALETTE[(index - 1) % len(_PALETTE)]
    return image


def _write_comparison(path: Path, case, rows: list[dict[str, Any]], baseline_root: str | Path | None) -> None:
    top = _draw_paths(case.rgb, rows, "top1")
    oracle_rows = []
    for row in rows:
        choice = None
        if row.get("top1") and row.get("oracle_rank") and row.get("_search"):
            choices = _non_null(row["_search"]["k5"])
            choice = choices[row["oracle_rank"] - 1]
        oracle_rows.append({**row, "oracle": choice})
    oracle = _draw_paths(case.rgb, oracle_rows, "oracle")
    if not case.truth["expected_behavior"]["determinate"]:
        options = case.truth["identity_solutions"]
        panels = [("Source evidence", case.rgb), ("R0 baseline (if available)", _baseline_preview(case, baseline_root)),
                  (f"Identity option: {options[0]['id']}", _identity_preview(case, options[0])),
                  (f"Identity option: {options[1]['id']}", _identity_preview(case, options[1])),
                  ("Search alternatives (not identity choice)", _alternative_preview(case, rows))]
    else:
        panels = [("Source evidence", case.rgb), ("R0 baseline (if available)", _baseline_preview(case, baseline_root)), ("Declared truth", _truth_preview(case)),
                  ("Top-1 hypothesis", top), ("Top-K oracle", oracle)]
    width, height = case.rgb.shape[1], case.rgb.shape[0]
    out = Image.new("RGB", (width * len(panels), height + 26), "white")
    draw = ImageDraw.Draw(out)
    for index, (label, pixels) in enumerate(panels):
        draw.text((index * width + 3, 4), label, fill="black")
        out.paste(Image.fromarray(pixels), (index * width, 26))
    out.save(path)


def _transform_case(case, name: str):
    height, width = case.tail_mask.shape
    if name == "horizontal_flip":
        return lambda x, y: (width - 1 - x, y), lambda x, y: (width - 1 - x, y), (height, width), lambda a: cv2.flip(a, 1)
    if name == "vertical_flip":
        return lambda x, y: (x, height - 1 - y), lambda x, y: (x, height - 1 - y), (height, width), lambda a: cv2.flip(a, 0)
    if name == "rotate_180":
        return lambda x, y: (width - 1 - x, height - 1 - y), lambda x, y: (width - 1 - x, height - 1 - y), (height, width), lambda a: cv2.rotate(a, cv2.ROTATE_180)
    if name == "padded_translation":
        dx, dy = 11, 7
        return lambda x, y: (x + dx, y + dy), lambda x, y: (x - dx, y - dy), (height + 2 * dy, width + 2 * dx), lambda a: cv2.copyMakeBorder(a, dy, dy, dx, dx, cv2.BORDER_CONSTANT, value=0)
    raise ValueError(name)


def _transformed_explicit(case, transform: Callable[[float, float], tuple[float, float]], shape) -> Any:
    graph = json.loads(json.dumps(case.truth["graph"]))
    for node in graph["nodes"]:
        node["xy"] = list(transform(*node["xy"]))
    for edge in graph["edges"]:
        edge["points"] = [list(transform(*point)) for point in edge["points"]]
    return from_explicit_graph(graph, shape=shape)


def _tie_groups(result: dict[str, Any]) -> list[list[tuple[str, ...]]]:
    groups = []
    current, previous = [], None
    for item in _non_null(result):
        score = float(item["score"])
        if previous is None or abs(score - previous) <= _TIE_TOLERANCE:
            current.append(tuple(item["segment_ids"]))
        else:
            groups.append(sorted(current)); current = [tuple(item["segment_ids"])]
        previous = score
    if current:
        groups.append(sorted(current))
    return groups


def _transform_checks(case, config: dict, explicit_searches: dict[str, dict[str, Any]], rendered_rows: list[dict[str, Any]], baseline_topology: tuple[int, int]) -> dict[str, Any]:
    checks = {}
    for name in ("horizontal_flip", "vertical_flip", "rotate_180", "padded_translation"):
        transform, inverse, shape, raster_transform = _transform_case(case, name)
        explicit = _search(_transformed_explicit(case, transform, shape), config)
        exact = {}
        for head in explicit_searches:
            before, after = _non_null(explicit_searches[head]["k5"]), _non_null(explicit[head]["k5"])
            exact[head] = {"rank_ties_preserved": _tie_groups(explicit[head]["k5"]) == _tie_groups(explicit_searches[head]["k5"]),
                           "costs_within_1e6": len(before) == len(after) and all(abs(float(a["score"]) - float(b["score"])) <= _TIE_TOLERANCE for a, b in zip(before, after))}
        tail = raster_transform(case.tail_mask.astype(np.uint8)).astype(bool)
        skeleton = atm.skeletonize_binary(tail.astype(np.uint8) * 255)
        coords = np.argwhere(skeleton > 0)
        anchors = {}
        for head_id, mask in _head_masks(case).items():
            _, _, xy = atm.nearest_skeleton_anchor(coords, raster_transform(mask.astype(np.uint8)) > 0)
            if xy is not None:
                anchors[head_id] = tuple(xy)
        rendered_graph = build_segment_graph(skeleton, tail_mask=tail, anchors_xy=anchors)
        rendered = _search(rendered_graph, config)
        transformed_rows = _head_rows(case, rendered, shape, rendered=True, metric_point_transform=inverse,
                                      metric_shape=case.tail_mask.shape)
        baseline_by_instance = {row["instance_id"]: row for row in rendered_rows}
        sensitivity = []
        for row in transformed_rows:
            before, after = baseline_by_instance.get(row["instance_id"], {}), row.get("top1_centerline")
            sensitivity.append({"instance_id": row["instance_id"], "top1_available": row["top1"] is not None,
                                "baseline_f1": before.get("top1_centerline", {}).get("f1"),
                                "transformed_f1": after.get("f1") if after else None})
        checks[name] = {"explicit": exact,
                        "rendered_topology": {"nodes": len(rendered_graph.nodes), "segments": len(rendered_graph.segments),
                                              "baseline_nodes": baseline_topology[0], "baseline_segments": baseline_topology[1],
                                              "matches_baseline": (len(rendered_graph.nodes), len(rendered_graph.segments)) == baseline_topology},
                        "rendered_centerline_2px": sensitivity}
    return checks


def _baseline_metrics(case, baseline_root: str | Path | None) -> dict[str, Any] | None:
    if not baseline_root:
        return None
    root = Path(baseline_root)
    report = root / "review" / "fixture_report.json"
    if not report.is_file():
        report = root / "fixture_report.json"
    if not report.is_file():
        return {"available": False, "reason": f"missing {report}"}
    rows = next((row for row in json.loads(report.read_text()).get("cases", []) if row.get("name") == case.name), None)
    return {"available": rows is not None, "conditional_assignment": rows.get("conditional_assignment", []) if rows else []}


def _identity_solution_availability(case, searches: dict[str, dict[str, Any]]) -> list[dict[str, Any]] | None:
    if case.truth["expected_behavior"]["determinate"]:
        return None
    output = []
    for solution in case.truth["identity_solutions"]:
        paths = []
        for declared in solution["paths"]:
            head_id = declared["node_ids"][0]
            hypotheses = _non_null(searches.get(head_id, {}).get("k5", {"hypotheses": []}))
            present, rank = _route_present(hypotheses, [declared])
            paths.append({"instance_id": declared["instance_id"], "head_id": head_id,
                          "route_available": present, "rank": rank})
        output.append({"solution_id": solution["id"], "all_routes_available": all(row["route_available"] for row in paths), "paths": paths})
    return output


def run_r1_fixtures(output_root: Path, config: dict) -> dict:
    """Write deterministic R1 fixture artifacts and return their report.

    ``baseline_run`` is optional and only imports already-recorded R0 fixture
    measurements; it never reruns or modifies the baseline.
    """
    settings = {**_DEFAULTS, **config}
    output_root = Path(output_root)
    fixtures_root = output_root / "fixtures"
    output_root.mkdir(parents=True, exist_ok=True)
    if fixtures_root.exists():
        raise FileExistsError(f"Refusing to overwrite fixture artifacts: {fixtures_root}")
    fixtures_root.mkdir()
    reports = []
    for name in case_names():
        case = generate_case(name, seed=int(settings["fixture_seed"]))
        directory = fixtures_root / name
        directory.mkdir(parents=True, exist_ok=True)
        Image.fromarray(case.rgb).save(directory / "source.png")
        Image.fromarray(case.tail_mask.astype(np.uint8) * 255).save(directory / "tail_evidence.png")
        case.export_truth(directory / "truth.json")
        explicit_graph = from_explicit_graph(case.truth["graph"], shape=case.tail_mask.shape)
        explicit_searches = _search(explicit_graph, settings)
        rendered_graph, skeleton, anchor_diagnostics = _rendered_graph(case)
        rendered_searches = _search(rendered_graph, settings)
        explicit_rows = _head_rows(case, explicit_searches, case.tail_mask.shape, rendered=False)
        rendered_rows = _head_rows(case, rendered_searches, case.tail_mask.shape, rendered=True)
        for row in rendered_rows:
            row["_search"] = rendered_searches[row["head_id"]]
        _write_comparison(directory / "comparison.png", case, rendered_rows, settings.get("baseline_run"))
        for row in rendered_rows:
            row.pop("_search", None)
        graph_json = {"explicit": explicit_graph.to_dict(), "rendered": rendered_graph.to_dict(),
                      "rendered_anchor_diagnostics": anchor_diagnostics, "rendered_skeleton_pixels": int(np.count_nonzero(skeleton))}
        search_json = {"explicit": explicit_searches, "rendered": rendered_searches}
        (directory / "graph.json").write_text(json.dumps(_json(graph_json), indent=2, sort_keys=True) + "\n")
        (directory / "search.json").write_text(json.dumps(_json(search_json), indent=2, sort_keys=True) + "\n")
        transforms = _transform_checks(case, settings, explicit_searches, rendered_rows,
                                       (len(rendered_graph.nodes), len(rendered_graph.segments)))
        reports.append({"name": name, "expected_behavior": case.truth["expected_behavior"],
                        "explicit_graph": explicit_rows, "rendered": rendered_rows,
                        "indeterminate_solution_availability": _identity_solution_availability(case, explicit_searches),
                        "baseline_r0": _baseline_metrics(case, settings.get("baseline_run")),
                        "transformations": transforms})
    report = {"schema_version": "r1.fixture-benchmark.v1", "config": _json(settings),
              "claims": "Synthetic engineering measurements only; rendered failures are reported, not forced to pass.",
              "cases": reports}
    (output_root / "fixture_report.json").write_text(json.dumps(_json(report), indent=2, sort_keys=True) + "\n")
    return report
