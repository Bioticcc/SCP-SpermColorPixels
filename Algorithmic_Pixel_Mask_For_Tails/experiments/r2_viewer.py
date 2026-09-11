"""Static HTML/SVG review pages for the experimental R2 assignment pool."""
from __future__ import annotations

import html
import json
import os
from pathlib import Path
from typing import Any, Iterable

PALETTE = ("#00b9e8", "#ee5090", "#75cf36", "#ad79ff", "#edb628", "#ee6a25")
STYLE = """body{font:15px system-ui;margin:22px;max-width:1500px;background:#f4f6f8;color:#20252b}
a{color:#1655a0}button{margin:3px;padding:6px}svg{max-width:100%;height:auto;background:#fff}
table{border-collapse:collapse;width:100%;background:#fff}td,th{border:1px solid #d7dbe0;padding:5px;text-align:left;vertical-align:top}
details{margin:10px 0;padding:9px;background:#fff}.note{padding:11px;background:#fff4d8}
pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}"""


def _esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _doc(title: str, body: str, script: str = "") -> str:
    return f'<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>{_esc(title)}</title><style>{STYLE}</style><body>{body}<script>{script}</script></body></html>'


def _metadata(item: dict[str, Any]) -> dict[str, Any]:
    value = item.get("metadata", {})
    return value if isinstance(value, dict) else {}


def _route_points(item: dict[str, Any]) -> list[Any]:
    metadata = _metadata(item)
    route = metadata.get("r1_route")
    return (route.get("points_xy") if isinstance(route, dict) else metadata.get("points_xy_ordered")) or []


def _offset(item: dict[str, Any]) -> tuple[float, float]:
    metadata = _metadata(item)
    value = metadata.get("source_roi_offset_xy") or metadata.get("roi_xyxy") or [0, 0]
    return (float(value[0]), float(value[1])) if isinstance(value, (list, tuple)) and len(value) >= 2 else (0.0, 0.0)


def _path(points: Iterable[Any], offset: tuple[float, float]) -> str:
    commands = []
    for index, point in enumerate(points):
        if isinstance(point, (list, tuple)) and len(point) >= 2:
            commands.append(f"{'M' if index == 0 else 'L'}{float(point[0])+offset[0]:.3f},{float(point[1])+offset[1]:.3f}")
    return " ".join(commands)


def _scatter_path(points: Iterable[Any], offset: tuple[float, float]) -> str:
    return " ".join(f"M{float(point[0])+offset[0]:.3f},{float(point[1])+offset[1]:.3f}h0" for point in points if isinstance(point, (list, tuple)) and len(point) >= 2)


def _selected(selection: dict[str, Any], head: str) -> str | None:
    value = selection.get(head)
    return str(value.get("id")) if isinstance(value, dict) and value.get("id") is not None else (str(value) if value is not None else None)


def _r1_rank_one(record: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """Keep every component/head pair's first non-null R1 route."""
    routes = []
    for component in record.get("components", []):
        roi = component.get("roi_xyxy", [0, 0])
        offset = list(roi[:2]) if isinstance(roi, (list, tuple)) else [0, 0]
        for head, search in component.get("searches", {}).items():
            route = next((row for row in search.get("hypotheses", []) if row.get("termination") != "null"), None)
            if route is not None:
                routes.append((str(head), {"metadata": {"source_roi_offset_xy": offset, "r1_route": route}}))
    return routes


def _complete_alternatives(assignment: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Promote a group-local counterfactual to an image-wide selection."""
    incumbent = assignment.get("selected_by_head", {})
    complete = {}
    for group in assignment.get("groups", []):
        for head, entry in group.get("assignment", {}).get("head_alternatives", {}).items():
            local = (entry.get("alternative") or {}).get("selected_by_head")
            if isinstance(local, dict):
                selection = dict(incumbent)
                selection.update(local)
                complete[str(head)] = {"entry": entry, "selected_by_head": selection, "group_id": group.get("id")}
    return complete


def _search_limits(record: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"tail_id": component.get("tail_id"), "head_id": str(head), "diagnostics": search.get("diagnostics", {})}
            for component in record.get("components", []) for head, search in component.get("searches", {}).items()
            if search.get("diagnostics", {}).get("truncated") or search.get("diagnostics", {}).get("frontier_pruned", 0)]


def write_image_viewer(directory: Path, r1_directory: Path, r1_record: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    hypotheses = [item for item in payload.get("hypotheses", []) if isinstance(item, dict)]
    by_id = {str(item["id"]): item for item in hypotheses if item.get("id") is not None}
    assignment, controls = payload.get("assignment", {}) or {}, payload.get("controls", {}) or {}
    joint = assignment.get("selected_by_head", {})
    selections = {"unary": assignment.get("independent_by_head", {}), "joint": joint,
                  "runnerup": (assignment.get("runner_up") or {}).get("selected_by_head", {})}
    alternatives = _complete_alternatives(assignment)
    selections.update({f"alt_{head}": row["selected_by_head"] for head, row in alternatives.items()})
    heads = sorted({str(item.get("head_id")) for item in hypotheses if item.get("head_id") is not None})
    colors = {head: PALETTE[index % len(PALETTE)] for index, head in enumerate(heads)}
    width, height = int(r1_record.get("width", 1)), int(r1_record.get("height", 1))
    rel_r1 = os.path.relpath(Path(r1_directory), directory).replace(os.sep, "/")
    original, baseline = f"{rel_r1}/original.jpg", f"{rel_r1}/baseline.jpg"

    geometry, ids = [], {}
    def route_use(item: dict[str, Any], head: str) -> str:
        item_id = str(item.get("id", "r1:" + _path(_route_points(item), _offset(item))))
        if item_id not in ids:
            ids[item_id] = f"route_{len(ids)}"
            metadata = _metadata(item)
            points = metadata.get("pixels_xy_unordered", []) if metadata.get("kind") == "baseline_control" else _route_points(item)
            geometry.append(f'<path id="{ids[item_id]}" d="{_scatter_path(points, _offset(item)) if metadata.get("kind") == "baseline_control" else _path(points, _offset(item))}" fill="none" stroke-linecap="round"/>')
        return f'<use href="#{ids[item_id]}" stroke="{colors.get(head, "#555")}" stroke-width="2"><title>Head {_esc(head)}</title></use>'

    layers: dict[str, list[str]] = {"r1": [], **{name: [] for name in selections}}
    for head, item in _r1_rank_one(r1_record): layers["r1"].append(route_use(item, head))
    for mode, selection in selections.items():
        for head in heads:
            item = by_id.get(_selected(selection, head) or "")
            if item is not None and (_route_points(item) or _metadata(item).get("pixels_xy_unordered")):
                layers[mode].append(route_use(item, head))
    groups = "".join(f'<g id="routes_{name}" style="display:none">{"".join(rows)}</g>' for name, rows in layers.items())
    control_pixels = [(metadata.get("pixels_xy_unordered", []), _offset(item)) for item in hypotheses
                      if (metadata := _metadata(item)).get("kind") == "baseline_control"]
    scatter = "".join(f'<path d="{_scatter_path(points, offset)}" fill="none" stroke="#00b9e8" stroke-width="1" stroke-linecap="round" opacity=".7"/>' for points, offset in control_pixels)

    comparisons = {str(row.get("head_id")): row for row in payload.get("baseline_changes", [])}
    unary, changed_rows = assignment.get("independent_by_head", {}), []
    for head in sorted(joint):
        detail = comparisons.get(str(head), {}); baseline_rows = detail.get("baseline_comparisons", [])
        exact = ", ".join(str(row.get("exact_same_source_pixels")) for row in baseline_rows) or "—"
        changed = detail.get("changed_from_unary_independent", _selected(unary, head) != _selected(joint, head))
        changed_rows.append(f'<tr><td>{_esc(head)}</td><td>{_esc(_selected(unary, head))}</td><td>{_esc(_selected(joint, head))}</td><td>{_esc(bool(changed))}</td><td>{_esc(detail.get("selected_pixel_count", "—"))}</td><td>{_esc(exact)}</td></tr>')
    alternative_rows = []
    for head, row in sorted(alternatives.items()):
        entry = row["entry"]
        alternative_rows.append(f'<tr><td>{_esc(head)}</td><td>{_esc(row["group_id"])}</td><td>{_esc(entry.get("selected_id"))}</td><td>{_esc(_selected(row["selected_by_head"], head))}</td><td>{_esc(entry.get("status"))}</td><td>{_esc(entry.get("score_margin", "unproven"))}</td><td>{_esc(json.dumps(entry.get("difference_from_incumbent_bounds"), sort_keys=True))}</td></tr>')
    search_limits = _search_limits(r1_record)
    solver_limits = [group.get("id") for group in assignment.get("groups", []) if group.get("assignment", {}).get("status") != "optimal"]
    missing = payload.get("pool_diagnostics", {}).get("approximate_missing_route_heads", [])
    buttons = "".join(f'<button onclick="show(\'alt_{_esc(head)}\',\'Constrained alternative for head {_esc(head)}\')">Alternative {_esc(head)}</button>' for head in alternatives)
    hypothesis_rows = "".join(f'<tr><td>{_esc(item.get("id"))}</td><td>{_esc(item.get("head_id"))}</td><td>{_esc(_metadata(item).get("kind"))}</td><td>{_esc(item.get("termination"))}</td><td>{_esc(item.get("cost"))}</td></tr>' for item in hypotheses)
    body = f'''<p><a href="../../index.html">All images</a> · <a href="assignment.json">Serialized R2 assignment</a> · <a href="{_esc(rel_r1)}/routes.json">Frozen R1 routes</a></p>
<h1>{_esc(r1_record.get("relative_image", directory.name))}</h1><p class="note">Experimental finite-pool R2 assignment over frozen R1 routes. Raw costs and margins are uncalibrated. R1 search limits and R2 solver limits are separate. This page makes no biological-accuracy or production-status claim.</p>
<p>{len(heads)} heads · {len(hypotheses)} pooled hypotheses · {len(layers["r1"])} component/head R1 rank-1 routes · {sum(len(points) for points, _ in control_pixels)} baseline-control raster pixels.</p>
<p><button onclick="show('original','Original')">Original</button><button onclick="show('baseline','Frozen baseline preview')">Frozen baseline</button><button onclick="show('controls','Baseline-control scatter')">Baseline controls</button><button onclick="show('r1','R1 rank-1 by component/head')">R1 rank-1</button><button onclick="show('unary','R2 unary independent')">Unary independent</button><button onclick="show('joint','R2 joint assignment')">Joint assignment</button><button onclick="show('runnerup','Joint runner-up')">Runner-up</button>{buttons}</p><p id="label">Original</p>
<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg"><defs>{''.join(geometry)}</defs><image id="source" href="{_esc(original)}" width="{width}" height="{height}"/><g id="baseline_controls" style="display:none">{scatter}</g>{groups}</svg>
<h2>Selections and exact retained-baseline comparisons</h2><table><tr><th>Head</th><th>Unary choice</th><th>Joint choice</th><th>Changed by joint assignment</th><th>Selected pixels</th><th>Exact same retained baseline pixels</th></tr>{''.join(changed_rows) or '<tr><td colspan="6">No joint selections supplied.</td></tr>'}</table>
<h2>Per-head constrained complete alternatives</h2><table><tr><th>Head</th><th>Conflict group</th><th>Joint choice</th><th>Constrained choice</th><th>Solver status</th><th>Raw margin</th><th>Limited-run bounds</th></tr>{''.join(alternative_rows) or '<tr><td colspan="7">No constrained alternatives supplied.</td></tr>'}</table>
<h2>Limits and missing-route diagnostics</h2><p>R1 route searches with limits: {len(search_limits)}. R2 solver groups with limits: {_esc(', '.join(str(item) for item in solver_limits) or 'none')}. Approximate missing-route diagnostics: {len(missing)}.</p><details><summary>R1 search-limit diagnostics</summary><pre>{_esc(json.dumps(search_limits, indent=2, sort_keys=True))}</pre></details><details><summary>R2 solver and pool diagnostics</summary><pre>{_esc(json.dumps({'assignment': assignment, 'pool_diagnostics': payload.get('pool_diagnostics', {}), 'missing_routes': missing, 'controls': controls}, indent=2, sort_keys=True))}</pre></details>'''
    body += f'''<h2>Hypothesis pool</h2><details><summary>{len(hypotheses)} pooled choices</summary><table><tr><th>ID</th><th>Head</th><th>Kind</th><th>Termination</th><th>Raw cost</th></tr>{hypothesis_rows}</table></details>'''
    script = """function show(mode,label){document.querySelectorAll('[id^=routes_],#baseline_controls').forEach(g=>g.style.display='none');let target=mode==='controls'?document.getElementById('baseline_controls'):document.getElementById('routes_'+mode);if(target)target.style.display='';document.getElementById('source').setAttribute('href',mode==='baseline'?%s:%s);document.getElementById('label').textContent=label;}""" % (json.dumps(baseline), json.dumps(original))
    (directory / "index.html").write_text(_doc(str(r1_record.get("relative_image", directory.name)), body, script), encoding="utf-8")
    return {"relative_image": r1_record.get("relative_image", directory.name), "page": f"images/{directory.name}/index.html", "heads": len(heads), "hypotheses": len(hypotheses), "r1_rank1_routes": len(layers["r1"]), "baseline_control_pixels": sum(len(points) for points, _ in control_pixels)}


def write_index(output: Path, summaries: list[dict[str, Any]], fixtures: dict[str, Any], demo_manifest: dict[str, Any]) -> None:
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    fixed = {row.get("relative_image"): row.get("reason", "") for row in demo_manifest.get("images", [])}
    rows = "".join(f'<tr><td><a href="{_esc(row.get("page"))}">{_esc(row.get("relative_image"))}</a></td><td>{_esc(fixed.get(row.get("relative_image"), "Full inventory"))}</td><td>{_esc(row.get("heads", 0))}</td><td>{_esc(row.get("hypotheses", 0))}</td><td>{_esc(row.get("r1_rank1_routes", 0))}</td><td>{_esc(row.get("changed_from_unary", 0))}</td><td>{_esc(row.get("limited_groups", 0))}</td></tr>' for row in sorted(summaries, key=lambda item: (item.get("relative_image") not in fixed, str(item.get("relative_image")))))
    cases = fixtures.get("cases", []) if isinstance(fixtures, dict) else []
    names = sorted({str(case.get("name", "")).removesuffix("_rendered") for case in cases})
    fixture_rows = "".join(f'<tr><td>{_esc(name)}</td><td><a href="fixtures/{_esc(name)}/comparison.png">comparison.png</a></td><td><a href="fixtures/{_esc(name)}/rendered/comparison.png">rendered comparison.png</a></td></tr>' for name in names)
    metrics = fixtures.get("rendered_centerline_summary", {}) if isinstance(fixtures, dict) else {}
    metric_rows = "".join(f'<tr><td>{_esc(name)}</td><td>{_esc(kind)}</td><td>{_esc(json.dumps(values, sort_keys=True))}</td></tr>'
                          for name, modes in sorted(metrics.items()) for kind, values in sorted(modes.items()))
    body = f'''<h1>R2 joint assignment viewer</h1><p class="note">Experimental finite-pool joint assignment over frozen R1 evidence. Constructed-geometry metrics are engineering measurements; raw assignment values are uncalibrated and this makes no biological-accuracy or production-status claim.</p><p><a href="fixture_report.json">Fixture report</a> · <a href="config.json">Configuration</a> · <a href="provenance.json">Provenance</a> · <a href="completion.json">Run results</a> · <a href="fixed_panel_contact_sheet.png">Fixed-panel contact sheet</a></p><h2>Real-image inventory</h2><table><tr><th>Image</th><th>Panel reason</th><th>Heads</th><th>Pool hypotheses</th><th>R1 rank-1 routes</th><th>Changed heads</th><th>Limited solver groups</th></tr>{rows}</table><h2>Constructed geometry</h2><table><tr><th>Fixture</th><th>Explicit graph</th><th>Rendered mask</th></tr>{fixture_rows}</table><h2>Rendered centerline metrics</h2><table><tr><th>Fixture</th><th>Coverage</th><th>Metrics</th></tr>{metric_rows}</table>'''
    (output / "index.html").write_text(_doc("R2 joint assignment", body), encoding="utf-8")
