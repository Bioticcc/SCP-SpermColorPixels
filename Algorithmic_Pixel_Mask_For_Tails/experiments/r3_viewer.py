"""Static R3 mask-reconstruction review pages; no reconstruction occurs here."""
from __future__ import annotations

import html
import json
import os
from pathlib import Path
from typing import Any

from PIL import Image


PALETTE = ("#00b9e8", "#ee5090", "#75cf36", "#ad79ff", "#edb628", "#ee6a25")
STYLE = """body{font:15px system-ui;margin:22px;max-width:1550px;background:#f4f6f8;color:#20252b}
a{color:#1655a0}button{margin:3px;padding:6px}svg{max-width:100%;height:auto;background:#fff;border:1px solid #d7dbe0}
table{border-collapse:collapse;width:100%;background:#fff}td,th{border:1px solid #d7dbe0;padding:5px;text-align:left;vertical-align:top}
details{margin:10px 0;padding:9px;background:#fff}.note{padding:11px;background:#fff4d8}.gallery{display:flex;flex-wrap:wrap;gap:10px}.card{background:#fff;padding:8px;max-width:260px}.card img{max-width:250px;max-height:190px;display:block}.muted{color:#58616b}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}"""


def _esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _doc(title: str, body: str, script: str = "") -> str:
    return f'<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>{_esc(title)}</title><style>{STYLE}</style><body>{body}<script>{script}</script></body></html>'


def _asset(files: dict[str, Any], key: str) -> str | None:
    value = files.get(key)
    return str(value).replace(os.sep, "/") if isinstance(value, (str, Path)) else None


def _roi(instance: dict[str, Any]) -> tuple[int, int, int, int]:
    values = instance.get("roi_xyxy", [0, 0, 1, 1])
    if not isinstance(values, (list, tuple)) or len(values) < 4:
        return 0, 0, 1, 1
    x0, y0, x1, y1 = (int(value) for value in values[:4])
    return x0, y0, max(x0 + 1, x1), max(y0 + 1, y1)


def _tinted_overlay(directory: Path, instance: dict[str, Any], color: str, ordinal: int, asset_key: str) -> str:
    """Write an RGBA view of an exported binary mask or centerline for SVG."""
    mask_path = _asset(instance.get("files", {}), asset_key)
    if not mask_path:
        raise ValueError(f"R3 instance {instance.get('id')} is missing {asset_key}")
    source = directory / mask_path
    if not source.is_file():
        raise ValueError(f"R3 instance asset is missing: {source}")
    try:
        mask = Image.open(source).convert("L")
        x0, y0, x1, y1 = _roi(instance)
        if mask.size != (x1 - x0, y1 - y0):
            raise ValueError(f"R3 {asset_key} dimensions do not match {instance.get('id')} ROI")
        rgb = tuple(int(color[index:index + 2], 16) for index in (1, 3, 5))
        overlay = Image.new("RGBA", mask.size, rgb + (0,))
        overlay.putalpha(mask.point(lambda value: 150 if value else 0))
        relative = Path("overlays") / f"{ordinal:03d}_{instance.get('id', 'instance')}_{asset_key}.png"
        target = directory / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        overlay.save(target)
        return relative.as_posix()
    except OSError as error:
        raise ValueError(f"Cannot read R3 instance asset: {source}") from error


def _diagnostic_summary(components: list[dict[str, Any]], instances: list[dict[str, Any]]) -> dict[str, Any]:
    """Expose empirical reconstruction diagnostics without inventing thresholds."""
    values: dict[str, list[Any]] = {"support_retained": [], "ordinary_duplicate_pixels": [], "shared_pixels": [],
                                    "supported_centerline_pixels": [], "unsupported_centerline_pixels": [], "width_unknown_samples": [], "solver_limits": []}
    for component in components:
        for key in values:
            if key in component:
                values[key].append(component[key])
        diagnostics = component.get("core_diagnostics", {})
        if isinstance(diagnostics, dict):
            for key in values:
                if key in diagnostics:
                    values[key].append(diagnostics[key])
    values["supported_centerline_pixels"] = [row.get("measurements", {}).get("supported_centerline_pixels") for row in instances
                                               if isinstance(row.get("measurements"), dict) and row["measurements"].get("supported_centerline_pixels") is not None]
    return {key: items for key, items in values.items() if items}


def write_image_viewer(directory: Path, r1_directory: Path, r2_directory: Path, record: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    """Write a single image page from already exported R3 instance assets."""
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    instances = [row for row in payload.get("instances", []) if isinstance(row, dict)]
    width, height = int(payload.get("width", record.get("width", 1))), int(payload.get("height", record.get("height", 1)))
    rel_r1 = os.path.relpath(Path(r1_directory), directory).replace(os.sep, "/")
    rel_r2 = os.path.relpath(Path(r2_directory), directory).replace(os.sep, "/")
    colors = {str(instance.get("id")): PALETTE[index % len(PALETTE)] for index, instance in enumerate(instances)}
    overlays, centerlines, priority_layers, uncertainty_layers = [], [], [], []
    is_v2 = payload.get("schema_version") == "scp.r3.reconstruction.v2"
    for index, instance in enumerate(instances):
        files = instance.get("files")
        if not isinstance(files, dict):
            raise ValueError(f"R3 instance {instance.get('id')} lacks files")
        x0, y0, x1, y1 = _roi(instance); expected = (x1 - x0, y1 - y0)
        required = ("tail_mask", "head_mask", "instance_mask", "centerline") + (("head_evidence_mask", "ownership_uncertainty_mask", "head_priority_instance_mask") if is_v2 else ())
        for key in required:
            asset = _asset(files, key)
            if not asset or not (directory / asset).is_file():
                raise ValueError(f"R3 instance {instance.get('id')} is missing {key}")
            with Image.open(directory / asset) as binary:
                if binary.size != expected:
                    raise ValueError(f"R3 {key} dimensions do not match {instance.get('id')} ROI")
        overlay = _tinted_overlay(directory, instance, colors[str(instance.get("id"))], index, "instance_mask")
        overlays.append((instance, overlay, x0, y0, x1 - x0, y1 - y0))
        centerline = _tinted_overlay(directory, instance, colors[str(instance.get("id"))], index, "centerline")
        centerlines.append((index, instance, centerline, x0, y0, x1 - x0, y1 - y0))
        if is_v2:
            priority = _tinted_overlay(directory, instance, colors[str(instance.get("id"))], index, "head_priority_instance_mask")
            uncertainty = _tinted_overlay(directory, instance, "#f2c200", index, "ownership_uncertainty_mask")
            priority_layers.append((index, instance, priority, x0, y0, x1 - x0, y1 - y0))
            uncertainty_layers.append((index, instance, uncertainty, x0, y0, x1 - x0, y1 - y0))

    layers = "".join(f'<image class="instance-layer mask-layer layer-{index}" href="{_esc(asset)}" x="{x}" y="{y}" width="{w}" height="{h}" style="display:none"><title>{_esc(instance.get("id"))}</title></image>'
                     for index, (instance, asset, x, y, w, h) in enumerate(overlays))
    layers += "".join(f'<image class="instance-layer centerline-layer centerline-{index}" href="{_esc(asset)}" x="{x}" y="{y}" width="{w}" height="{h}" style="display:none"><title>{_esc(instance.get("id"))} selected centerline</title></image>'
                      for index, instance, asset, x, y, w, h in centerlines)
    layers += "".join(f'<image class="instance-layer head-priority-layer" href="{_esc(asset)}" x="{x}" y="{y}" width="{w}" height="{h}" style="display:none"><title>{_esc(instance.get("id"))} detected-head-priority alternative</title></image>'
                      for index, instance, asset, x, y, w, h in priority_layers)
    layers += "".join(f'<image class="instance-layer uncertainty-layer" href="{_esc(asset)}" x="{x}" y="{y}" width="{w}" height="{h}" style="display:none"><title>{_esc(instance.get("id"))} ownership uncertainty</title></image>'
                      for index, instance, asset, x, y, w, h in uncertainty_layers)
    buttons = "".join(f'<button onclick="showOne({index})">{_esc(instance.get("id"))}</button>' for index, (instance, *_rest) in enumerate(overlays))
    gallery = []
    labels = (("original_crop", "Original RGB crop"), ("tail_mask", "Tail evidence mask"), ("head_mask", "Head mask"),
              ("instance_mask", "Logical instance mask"), ("centerline", "Selected centerline"), ("highlighted_crop", "Highlighted crop"),
              ("head_evidence_mask", "Full detected head evidence"), ("ownership_uncertainty_mask", "Ownership uncertainty"),
              ("head_priority_instance_mask", "Detected-head-priority alternative"))
    for instance in instances:
        files = instance.get("files", {}) if isinstance(instance.get("files"), dict) else {}
        images = "".join(f'<p>{_esc(label)}<br><img src="{_esc(asset)}" alt="{_esc(label)} for {_esc(instance.get("id"))}"></p>'
                         for key, label in labels if (asset := _asset(files, key)))
        gallery.append(f'<details><summary>{_esc(instance.get("id"))}: head {_esc(instance.get("head_id"))}, tail {_esc(instance.get("tail_id"))}, {_esc(instance.get("termination"))}</summary><div class="gallery">{images or "No exported local assets."}</div><p><b>Uncertainty metadata</b>: {_esc(json.dumps(instance.get("uncertainty", {}), sort_keys=True))}</p><p><b>Measurements</b>: {_esc(json.dumps(instance.get("measurements", {}), sort_keys=True))}</p></details>')
    diagnostics = _diagnostic_summary([row for row in payload.get("components", []) if isinstance(row, dict)], instances)
    null_rows = "".join(f'<tr><td>{_esc(row.get("head_id"))}</td><td>{_esc(row.get("reason", row.get("termination", "null")))}</td><td>{_esc(json.dumps(row, sort_keys=True))}</td></tr>' for row in payload.get("null_choices", []) if isinstance(row, dict))
    uncertainty_rows = "".join(f'<tr><td>{_esc(row.get("id"))}</td><td>{_esc(row.get("uncertainty", {}).get("solver_status"))}</td><td>{_esc(row.get("uncertainty", {}).get("upstream_r1_truncated"))}</td><td>{_esc(row.get("uncertainty", {}).get("per_head_score_margin_uncalibrated"))}</td></tr>' for row in instances)
    component_rows = []
    for row in payload.get("components", []):
        if not isinstance(row, dict):
            continue
        crossing_links = " ".join('<a href="%s">%s</a>' % (_esc(region.get("mask_file")), _esc(region.get("id"))) for region in row.get("crossing_regions", [])) or "none"
        component_rows.append(f'<tr><td>{_esc(row.get("tail_id"))}</td><td>{_esc(row.get("source_roi_xyxy"))}</td><td><a href="{_esc(row.get("evidence_mask_file"))}">evidence mask</a></td><td>{crossing_links}</td></tr>')
    component_rows = "".join(component_rows)
    body = f'''<p><a href="../../index.html">All images</a> · <a href="{_esc(rel_r1)}/original.jpg">Original image</a> · <a href="{_esc(rel_r1)}/baseline.jpg">Frozen baseline</a> · <a href="{_esc(rel_r2)}/index.html">R2 assignment page</a> · <a href="reconstruction.json">Serialized R3 record</a></p>
<h1>{_esc(payload.get("relative_image", record.get("relative_image", directory.name)))}</h1><p class="note">Experimental R3 logical masks reconstruct selected R2 centerlines. The default representation preserves pinned tail centerline support and trims foreign head margins; full detected head evidence remains unchanged in its own mask. The detected-head-priority alternative preserves raw head evidence but can omit conflicting tail pixels. Detected-head morphology uses <code>head_evidence_mask</code>. RGB crops are unaltered.</p>
<p>{len(instances)} exported instances · {len(payload.get("null_choices", []))} null choices · {len(overlays)} displayable logical-mask overlays · {len(centerlines)} selected-centerline overlays.</p>
<p><button onclick="showOriginal()">Original</button><button onclick="showFrozen()">Frozen baseline</button><button onclick="showMasks()">Masks only</button><button onclick="showCenterlines()">Centerlines only</button><button onclick="showBoth()">Masks and centerlines</button>{'<button onclick="showHeadPriority()">Detected-head-priority alternative</button><button onclick="showUncertainty()">Ownership uncertainty</button>' if is_v2 else ''}{buttons}</p><p id="label">Original</p>
<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" aria-label="Logical R3 masks in original-image coordinates"><image id="source" href="{_esc(rel_r1)}/original.jpg" width="{width}" height="{height}"/>{layers}</svg>
<h2>Per-instance exported assets</h2>{''.join(gallery) or '<p>No non-null instances exported.</p>'}
<h2>Null and partial choices</h2><table><tr><th>Head</th><th>Reason / termination</th><th>Record</th></tr>{null_rows or '<tr><td colspan="3">No null choices supplied.</td></tr>'}</table>
<h2>Empirical reconstruction diagnostics</h2><p class="muted">These values describe exported evidence ownership; upstream R1 search and R2 solver limitations remain separate.</p><pre>{_esc(json.dumps(diagnostics, indent=2, sort_keys=True))}</pre><table><tr><th>Instance</th><th>R2 solver status</th><th>R1 search truncated</th><th>Uncalibrated margin</th></tr>{uncertainty_rows}</table><h2>Component evidence and modeled crossing regions</h2><table><tr><th>Tail</th><th>Source ROI</th><th>Evidence</th><th>Crossing masks</th></tr>{component_rows or '<tr><td colspan="4">No reconstructed components.</td></tr>'}</table><details><summary>Source comparisons and status preservation</summary><pre>{_esc(json.dumps(payload.get("source_comparisons", {}), indent=2, sort_keys=True))}</pre></details>'''
    script = """function clear(){document.querySelectorAll('.instance-layer').forEach(x=>x.style.display='none')}function image(href,label){clear();document.getElementById('source').setAttribute('href',href);document.getElementById('label').textContent=label}function showOriginal(){image(%s,'Original')}function showFrozen(){image(%s,'Frozen baseline')}function showMasks(){showOriginal();document.querySelectorAll('.mask-layer').forEach(x=>x.style.display='');document.getElementById('label').textContent='Logical masks'}function showCenterlines(){showOriginal();document.querySelectorAll('.centerline-layer').forEach(x=>x.style.display='');document.getElementById('label').textContent='Selected centerlines'}function showBoth(){showOriginal();document.querySelectorAll('.mask-layer,.centerline-layer').forEach(x=>x.style.display='');document.getElementById('label').textContent='Logical masks and centerlines'}function showHeadPriority(){showOriginal();document.querySelectorAll('.head-priority-layer').forEach(x=>x.style.display='');document.getElementById('label').textContent='Detected-head-priority alternative'}function showUncertainty(){showOriginal();document.querySelectorAll('.uncertainty-layer').forEach(x=>x.style.display='');document.getElementById('label').textContent='Ownership uncertainty'}function showOne(n){showOriginal();document.querySelectorAll('.layer-'+n+',.centerline-'+n).forEach(x=>x.style.display='');document.getElementById('label').textContent='Logical mask and selected centerline '+n}""" % (json.dumps(f"{rel_r1}/original.jpg"), json.dumps(f"{rel_r1}/baseline.jpg"))
    (directory / "index.html").write_text(_doc(str(payload.get("relative_image", directory.name)), body, script), encoding="utf-8")
    return {"relative_image": payload.get("relative_image", record.get("relative_image", directory.name)), "page": f"images/{directory.name}/index.html", "instances": len(instances), "null_choices": len(payload.get("null_choices", [])), "displayable_masks": len(overlays), "displayable_centerlines": len(centerlines)}


def write_index(output: Path, summaries: list[dict[str, Any]], fixture_report: dict[str, Any], demo_manifest: dict[str, Any]) -> None:
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    fixed = {row.get("relative_image"): row.get("reason", "") for row in demo_manifest.get("images", [])}
    rows = "".join(f'<tr><td><a href="{_esc(row.get("page"))}">{_esc(row.get("relative_image"))}</a></td><td>{_esc(fixed.get(row.get("relative_image"), "Full inventory"))}</td><td>{_esc(row.get("instances", 0))}</td><td>{_esc(row.get("null_choices", 0))}</td><td>{_esc(row.get("displayable_masks", 0))}</td></tr>' for row in sorted(summaries, key=lambda row: (row.get("relative_image") not in fixed, str(row.get("relative_image")))))
    cases = fixture_report.get("cases", []) if isinstance(fixture_report, dict) else []
    fixture_rows = "".join(f'<tr><td>{_esc(case.get("name", case.get("directory", "fixture")))}</td><td><a href="fixtures/{_esc(case.get("directory", case.get("name", "")))}/comparison.png">comparison.png</a></td></tr>' for case in cases if isinstance(case, dict))
    body = f'''<h1>R3 logical instance-mask viewer</h1><p class="note">Experimental reconstruction from selected R2 paths. It separates logical-mask ownership from unaltered source RGB crop content and makes no identity, biological-accuracy, clinical, or production-status claim.</p><p><a href="fixture_report.json">Fixture report</a> · <a href="config.json">Configuration</a> · <a href="provenance.json">Provenance</a> · <a href="completion.json">Run results</a> · <a href="fixed_panel_contact_sheet.png">Fixed-panel contact sheet</a></p><h2>Real-image inventory</h2><table><tr><th>Image</th><th>Panel reason</th><th>Instances</th><th>Null choices</th><th>Displayable masks</th></tr>{rows}</table><h2>Fixtures</h2><table><tr><th>Fixture</th><th>Comparison</th></tr>{fixture_rows}</table>'''
    (output / "index.html").write_text(_doc("R3 logical masks", body), encoding="utf-8")
