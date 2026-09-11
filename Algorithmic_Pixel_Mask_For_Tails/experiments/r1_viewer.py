"""Local SVG/HTML route comparisons with no server or network dependencies."""
from __future__ import annotations

import html
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

PALETTE = ['#00b9e8', '#ee5090', '#75cf36', '#ad79ff', '#edb628', '#ee6a25']
STYLE = '''body{font:16px system-ui;margin:24px;max-width:1400px;background:#f5f6f8;color:#20252b}
a{color:#1655a0}button{margin:4px;padding:7px}svg,img{max-width:100%;height:auto;background:white}
table{border-collapse:collapse;width:100%;background:white}td,th{border:1px solid #ddd;padding:6px;text-align:left}
details{margin:12px 0;padding:10px;background:white}pre{white-space:pre-wrap;overflow-wrap:anywhere}
.note{padding:12px;background:#fff4d8}.limited{color:#9c360a}h1{font-size:26px}h2{font-size:21px}'''


def _escape(value):
    return html.escape(str(value), quote=True)


def _document(title, body, script=''):
    return f'<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>{_escape(title)}</title><style>{STYLE}</style><body>{body}<script>{script}</script></body></html>'


def _nonnull(search):
    return [h for h in search['hypotheses'] if h['termination'] != 'null']


def _limited(search):
    diagnostic = search.get('diagnostics', {})
    return bool(diagnostic.get('truncated') or diagnostic.get('frontier_pruned', 0))


def write_image_viewer(directory: Path, source: dict, result: dict, config: dict) -> dict:
    rgb = Image.open(source['image']).convert('RGB')
    rgb.thumbnail((1400, 1400))
    rgb.save(directory/'original.jpg', quality=92)
    preview = np.array(rgb)
    height, width = result['height'], result['width']
    sx, sy = preview.shape[1]/width, preview.shape[0]/height
    baseline_pixels = np.zeros(preview.shape[:2], np.uint8)
    ranks = [[] for _ in range(config['k'])]
    nodes, tables = [], []
    baseline_count = heads = limited = 0
    route_count = 0
    for component in result['components']:
        x0, y0, _, _ = component['roi_xyxy']
        for old in component['baseline']:
            baseline_count += 1
            points = np.asarray(old['pixels_xy'], float).reshape(-1, 2)
            if len(points):
                xx = np.clip(np.round((points[:, 0]+x0)*sx).astype(int), 0, preview.shape[1]-1)
                yy = np.clip(np.round((points[:, 1]+y0)*sy).astype(int), 0, preview.shape[0]-1)
                baseline_pixels[yy, xx] = 255
        for node in component['graph']['nodes']:
            if node['kind'] != 'junction':
                continue
            x, y = node['xy']
            nodes.append(f'<circle cx="{x+x0}" cy="{y+y0}" r="{4/sx}" fill="none" stroke="#e00010" stroke-width="{1/sx}"><title>Tail {component["tail_id"]}: {_escape(node["id"])}</title></circle>')
        component_tables = []
        for head_id, search in component['searches'].items():
            color = PALETTE[heads % len(PALETTE)]
            heads += 1
            limited += int(_limited(search))
            routes = _nonnull(search)
            route_count += len(routes)
            table_rows = []
            for rank, hypothesis in enumerate(routes):
                points = np.asarray(hypothesis['points_xy'], float).reshape(-1, 2)
                coords = ' '.join(f'{x+x0:.3f},{y+y0:.3f}' for x, y in points)
                line_id = f't{component["tail_id"]}h{head_id}r{rank}'
                if rank < len(ranks):
                    ranks[rank].append(f'<polyline id="{line_id}" points="{coords}" fill="none" stroke="{color}" stroke-width="{2/sx}" stroke-linecap="round"><title>Tail {component["tail_id"]}, head {head_id}, route {rank+1}: {_escape(hypothesis["termination"])}; cost {hypothesis["score"]:.4f}</title></polyline>')
                table_rows.append(f'<tr><td><button onclick="showOne(\'{line_id}\')">Show {rank+1}</button></td><td>{_escape(hypothesis["termination"])}</td><td>{hypothesis["score"]:.5f}</td><td>{_escape(json.dumps(hypothesis["costs"], sort_keys=True))}</td><td>{_escape(", ".join(hypothesis["segment_ids"]))}</td></tr>')
            diagnostic = _escape(json.dumps(search.get('diagnostics', {}), sort_keys=True))
            component_tables.append(f'<details><summary>Head {_escape(head_id)} — {len(routes)} non-null alternatives; null is also available</summary><p>{diagnostic}</p><table><tr><th>Route</th><th>Termination</th><th>Cost (lower better)</th><th>Cost terms</th><th>Segments</th></tr>{"".join(table_rows)}</table></details>')
        # Pairings are a property of the graph and are shared by its heads.
        first_search = next(iter(component['searches'].values()), {})
        pairings = first_search.get('junction_pairings', [])
        pairing_rows = ''.join(f'<tr><td>{_escape(p.get("node_id"))}</td><td>{_escape(p.get("incoming"))}</td><td>{_escape(p.get("outgoing"))}</td><td>{_escape(json.dumps(p.get("costs", {}), sort_keys=True))}</td></tr>' for p in pairings)
        if pairings:
            component_tables.append(f'<details><summary>{len(pairings)} directed junction pairings</summary><table><tr><th>Junction</th><th>Incoming</th><th>Outgoing</th><th>Costs</th></tr>{pairing_rows}</table></details>')
        tables.append(f'<details><summary>Tail component {component["tail_id"]}: {len(component["graph"]["segments"])} segments, {len(component["searches"])} anchored heads</summary><p>Graph diagnostics: {_escape(json.dumps(component["graph"]["diagnostics"], sort_keys=True))}</p>{"".join(component_tables)}</details>')
    preview[cv2.dilate(baseline_pixels, np.ones((3, 3), np.uint8)) > 0] = (0, 185, 232)
    Image.fromarray(preview).save(directory/'baseline.jpg', quality=92)
    line_groups = ''.join(f'<g id="rank{i}" class="rank" style="display:none">{"".join(lines)}</g>' for i, lines in enumerate(ranks))
    controls = '<button onclick="showRank(-1)">Original</button><button onclick="showRank(-2)">Baseline</button>'
    controls += ''.join(f'<button onclick="showRank({i})">Independent route {i+1}</button>' for i in range(config['k']))
    controls += '<button onclick="toggleJunctions()">Junctions</button>'
    body = f'''<p><a href="../../index.html">All images</a> · <a href="routes.json">Complete route records, K={config['diagnostic_k']} diagnostics</a></p>
<h1>{_escape(result['relative_image'])}</h1>
<p class="note">Experimental alternatives for each head. These independent routes may conflict with one another; R1 does not perform joint assignment. Costs are uncalibrated. Every saved baseline candidate and its original status remain unchanged.</p>
<p>{heads} anchored searches; {limited} searches with reported limits; {route_count} non-null alternatives; {baseline_count} retained baseline candidates.</p>
{controls}<p id="displayLabel">Baseline</p><svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" aria-label="Sperm paths in original image coordinates">
<image id="sourceImage" href="baseline.jpg" width="{width}" height="{height}"/>{line_groups}<g id="selected"></g><g id="junctions" style="display:none">{''.join(nodes)}</g></svg>
<p>The image preview is resized for display; path and junction coordinates remain in source-image pixels. Open a head below to inspect its individual alternatives and decomposed costs.</p>{''.join(tables)}'''
    script = '''function showRank(rank){document.getElementById('selected').replaceChildren();document.querySelectorAll('.rank').forEach((g,i)=>g.style.display=(i===rank?'':'none'));document.getElementById('sourceImage').setAttribute('href',rank===-2?'baseline.jpg':'original.jpg');document.getElementById('displayLabel').textContent=rank===-2?'Baseline':rank===-1?'Original':'Independent route '+(rank+1)+' per head';}
function showOne(id){showRank(-1);let line=document.getElementById(id);if(line){let copy=line.cloneNode(true);copy.removeAttribute('id');document.getElementById('selected').appendChild(copy);document.getElementById('displayLabel').textContent=line.textContent;}}
function toggleJunctions(){let g=document.getElementById('junctions');g.style.display=g.style.display==='none'?'':'none';}'''
    (directory/'index.html').write_text(_document(result['relative_image'], body, script))
    return {'relative_image': result['relative_image'], 'page': 'images/'+directory.name+'/index.html',
            'components': len(result['components']), 'heads': heads, 'limited_searches': limited,
            'nonnull_routes': route_count, 'retained_baseline_candidates': baseline_count}


def write_index(output: Path, summaries: list[dict], fixtures: dict, demo: dict) -> None:
    fixed = {row['relative_image']: row['reason'] for row in demo['images']}
    rows = ''.join(f'<tr><td><a href="{_escape(row["page"])}">{_escape(row["relative_image"])}</a></td><td>{_escape(fixed.get(row["relative_image"], "Full-inventory regression"))}</td><td>{row["heads"]}</td><td>{row["nonnull_routes"]}</td><td>{row["limited_searches"]}</td></tr>' for row in sorted(summaries, key=lambda r: (r['relative_image'] not in fixed, r['relative_image'])))
    fixture_panels = ''.join(f'<details open><summary>{_escape(path.parent.name)}</summary><img src="{path.relative_to(output).as_posix()}" alt="{_escape(path.parent.name)} constructed geometry comparison"></details>' for path in sorted((output/'fixtures').glob('*/comparison.png')))
    body = f'''<h1>R1: direction-aware tail alternatives</h1><p class="note">Route-search demonstration using existing Ward images and constructed geometry. No new annotations or model training. Real-image accuracy is not measured. Production predictions remain frozen; alternatives are experimental and independently generated per head.</p>
<p><a href="fixture_report.json">Synthetic metrics and transformation checks</a> · <a href="config.json">Configuration</a> · <a href="provenance.json">Provenance</a> · <a href="integrity.json">Integrity</a> · <a href="completion.json">Run results</a></p>
<h2>{len(summaries)} real images</h2><p>The fixed six-image panel appears first. All processed images, including limited searches and failures, remain available.</p><table><tr><th>Image</th><th>Selection reason</th><th>Anchored searches</th><th>Alternatives</th><th>Limited searches</th></tr>{rows}</table><h2>Constructed geometry</h2><p>Truth-assisted oracle panels show whether a route was available, not whether automatic ranking selected it. Consult the metrics for graph-only versus rendered-mask results and deliberately indeterminate identities.</p>{fixture_panels}'''
    (output/'index.html').write_text(_document('R1 tail alternatives', body))
