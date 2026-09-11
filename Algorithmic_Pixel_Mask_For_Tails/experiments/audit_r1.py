"""Audit completed R1 artifacts without treating synthetic scores as real accuracy."""
from __future__ import annotations

import argparse
from html.parser import HTMLParser
import json
import math
from pathlib import Path
import sys
from urllib.parse import unquote, urlsplit

SCP_ROOT = Path(__file__).resolve().parents[1]
if str(SCP_ROOT) not in sys.path:
    sys.path.insert(0, str(SCP_ROOT))

import numpy as np
from PIL import Image
from experiments.provenance import sha256_file
from experiments.run_r1 import load_image_checkpoint


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
        self.buttons = 0

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        self.buttons += tag == 'button'
        self.links.extend(attrs[key] for key in ('src', 'href') if attrs.get(key))


def audit(root: Path) -> dict:
    root = root.resolve()
    completion = json.loads((root/'completion.json').read_text())
    integrity = json.loads((root/'integrity.json').read_text())
    config = json.loads((root/'config.json').read_text())
    provenance = json.loads((root/'provenance.json').read_text())
    original_candidates = {}
    for filename in provenance['frozen_evidence_sha256']:
        path = Path(filename)
        if path.suffix == '.json' and path.parent.name == 'json':
            saved = json.loads(path.read_text())
            original_candidates.update({row['crop_id']: row for row in saved['head_connected_postprocessing']['crop_candidates_detail']})
    errors, counts = [], {'images': 0, 'components': 0, 'baseline_candidates': 0, 'hypotheses': 0,
                         'limited_searches': 0, 'local_links': 0, 'buttons': 0}
    if not completion['full_inventory_run'] or not integrity['passed']:
        errors.append('Incomplete inventory or failed final integrity')
    for path, fingerprint in provenance['frozen_evidence_sha256'].items():
        if sha256_file(Path(path)) != fingerprint:
            errors.append(f'Frozen evidence changed: {path}')
    inventory = set()
    for directory in sorted((root/'images').iterdir()):
        record = json.loads((directory/'routes.json').read_text())
        counts['images'] += 1
        if record['relative_image'] in inventory:
            errors.append('Duplicate source image')
        inventory.add(record['relative_image'])
        if load_image_checkpoint(directory, record['relative_image']) is None:
            errors.append(f'Missing image checkpoint: {directory.name}')
        old_count = 0
        for component in record['components']:
            counts['components'] += 1
            x0, y0, _, _ = component['roi_xyxy']
            graph = component['graph']
            diag = graph['diagnostics']
            if diag.get('unrepresented_skeleton_pixels', 0) or diag.get('unrepresented_skeleton_edges', 0):
                errors.append(f'Unrepresented graph evidence: {directory.name}/{component["tail_id"]}')
            supported = {tuple(point) for node in graph['nodes'] for point in node['pixels_xy']}
            supported.update(tuple(point) for segment in graph['segments'] for point in segment['points_xy'])
            for label, maximum in [('searches', config['k']), ('diagnostic_searches', config['diagnostic_k'])]:
                for head_id, search in component[label].items():
                    routes = [h for h in search['hypotheses'] if h['termination'] != 'null']
                    if len(routes) > maximum:
                        errors.append('Search exceeded configured K')
                    if label == 'searches':
                        counts['limited_searches'] += bool(search['diagnostics']['truncated'])
                        counts['hypotheses'] += len(routes)
                    for route in routes:
                        if len(set(route['segment_ids'])) != len(route['segment_ids']):
                            errors.append('Repeated ordinary segment in route')
                        if not route['points_xy'] or any(tuple(point) not in supported for point in route['points_xy']):
                            errors.append('Route contains absent or unsupported coordinates')
                        if any(max(abs(a[0]-b[0]), abs(a[1]-b[1])) > 1.001 for a, b in zip(route['points_xy'], route['points_xy'][1:])):
                            errors.append('Unsupported gap between real-image route pixels')
                        terms = sum(v for key, v in route['costs'].items() if key != 'total')
                        if not math.isfinite(route['score']) or abs(terms-route['score']) > 1e-8:
                            errors.append('Route score does not equal reported cost terms')
            for old in component['baseline']:
                old_count += 1
                mask = np.asarray(Image.open(old['skeleton_file']).convert('L')) > 0
                saved = original_candidates[old['crop_id']]
                bx0, by0, _, _ = saved['bbox_xyxy']
                yy, xx = np.nonzero(mask)
                expected_pixels = np.column_stack((xx+bx0-x0, yy+by0-y0)).tolist()
                if old['pixels_xy'] != expected_pixels or old['status'] != saved['candidate_status'] or old['head_id'] != str(saved['head_label_id']):
                    errors.append('Baseline coordinates, identity, or status changed')
        counts['baseline_candidates'] += old_count
        if old_count != record['baseline_candidates']:
            errors.append('Baseline candidate inventory changed')
    if len(inventory) != config['expected_image_count']:
        errors.append('Image count differs from required inventory')
    if inventory != {r['relative_image'] for r in provenance['input_manifest']['images']}:
        errors.append('Image identity inventory differs from source manifest')
    fixtures = json.loads((root/'fixture_report.json').read_text())
    graph_total = graph_correct = 0
    for case in fixtures['cases']:
        if case['expected_behavior']['determinate']:
            for row in case['explicit_graph']:
                graph_total += 1
                graph_correct += bool(row['k5_contains_declared_route'])
                if not row['k5_contains_declared_route']:
                    errors.append(f'Missing declared graph route: {case["name"]}/{row["head_id"]}')
        else:
            if not all(row['all_routes_available'] for row in case['indeterminate_solution_availability']):
                errors.append('Indeterminate graph alternatives lost')
    for path in root.rglob('*.html'):
        parser = Links()
        parser.feed(path.read_text())
        counts['buttons'] += parser.buttons
        for link in parser.links:
            parts = urlsplit(link)
            if not parts.scheme and parts.path:
                counts['local_links'] += 1
                if not (path.parent/unquote(parts.path)).is_file():
                    errors.append(f'Missing local viewer target: {path.name}: {link}')
    return {'passed': not errors, 'errors': sorted(set(errors)), 'counts': counts,
            'explicit_graph_routes': {'present': graph_correct, 'total': graph_total},
            'scope': 'Artifact invariants, route availability, and local asset checks; no browser interaction or real-image accuracy claim.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_root', type=Path)
    args = parser.parse_args()
    result = audit(args.run_root)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['passed'] else 1)
