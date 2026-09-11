"""R4a attachment alternatives on unchanged frozen R1 graph geometry."""
from __future__ import annotations

import copy
from dataclasses import asdict
import numpy as np

from graph_construction import from_explicit_graph
from head_anchors import extend_attachment_routes
from experiments.r2_candidates import build_candidate_pool
from experiments.r2_assignment import assign_pool


def _length(route):
    points = np.asarray(route.get('points_xy', []), float)
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum()) if len(points) > 1 else 0.0


def extend_image(record: dict, config: dict) -> tuple[dict, list[dict]]:
    extended = copy.deepcopy(record)
    diagnostics = []
    for component in extended['components']:
        graph = from_explicit_graph(component['graph'], anchors=component['graph']['anchors'])
        for head, baseline in list(component['searches'].items()):
            result = extend_attachment_routes(graph, head, baseline,
                additional_k=config['additional_k'], max_expansions=config['search_max_expansions'])
            old_ids = {route['id'] for route in baseline['hypotheses']}
            added = [route for route in result['hypotheses'] if route['id'] not in old_ids]
            extra = result['attachment_extension']
            diagnostics.append({'tail_id': str(component['tail_id']), 'head_id': head,
                'added_count': len(added), 'added_route_ids': [route['id'] for route in added],
                'max_old_arclength': max(map(_length, baseline['hypotheses']), default=0),
                'max_new_arclength': max(map(_length, result['hypotheses']), default=0),
                'passed_head_ids': sorted({key for route in added for key in route['metadata']['passed_foreign_head_ids']}),
                'search_truncated': bool((extra.get('extension_search') or {}).get('truncated')),
                'extension_search': extra.get('extension_search'),
                'baseline_search_truncated': bool(baseline['diagnostics'].get('truncated'))})
            component['searches'][head] = result
    extended['scope'] = 'R4a optional foreign attachment continuations; original graph geometry and all R1 hypotheses retained.'
    return extended, diagnostics


def resolve_image(record: dict, frozen_r2: dict, config: dict) -> tuple[dict, dict]:
    if record['relative_image'] != frozen_r2['relative_image']:
        raise ValueError('R1/R2 source identities differ')
    extended, diagnostics = extend_image(record, config)
    pool = build_candidate_pool(extended, config)
    assignment = assign_pool(pool['hypotheses'], max_expansions=config['solver_max_expansions'])
    hypotheses = [asdict(item) for item in pool['hypotheses']]
    old = frozen_r2['assignment']['selected_by_head']
    new = assignment['selected_by_head']
    if set(old) != set(new):
        raise ValueError('R4a unexpectedly changed head inventory')
    by_id = {item['id']: item for item in hypotheses}
    selected_passes = {head: by_id[key]['metadata'].get('r1_route', {}).get('metadata', {}).get('passed_foreign_head_ids', [])
                       for head, key in new.items()}
    payload = {'schema_version': 'scp.r4a.assignment.v1', 'relative_image': record['relative_image'],
        'width': record['width'], 'height': record['height'], 'hypotheses': hypotheses, 'assignment': assignment,
        'controls': pool['controls'], 'pool_diagnostics': pool['pool_diagnostics'],
        'frozen_r2': {'hypotheses': frozen_r2['hypotheses'], 'assignment': frozen_r2['assignment']},
        'extension_diagnostics': diagnostics, 'baseline_candidates': record['baseline_candidates'],
        'comparison': {'changed_head_ids': [head for head in sorted(new) if old[head] != new[head]],
            'selected_passed_heads': selected_passes, 'production_statuses_changed': False,
            'interpretation': 'Changed finite-pool proposals; no confirmed biological corrections. Existing R2 costs normalize coverage against the expanded pool.'}}
    return extended, payload
