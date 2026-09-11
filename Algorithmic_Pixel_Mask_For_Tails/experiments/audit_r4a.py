"""Audit R4a baseline preservation, route support, assignment and artifact integrity."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from urllib.parse import unquote,urlsplit
import numpy as np

SCP_ROOT = Path(__file__).resolve().parents[1]
if str(SCP_ROOT) not in sys.path: sys.path.insert(0,str(SCP_ROOT))
from graph_construction import from_explicit_graph
from path_hypotheses.search import _join
from experiments.provenance import sha256_file
from experiments.run_r2 import load_checkpoint
from experiments.run_r3 import tree_hashes
from experiments.audit_r3 import _Links
from experiments.r2_candidates import build_candidate_pool
from experiments.r2_assignment import assign_pool


def check_routes(old, new, additional_k):
    if old['relative_image'] != new['relative_image'] or old['width'] != new['width'] or old['height'] != new['height']:
        raise ValueError('Source image identity/dimensions changed')
    original = {str(c['tail_id']):c for c in old['components']}
    if len(new['components']) != len(original): raise ValueError('Component inventory changed')
    added_count = baseline_count = 0
    for component in new['components']:
        previous = original[str(component['tail_id'])]
        for key in previous:
            if key != 'searches' and previous[key] != component[key]:
                raise ValueError(f'Frozen component field changed: {key}')
        graph = from_explicit_graph(component['graph'],anchors=component['graph']['anchors'])
        if set(component['searches']) != set(previous['searches']): raise ValueError('Head search inventory changed')
        for head,search in component['searches'].items():
            base = previous['searches'][head]
            n = len(base['hypotheses']); baseline_count += n
            if search['hypotheses'][:n] != base['hypotheses']:
                raise ValueError('Original route hypotheses changed or were dropped')
            for key in base:
                if key != 'hypotheses' and search[key] != base[key]: raise ValueError('Original search diagnostics changed')
            extra = search['hypotheses'][n:]
            if len(extra) > additional_k: raise ValueError('Additional route budget exceeded')
            ids = [item['id'] for item in search['hypotheses']]
            if len(ids) != len(set(ids)): raise ValueError('Duplicate route IDs')
            signatures = set()
            for route in search['hypotheses']:
                signature = (tuple(route['segment_ids']),tuple(route['directions']),route['termination'])
                if signature in signatures: raise ValueError('Duplicate directed route')
                signatures.add(signature)
            for route in extra:
                if route['head_id'] != head or route['termination'] == 'null': raise ValueError('Invalid added route identity')
                segments = route['segment_ids']; directions = route['directions']; nodes = route['node_ids']
                if len(segments) != len(set(segments)) or len(segments) != len(directions) or len(nodes) != len(segments)+1:
                    raise ValueError('Invalid or repeated segment traversal')
                current = graph.anchors[head]
                if nodes[0] != current: raise ValueError('Route begins at another head')
                for index,(segment,direction) in enumerate(zip(segments,directions)):
                    if (segment,direction) not in graph.outgoing(current): raise ValueError('Unsupported directed transition')
                    current = graph.destination(segment,direction)
                    if current != nodes[index+1]: raise ValueError('Route nodes differ from segments')
                joined = _join(graph,list(zip(segments,directions)))
                if joined is None or not np.array_equal(joined,np.asarray(route['points_xy'],float)):
                    raise ValueError('Route geometry differs from supported graph segments')
                passed = sorted(key for key,node in graph.anchors.items() if key != head and node in nodes[1:-1])
                if not passed or route['metadata']['passed_foreign_head_ids'] != passed:
                    raise ValueError('Foreign attachment passage is missing or misstated')
                if nodes[-1] in {node for key,node in graph.anchors.items() if key != head} and route['termination'] != 'partial':
                    raise ValueError('Foreign head endpoint is falsely labeled complete')
            if search['attachment_extension']['added_count'] != len(extra): raise ValueError('Added count differs')
            added_count += len(extra)
    return {'added_routes':added_count,'preserved_route_records_including_null':baseline_count}


def deterministic_assignment(value):
    """Wall-clock timing is recorded separately from deterministic solver output."""
    if isinstance(value, dict):
        return {key: deterministic_assignment(item) for key,item in value.items() if key != 'runtime_seconds'}
    if isinstance(value, (list, tuple)):
        return [deterministic_assignment(item) for item in value]
    return value


def audit(root):
    root = Path(root).resolve(); errors=[]
    counts={'images':0,'added_routes':0,'preserved_route_records_including_null':0,'changed_heads':0,'local_links':0,'null_choices':0}
    try:
        completion=json.loads((root/'completion.json').read_text()); provenance=json.loads((root/'provenance.json').read_text())
        config=json.loads((root/'config.json').read_text()); integrity=json.loads((root/'integrity.json').read_text())
        if not completion['full_inventory_run'] or not integrity['passed']: errors.append('Incomplete inventory or failed integrity')
        checkpoint=json.loads((root/'fixture_completion.json').read_text())
        if checkpoint != {'files':tree_hashes(root/'fixtures'),'report_sha256':sha256_file(root/'fixture_report.json')}: errors.append('Fixture checkpoint changed')
        for name,digest in provenance['frozen_evidence_sha256'].items():
            if sha256_file(Path(name)) != digest: errors.append('Frozen evidence changed: '+name)
        for row in provenance['input_manifest']['images']:
            if sha256_file(Path(provenance['input_manifest']['input_root'])/row['relative_image']) != row['sha256']: errors.append('Raw source changed')
        r1=Path(provenance['source_runs']['r1']); r2=Path(provenance['source_runs']['r2']); identities=set()
        for directory in sorted((root/'images').iterdir()):
            payload=json.loads((directory/'assignment.json').read_text()); routes=json.loads((directory/'routes.json').read_text())
            identity=payload['relative_image']; counts['images']+=1
            if identity in identities: errors.append('Duplicate image identity')
            identities.add(identity)
            if load_checkpoint(directory,identity) is None: errors.append('Missing image checkpoint')
            oldpath=r1/'images'/directory.name/'routes.json'; r2path=r2/'images'/directory.name/'assignment.json'
            old=json.loads(oldpath.read_text()); oldr2=json.loads(r2path.read_text())
            if payload['r1_routes_sha256'] != sha256_file(oldpath) or payload['r2_assignment_sha256'] != sha256_file(r2path): errors.append('Frozen source record mismatch')
            if payload['frozen_r2'] != {key:oldr2[key] for key in ('hypotheses','assignment')}: errors.append('Frozen R2 comparison changed')
            checked=check_routes(old,routes,config['additional_k'])
            for key,value in checked.items(): counts[key]+=value
            pool=build_candidate_pool(routes,config)
            if payload['hypotheses'] != json.loads(json.dumps([asdict(item) for item in pool['hypotheses']])): errors.append('Candidate pool differs from preserved graph routes')
            assignment=assign_pool(pool['hypotheses'],max_expansions=config['solver_max_expansions'])
            if deterministic_assignment(assignment) != deterministic_assignment(payload['assignment']): errors.append('Joint assignment is not deterministic')
            before=oldr2['assignment']['selected_by_head']; after=assignment['selected_by_head']
            changed=[head for head in sorted(after) if after[head]!=before[head]]
            if changed != payload['comparison']['changed_head_ids'] or payload['comparison']['production_statuses_changed'] is not False: errors.append('Comparison inventory or status preservation differs')
            counts['changed_heads']+=len(changed)
            by={item.id:item for item in pool['hypotheses']}
            counts['null_choices']+=sum(by[key].termination=='null' for key in after.values())
        expected={row['relative_image'] for row in provenance['input_manifest']['images']}
        if identities != expected or counts['images'] != config['expected_image_count']: errors.append('Full image inventory differs')
        for page in root.rglob('*.html'):
            parser=_Links();parser.feed(page.read_text())
            for link in parser.links:
                parsed=urlsplit(link)
                if not parsed.scheme and parsed.path:
                    counts['local_links']+=1
                    if not (page.parent/unquote(parsed.path)).is_file(): errors.append('Missing local viewer target: '+link)
    except (OSError,KeyError,ValueError,TypeError) as exc:
        errors.append('Invalid or missing artifact: '+str(exc))
    return {'passed':not errors,'errors':sorted(set(errors)),'counts':counts,
        'scope':'Frozen graph/baseline preservation, supported attachment alternatives and finite-pool assignment; no biological accuracy claim.'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('run_root',type=Path)
    result=audit(parser.parse_args().run_root);print(json.dumps(result,indent=2));raise SystemExit(0 if result['passed'] else 1)
