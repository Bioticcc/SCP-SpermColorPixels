"""Build the annotation-optional R4a attachment-continuation experiment."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import sys
import time

SCP_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SCP_ROOT.parent
if str(SCP_ROOT) not in sys.path:
    sys.path.insert(0, str(SCP_ROOT))
from experiments.audit_r3 import audit as audit_r3
from experiments.integrity import verify_snapshot
from experiments.provenance import build_provenance, dependency_metadata, sha256_file, write_metadata_atomic
from experiments.run_r2 import image_hashes, load_checkpoint
from experiments.run_r3 import tree_hashes
from experiments.r4_adapter import resolve_image


def validate_config(config):
    required = {'schema_version','stage','r3_manifest','expected_image_count','fixture_seed','additional_k',
                'search_max_expansions','solver_max_expansions','null_cost','partial_cost','coverage_weight'}
    if set(config) != required or config['schema_version'] != 'scp.r4a.config.v1' or config['stage'] != 'R4a':
        raise ValueError('Unexpected R4a configuration schema or keys')
    for key in ('expected_image_count','search_max_expansions','solver_max_expansions'):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    for key in ('additional_k','fixture_seed'):
        if type(config[key]) is not int or config[key] < 0:
            raise ValueError(f'{key} must be a nonnegative integer')
    for key in ('null_cost','partial_cost','coverage_weight'):
        if type(config[key]) not in (int,float) or not math.isfinite(config[key]) or config[key] < 0:
            raise ValueError(f'{key} must be finite and nonnegative')


def run(config_path, output, limit=None, resume=False):
    from experiments.r4_benchmark import run_r4a_fixtures
    from experiments.r4_viewer import write_image_viewer, write_index
    started = time.monotonic()
    config_path, output = Path(config_path).resolve(), Path(output).resolve()
    config = json.loads(config_path.read_text()); validate_config(config)
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError('limit must be a positive integer')
    manifest_path = (PROJECT_ROOT/config['r3_manifest']).resolve()
    manifest = json.loads(manifest_path.read_text()); r3 = (PROJECT_ROOT/manifest['run_root']).resolve()
    for name in ('audit','provenance'):
        if sha256_file(r3/f'{name}.json') != manifest[f'{name}_sha256']:
            raise ValueError('Frozen R3 release manifest mismatch')
    input_audit = audit_r3(r3)
    if not input_audit['passed']:
        raise ValueError(f'Frozen R3 audit failed: {input_audit["errors"]}')
    previous = json.loads((r3/'provenance.json').read_text())
    snapshot = {**previous,'source_files':[row for row in previous['source_files'] if row['path']=='Algorithmic_Pixel_Mask_For_Tails/algorithmic_tail_mask.py']}
    if len(snapshot['source_files']) != 1 or not verify_snapshot(snapshot,PROJECT_ROOT)['passed']:
        raise ValueError('Predictor, raw inputs or frozen configuration changed')
    if dependency_metadata() != previous['dependencies']:
        raise ValueError('Dependencies differ from frozen inputs')
    r3_config = json.loads((r3/'config.json').read_text())
    r2_manifest = json.loads((PROJECT_ROOT/r3_config['r2_manifest']).read_text()); r2 = (PROJECT_ROOT/r2_manifest['run_root']).resolve()
    r2_config = json.loads((r2/'config.json').read_text())
    r1_manifest = json.loads((PROJECT_ROOT/r2_config['r1_manifest']).read_text()); r1 = (PROJECT_ROOT/r1_manifest['run_root']).resolve()
    r1_config = json.loads((r1/'config.json').read_text()); baseline = (PROJECT_ROOT/r1_config['baseline_run']).resolve()
    if (SCP_ROOT/'outputs/experiments').resolve() not in output.parents or any(output == frozen or frozen in output.parents for frozen in (r3,r2,r1,baseline)):
        raise ValueError('Use a new experiment output directory outside frozen releases')
    records = [(path.parent,json.loads(path.read_text())) for path in sorted((r1/'images').glob('*/routes.json'))]
    expected = {row['relative_image'] for row in previous['input_manifest']['images']}
    if len(records) != config['expected_image_count'] or {record['relative_image'] for _,record in records} != expected:
        raise ValueError('Frozen image inventory differs')
    for key in ('null_cost','partial_cost','coverage_weight','solver_max_expansions'):
        if config[key] != r2_config[key]:
            raise ValueError('R4a preserves the frozen R2 assignment controls')
    assets = dict(previous['frozen_evidence_sha256'])
    assets.update({str(r3/name):sha256_file(r3/name) for name in ('provenance.json','audit.json','completion.json','config.json')})
    if any(sha256_file(Path(name)) != digest for name,digest in assets.items()):
        raise ValueError('Frozen evidence changed')
    code = [SCP_ROOT/'algorithmic_tail_mask.py']
    for folder in ('experiments','graph_construction','junction_transitions','path_hypotheses','global_assignment','mask_reconstruction','head_anchors','tests/fixtures'):
        code.extend(sorted((SCP_ROOT/folder).glob('*.py')))
    effective = {**config,'limit':limit}
    if resume:
        provenance = json.loads((output/'provenance.json').read_text())
        if (output/'completion.json').exists(): raise ValueError('Run is already complete')
        if provenance['effective_args'] != effective or json.loads((output/'config.json').read_text()) != config:
            raise ValueError('Resume arguments/configuration changed')
        if not verify_snapshot(provenance,PROJECT_ROOT)['passed'] or provenance['frozen_evidence_sha256'] != assets:
            raise ValueError('Resume requires unchanged code and evidence')
        checkpoint = json.loads((output/'fixture_completion.json').read_text())
        if checkpoint != {'files':tree_hashes(output/'fixtures'),'report_sha256':sha256_file(output/'fixture_report.json')}:
            raise ValueError('Fixture checkpoint changed')
        fixtures = json.loads((output/'fixture_report.json').read_text())
    else:
        output.mkdir(parents=True,exist_ok=False)
        provenance = build_provenance(input_root=Path(previous['input_manifest']['input_root']),project_root=PROJECT_ROOT,
            code_paths=code,config_paths=[config_path,manifest_path,baseline/'demo_manifest.json'],
            namespace=argparse.Namespace(**effective),command=list(sys.argv))
        provenance['frozen_evidence_sha256'] = assets
        provenance['source_runs'] = {'r1':str(r1),'r2':str(r2),'r3':str(r3),'baseline':str(baseline)}
        provenance['scope'] = 'Optional attachment continuations on frozen graph evidence; no detector or production status changes.'
        write_metadata_atomic(output/'provenance.json',provenance)
        write_metadata_atomic(output/'config.json',config)
        write_metadata_atomic(output/'r3_input_audit.json',input_audit)
        fixtures = run_r4a_fixtures(output,config)
        if json.loads((output/'fixture_report.json').read_text()) != json.loads(json.dumps(fixtures)):
            raise ValueError('Fixture return value differs from written report')
        write_metadata_atomic(output/'fixture_completion.json',{'files':tree_hashes(output/'fixtures'),'report_sha256':sha256_file(output/'fixture_report.json')})
    summaries = []
    for index,(r1_directory,record) in enumerate(records[:limit],1):
        directory = output/'images'/r1_directory.name
        if resume:
            saved = load_checkpoint(directory,record['relative_image'])
            if saved is not None:
                summaries.append(saved); continue
            if directory.exists(): raise ValueError('Unverified partial image directory requires inspection')
        tick = time.monotonic()
        r2_path = r2/'images'/directory.name/'assignment.json'
        frozen = json.loads(r2_path.read_text())
        routes,payload = resolve_image(record,frozen,config)
        payload['r1_routes_sha256'] = sha256_file(r1_directory/'routes.json')
        payload['r2_assignment_sha256'] = sha256_file(r2_path)
        write_metadata_atomic(directory/'routes.json',routes)
        write_metadata_atomic(directory/'assignment.json',payload)
        summary = write_image_viewer(directory,r1_directory,payload)
        summary.update({'directory':directory.name,'runtime_seconds':time.monotonic()-tick,
            'changed_heads':len(payload['comparison']['changed_head_ids']),
            'extension_limited_searches':sum(row['search_truncated'] for row in payload['extension_diagnostics']),
            'missing_route_observations':len(payload['pool_diagnostics']['approximate_missing_route_heads']),
            'solver_status':payload['assignment']['status']})
        write_metadata_atomic(directory/'image_completion.json',{'summary':summary,'sha256':image_hashes(directory)})
        summaries.append(summary)
        print(f'[{index}/{limit or len(records)}] {record["relative_image"]}: {summary["added_routes"]} added routes, {summary["changed_heads"]} changed heads; {summary["runtime_seconds"]:.2f}s',flush=True)
    demo = json.loads((baseline/'demo_manifest.json').read_text())
    write_index(output,summaries,fixtures,demo)
    integrity = verify_snapshot(provenance,PROJECT_ROOT)
    integrity['changed_frozen_evidence'] = [name for name,digest in assets.items() if sha256_file(Path(name)) != digest]
    integrity['passed'] = integrity['passed'] and not integrity['changed_frozen_evidence']
    write_metadata_atomic(output/'integrity.json',integrity)
    completion = {'stage':'R4a','processed_images':len(summaries),'available_images':len(records),
        'full_inventory_run':len(summaries)==len(records),'integrity_passed':integrity['passed'],
        'invocation_wall_seconds':time.monotonic()-started,'images':summaries,'resumed':resume,
        'scope':'Changed deterministic path proposals; no biological accuracy measurement.'}
    write_metadata_atomic(output/'completion.json',completion)
    if not integrity['passed']: raise RuntimeError('Final source/evidence integrity failed')
    return completion


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=PROJECT_ROOT/'configs/overlap_r4a.json')
    parser.add_argument('--output-root',type=Path,required=True)
    parser.add_argument('--limit',type=int)
    parser.add_argument('--resume',action='store_true')
    args = parser.parse_args()
    result = run(args.config,args.output_root,args.limit,args.resume)
    print(json.dumps({key:value for key,value in result.items() if key!='images'},indent=2))
