"""Build the R2 joint-assignment experiment from verified, frozen R1 routes."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys
import time

SCP_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SCP_ROOT.parent
if str(SCP_ROOT) not in sys.path:
    sys.path.insert(0, str(SCP_ROOT))

from experiments.audit_r1 import audit as audit_r1
from experiments.integrity import verify_snapshot
from experiments.provenance import build_provenance, dependency_metadata, sha256_file, write_metadata_atomic
from experiments.r2_assignment import assign_pool
from experiments.r2_benchmark import run_r2_fixtures
from experiments.r2_candidates import build_candidate_pool
from experiments.r2_contact_sheet import write_contact_sheet


def validate_config(config: dict) -> None:
    required = {'schema_version', 'stage', 'r1_manifest', 'expected_image_count', 'k',
                'search_max_expansions', 'solver_max_expansions', 'fixture_seed',
                'null_cost', 'partial_cost', 'coverage_weight'}
    if set(config) != required or config['stage'] != 'R2' or config['schema_version'] != 'scp.r2.config.v1':
        raise ValueError('Unexpected R2 configuration schema or keys')
    for key in ('expected_image_count', 'k', 'search_max_expansions', 'solver_max_expansions'):
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError(f'{key} must be a positive integer')
    if type(config['fixture_seed']) is not int or config['fixture_seed'] < 0:
        raise ValueError('fixture_seed must be a non-negative integer')
    for key in ('null_cost', 'partial_cost', 'coverage_weight'):
        if type(config[key]) not in (int, float) or not math.isfinite(config[key]) or config[key] < 0:
            raise ValueError(f'{key} must be a finite non-negative number')


def source_pixels(item: dict) -> set[tuple[int, int]]:
    metadata = item['metadata']
    points = metadata.get('points_xy_ordered', metadata.get('pixels_xy_unordered', []))
    roi = metadata.get('roi_xyxy') or [0, 0]
    return {(int(x) + roi[0], int(y) + roi[1]) for x, y in points}


def compare_to_baseline(record: dict, hypotheses: list[dict], assignment: dict) -> list[dict]:
    """Exact raster overlap diagnostics; neither raster is biological gold."""
    by_id = {item['id']: item for item in hypotheses}
    rows = []
    for head, choice in sorted(assignment['selected_by_head'].items()):
        item = by_id[choice]
        pixels = source_pixels(item)
        previous = []
        for component in record['components']:
            x0, y0 = component['roi_xyxy'][:2]
            for old in component['baseline']:
                if str(old['head_id']) != head:
                    continue
                old_pixels = {(x+x0, y+y0) for x, y in old['pixels_xy']}
                previous.append({'crop_id': old['crop_id'], 'tail_id': component['tail_id'],
                                 'status': old['status'], 'baseline_pixel_count': len(old_pixels),
                                 'shared_pixel_count': len(old_pixels & pixels),
                                 'exact_same_source_pixels': pixels == old_pixels})
        rows.append({'head_id': head, 'selected_id': choice, 'termination': item['termination'],
                     'selected_pixel_count': len(pixels), 'baseline_comparisons': previous,
                     'changed_from_unary_independent': head in assignment['changed_head_ids'],
                     'interpretation': 'Pixel overlap with retained baseline, not accuracy or completeness truth.'})
    return rows


def image_hashes(directory: Path) -> dict:
    return {str(path.relative_to(directory)): sha256_file(path)
            for path in sorted(directory.rglob('*')) if path.is_file() and path.name != 'image_completion.json'}


def load_checkpoint(directory: Path, relative_image: str) -> dict | None:
    path = directory/'image_completion.json'
    if not path.exists():
        return None
    checkpoint = json.loads(path.read_text())
    if checkpoint['summary']['relative_image'] != relative_image or checkpoint['sha256'] != image_hashes(directory):
        raise ValueError(f'R2 image checkpoint changed: {directory}')
    return checkpoint['summary']


def run(config_path: Path, output: Path, limit: int | None = None, resume: bool = False) -> dict:
    from experiments.r2_viewer import write_image_viewer, write_index

    started = time.monotonic()
    config_path, output = config_path.resolve(), output.resolve()
    config = json.loads(config_path.read_text())
    validate_config(config)
    manifest_path = (PROJECT_ROOT/config['r1_manifest']).resolve()
    manifest = json.loads(manifest_path.read_text())
    r1 = (PROJECT_ROOT/manifest['run_root']).resolve()
    if (SCP_ROOT/'outputs/experiments').resolve() not in output.parents or r1 == output or r1 in output.parents:
        raise ValueError('Use a new experiment output directory outside the R1 release')
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError('limit must be a positive integer')
    for name in ('audit', 'provenance'):
        if sha256_file(r1/f'{name}.json') != manifest[f'{name}_sha256']:
            raise ValueError(f'Frozen R1 {name} differs from release manifest')
    verified_r1 = audit_r1(r1)
    if not verified_r1['passed']:
        raise ValueError(f'Frozen R1 artifact audit failed: {verified_r1["errors"]}')
    original = json.loads((r1/'provenance.json').read_text())
    snapshot = {**original, 'source_files': [row for row in original['source_files']
                if row['path'] == 'Algorithmic_Pixel_Mask_For_Tails/algorithmic_tail_mask.py']}
    if len(snapshot['source_files']) != 1 or not verify_snapshot(snapshot, PROJECT_ROOT)['passed']:
        raise ValueError('Current predictor, raw inputs or frozen R1 configuration differ from the recorded evidence')
    if dependency_metadata() != original['dependencies']:
        raise ValueError('Runtime environment differs from the frozen R1 environment')
    r1_config = json.loads((r1/'config.json').read_text())
    if config['k'] != r1_config['k'] or config['search_max_expansions'] != r1_config['max_expansions']:
        raise ValueError('Fixture route settings must match frozen R1 settings')
    baseline = (PROJECT_ROOT/r1_config['baseline_run']).resolve()
    directories = sorted((r1/'images').iterdir())
    records = [(directory, json.loads((directory/'routes.json').read_text())) for directory in directories]
    identities = [record['relative_image'] for _, record in records]
    if len(identities) != config['expected_image_count'] or len(set(identities)) != len(identities):
        raise ValueError('R1 inventory is incomplete or duplicated')
    if set(identities) != {row['relative_image'] for row in original['input_manifest']['images']}:
        raise ValueError('R1 inventory does not match source inputs')
    assets = [r1/name for name in ('provenance.json', 'audit.json', 'config.json', 'completion.json', 'integrity.json')]
    assets += [path for directory in directories for path in directory.iterdir() if path.is_file()]
    fingerprints = {str(path): sha256_file(path) for path in assets}
    code = [SCP_ROOT/'algorithmic_tail_mask.py']
    for folder in ('experiments', 'graph_construction', 'junction_transitions', 'path_hypotheses', 'global_assignment', 'tests/fixtures'):
        code.extend(sorted((SCP_ROOT/folder).glob('*.py')))
    effective_args = {**config, 'limit': limit}
    if resume:
        provenance = json.loads((output/'provenance.json').read_text())
        if (output/'completion.json').exists():
            raise ValueError('Run is already complete')
        if provenance['effective_args'] != effective_args or json.loads((output/'config.json').read_text()) != config:
            raise ValueError('Resume arguments/configuration changed')
        if not verify_snapshot(provenance, PROJECT_ROOT)['passed'] or fingerprints != provenance['frozen_r1_sha256']:
            raise ValueError('Resume requires unchanged code, configuration, inputs and frozen R1 artifacts')
        checkpoint = json.loads((output/'fixture_completion.json').read_text())
        current = {str(p.relative_to(output)): sha256_file(p) for p in sorted((output/'fixtures').rglob('*')) if p.is_file()}
        current['fixture_report.json'] = sha256_file(output/'fixture_report.json')
        if checkpoint != current:
            raise ValueError('Fixture checkpoint changed')
        fixtures = json.loads((output/'fixture_report.json').read_text())
    else:
        output.mkdir(parents=True, exist_ok=False)
        provenance = build_provenance(input_root=Path(original['input_manifest']['input_root']),
            project_root=PROJECT_ROOT, code_paths=code, config_paths=[config_path, manifest_path, baseline/'demo_manifest.json'],
            namespace=argparse.Namespace(**effective_args), command=list(sys.argv))
        provenance['scope'] = 'R2 assignment of frozen R1 graph routes. Cached evidence; no rerun of RGB segmentation.'
        provenance['frozen_r1_sha256'] = fingerprints
        write_metadata_atomic(output/'provenance.json', provenance)
        write_metadata_atomic(output/'config.json', config)
        write_metadata_atomic(output/'r1_input_audit.json', verified_r1)
        fixtures = run_r2_fixtures(output, config)
        checkpoint = {str(p.relative_to(output)): sha256_file(p) for p in sorted((output/'fixtures').rglob('*')) if p.is_file()}
        checkpoint['fixture_report.json'] = sha256_file(output/'fixture_report.json')
        write_metadata_atomic(output/'fixture_completion.json', checkpoint)
    summaries = []
    for number, (r1_directory, record) in enumerate(records[:limit] if limit else records, 1):
        directory = output/'images'/r1_directory.name
        if resume:
            saved = load_checkpoint(directory, record['relative_image'])
            if saved:
                summaries.append(saved)
                print(f'[{number}] verified checkpoint: {record["relative_image"]}', flush=True)
                continue
            if directory.exists():
                raise ValueError(f'Incomplete image output requires inspection: {directory}')
        tick = time.monotonic()
        pool = build_candidate_pool(record, config)
        solution = assign_pool(pool['hypotheses'], max_expansions=config['solver_max_expansions'])
        serialized = [asdict(item) for item in pool['hypotheses']]
        payload = {**pool, 'schema_version': 'scp.r2.image-assignment.v1', 'relative_image': record['relative_image'],
                   'hypotheses': serialized, 'assignment': solution,
                   'baseline_changes': compare_to_baseline(record, serialized, solution),
                   'r1_record_sha256': sha256_file(r1_directory/'routes.json'),
                   'production_statuses_changed': False}
        directory.mkdir(parents=True)
        write_metadata_atomic(directory/'assignment.json', payload)
        summary = write_image_viewer(directory, r1_directory, record, payload)
        by_id = {item.id: item for item in pool['hypotheses']}
        summary.update({'relative_image': record['relative_image'], 'directory': directory.name,
            'heads': len(solution['selected_by_head']), 'hypotheses': len(serialized),
            'solver_status': solution['status'], 'groups': len(solution['groups']),
            'limited_groups': sum(g['assignment']['status'] != 'optimal' for g in solution['groups']),
            'limited_head_alternatives': sum(a['status'] == 'limit' for g in solution['groups'] for a in g['assignment']['head_alternatives'].values()),
            'route_searches_limited': sum(bool(s['diagnostics']['truncated']) for c in record['components'] for s in c['searches'].values()),
            'changed_from_unary': len(solution['changed_head_ids']),
            'independent_conflicts': len(solution['independent_conflicts']),
            'joint_conflicts': len(solution['joint_conflicts']),
            'isolated_controls': len(pool['pool_diagnostics']['isolated_controls']),
            'approximate_missing_routes': len(pool['pool_diagnostics']['approximate_missing_route_heads']),
            'terminations': dict(Counter(by_id[value].termination for value in solution['selected_by_head'].values())),
            'runtime_seconds': time.monotonic()-tick})
        write_metadata_atomic(directory/'image_completion.json', {'summary': summary, 'sha256': image_hashes(directory)})
        summaries.append(summary)
        print(f'[{number}/{limit or len(records)}] {record["relative_image"]}: {summary["heads"]} heads, '
              f'{summary["changed_from_unary"]} changed, {solution["status"]}; {summary["runtime_seconds"]:.2f}s', flush=True)
    demo = json.loads((baseline/'demo_manifest.json').read_text())
    write_contact_sheet(output, r1, summaries, demo)
    write_index(output, summaries, fixtures, demo)
    integrity = verify_snapshot(provenance, PROJECT_ROOT)
    integrity['changed_frozen_r1'] = [path for path, fingerprint in fingerprints.items() if sha256_file(Path(path)) != fingerprint]
    integrity['passed'] = integrity['passed'] and not integrity['changed_frozen_r1']
    write_metadata_atomic(output/'integrity.json', integrity)
    completion = {'stage': 'R2', 'processed_images': len(summaries), 'available_images': len(records),
                  'full_inventory_run': len(summaries) == len(records), 'resumed': resume,
                  'integrity_passed': integrity['passed'], 'invocation_wall_seconds': time.monotonic()-started,
                  'image_processing_seconds': sum(s['runtime_seconds'] for s in summaries), 'images': summaries,
                  'production_statuses_changed': False,
                  'scope': 'Finite-pool optimization and constructed geometry evidence; no real-image accuracy estimate.'}
    write_metadata_atomic(output/'completion.json', completion)
    if not integrity['passed']:
        raise RuntimeError('Code, inputs, configuration or frozen artifacts changed during execution')
    return completion


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=PROJECT_ROOT/'configs/overlap_r2.json')
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--limit', type=int, help='Partial development smoke; releases require all 24 images')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    result = run(args.config, args.output_root, args.limit, args.resume)
    print(json.dumps({k: v for k, v in result.items() if k != 'images'}, indent=2))
