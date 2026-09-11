"""Reconstruct logical masks around frozen R2 selections without reselecting paths."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

SCP_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SCP_ROOT.parent
if str(SCP_ROOT) not in sys.path:
    sys.path.insert(0, str(SCP_ROOT))

from experiments.audit_r2 import audit as audit_r2
from experiments.integrity import verify_snapshot
from experiments.provenance import build_provenance, dependency_metadata, sha256_file, write_metadata_atomic
from experiments.r3_adapter import reconstruct_image
from experiments.r3_contact_sheet import write_contact_sheet
from experiments.run_r2 import image_hashes, load_checkpoint


def tree_hashes(root: Path) -> dict:
    return {str(path.relative_to(root)): sha256_file(path) for path in sorted(root.rglob('*')) if path.is_file()}


def validate_config(config: dict) -> None:
    required = {'schema_version', 'stage', 'r2_manifest', 'expected_image_count', 'fixture_seed', 'reconstruction'}
    if set(config) != required or config['schema_version'] != 'scp.r3.config.v1' or config['stage'] != 'R3':
        raise ValueError('Unexpected R3 configuration schema or keys')
    if type(config['expected_image_count']) is not int or config['expected_image_count'] < 1:
        raise ValueError('expected_image_count must be a positive integer')
    if type(config['fixture_seed']) is not int or config['fixture_seed'] < 0:
        raise ValueError('fixture_seed must be a non-negative integer')
    if not isinstance(config['reconstruction'], dict) or set(config['reconstruction']) != {'min_radius_px', 'crossing_exclusion_px', 'tie_epsilon'}:
        raise ValueError('R3 must explicitly record all reconstruction settings')
    # The reconstruction module owns its parameter validation.
    import numpy as np
    from mask_reconstruction import reconstruct_tracks
    reconstruct_tracks(np.zeros((2, 2), bool), [], config=config['reconstruction'])


def run(config_path: Path, output: Path, limit: int | None = None, resume: bool = False) -> dict:
    from experiments.r3_benchmark import run_r3_fixtures
    from experiments.r3_viewer import write_image_viewer, write_index

    started = time.monotonic()
    config_path, output = config_path.resolve(), output.resolve()
    config = json.loads(config_path.read_text()); validate_config(config)
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError('limit must be a positive integer')
    manifest_path = (PROJECT_ROOT/config['r2_manifest']).resolve()
    manifest = json.loads(manifest_path.read_text())
    r2 = (PROJECT_ROOT/manifest['run_root']).resolve()
    if (SCP_ROOT/'outputs/experiments').resolve() not in output.parents or output == r2 or r2 in output.parents:
        raise ValueError('Use a new experiment directory outside frozen input releases')
    for name in ('audit', 'provenance'):
        if sha256_file(r2/f'{name}.json') != manifest[f'{name}_sha256']:
            raise ValueError(f'R2 {name} differs from its release manifest')
    input_audit = audit_r2(r2)
    if not input_audit['passed']:
        raise ValueError(f'Frozen R2 audit failed: {input_audit["errors"]}')
    original = json.loads((r2/'provenance.json').read_text())
    snapshot = {**original, 'source_files': [row for row in original['source_files']
                if row['path'] == 'Algorithmic_Pixel_Mask_For_Tails/algorithmic_tail_mask.py']}
    if len(snapshot['source_files']) != 1 or not verify_snapshot(snapshot, PROJECT_ROOT)['passed']:
        raise ValueError('Raw images, original predictor or frozen configuration changed')
    if dependency_metadata() != original['dependencies']:
        raise ValueError('Runtime dependencies differ from frozen R2')
    r2_config = json.loads((r2/'config.json').read_text())
    r1_manifest = json.loads((PROJECT_ROOT/r2_config['r1_manifest']).read_text())
    r1 = (PROJECT_ROOT/r1_manifest['run_root']).resolve()
    r1_config = json.loads((r1/'config.json').read_text())
    baseline = (PROJECT_ROOT/r1_config['baseline_run']).resolve()
    if output == r1 or r1 in output.parents or output == baseline or baseline in output.parents:
        raise ValueError('Output must be outside all frozen input runs')
    source_records = {record['relative_image']: record for path in sorted((baseline/'predictions/json').glob('*.json'))
                      for record in [json.loads(path.read_text())]}
    r2_records = [(path.parent, json.loads(path.read_text())) for path in sorted((r2/'images').glob('*/assignment.json'))]
    expected = {row['relative_image'] for row in original['input_manifest']['images']}
    if len(r2_records) != config['expected_image_count'] or {row['relative_image'] for _, row in r2_records} != expected or set(source_records) != expected:
        raise ValueError('Frozen source/assignment inventory is incomplete or differs')
    r1_provenance = json.loads((r1/'provenance.json').read_text())
    assets = {str(path): sha256_file(path) for path in r2.rglob('*') if path.is_file()}
    assets.update(original['frozen_r1_sha256'])
    assets.update(r1_provenance['frozen_evidence_sha256'])
    for name, digest in assets.items():
        if sha256_file(Path(name)) != digest:
            raise ValueError(f'Frozen evidence changed: {name}')
    code = [SCP_ROOT/'algorithmic_tail_mask.py']
    for folder in ('experiments', 'graph_construction', 'junction_transitions', 'path_hypotheses', 'global_assignment', 'mask_reconstruction', 'tests/fixtures'):
        code.extend(sorted((SCP_ROOT/folder).glob('*.py')))
    effective = {**config, 'limit': limit}
    if resume:
        provenance = json.loads((output/'provenance.json').read_text())
        if (output/'completion.json').exists():
            raise ValueError('Run is already complete')
        if provenance['effective_args'] != effective or json.loads((output/'config.json').read_text()) != config:
            raise ValueError('Resume configuration or arguments changed')
        if not verify_snapshot(provenance, PROJECT_ROOT)['passed'] or provenance['frozen_evidence_sha256'] != assets:
            raise ValueError('Resume requires unchanged code, configuration, inputs and frozen evidence')
        checkpoint = json.loads((output/'fixture_completion.json').read_text())
        if checkpoint['files'] != tree_hashes(output/'fixtures') or checkpoint['report_sha256'] != sha256_file(output/'fixture_report.json'):
            raise ValueError('Fixture checkpoint changed')
        fixtures = json.loads((output/'fixture_report.json').read_text())
    else:
        output.mkdir(parents=True, exist_ok=False)
        provenance = build_provenance(input_root=Path(original['input_manifest']['input_root']), project_root=PROJECT_ROOT,
            code_paths=code, config_paths=[config_path, manifest_path, baseline/'demo_manifest.json'],
            namespace=argparse.Namespace(**effective), command=list(sys.argv))
        provenance['scope'] = 'R3 mask reconstruction around frozen R2 paths; no path reselection or RGB detection rerun.'
        provenance['frozen_evidence_sha256'] = assets
        write_metadata_atomic(output/'provenance.json', provenance)
        write_metadata_atomic(output/'config.json', config)
        write_metadata_atomic(output/'r2_input_audit.json', input_audit)
        fixtures = run_r3_fixtures(output/'fixtures', {**config, 'r2_run': str(r2)})
        write_metadata_atomic(output/'fixture_report.json', fixtures)
        write_metadata_atomic(output/'fixture_completion.json', {'files': tree_hashes(output/'fixtures'), 'report_sha256': sha256_file(output/'fixture_report.json')})
    summaries = []
    for index, (r2_directory, r2_record) in enumerate(r2_records[:limit] if limit else r2_records, 1):
        directory = output/'images'/r2_directory.name
        if resume:
            saved = load_checkpoint(directory, r2_record['relative_image'])
            if saved:
                summaries.append(saved)
                print(f'[{index}] verified checkpoint: {r2_record["relative_image"]}', flush=True)
                continue
            if directory.exists():
                raise ValueError(f'Incomplete image output requires inspection: {directory}')
        tick = time.monotonic()
        r1_directory = r1/'images'/r2_directory.name
        r1_record = json.loads((r1_directory/'routes.json').read_text())
        payload = reconstruct_image(directory, source_records[r2_record['relative_image']], r1_record, r2_record, config)
        payload['r2_assignment_sha256'] = sha256_file(r2_directory/'assignment.json')
        payload['r1_routes_sha256'] = sha256_file(r1_directory/'routes.json')
        write_metadata_atomic(directory/'reconstruction.json', payload)
        summary = write_image_viewer(directory, r1_directory, r2_directory, r1_record, payload)
        summary.update({'directory': directory.name, 'runtime_seconds': time.monotonic()-tick,
                        'head_tail_evidence_conflict_pixels': sum(len(row['source_pixels_xy']) for row in payload['head_tail_evidence_conflicts']),
                        'tail_growth_pixels_removed_at_foreign_heads': sum(item['measurements']['tail_growth_pixels_removed_at_foreign_heads'] for item in payload['instances']),
                        'ordinary_duplicate_pixels': sum(c['core_diagnostics']['ordinary_duplicate_pixels'] for c in payload['components']),
                        'shared_pixels': sum(c['core_diagnostics']['shared_pixels'] for c in payload['components'])})
        write_metadata_atomic(directory/'image_completion.json', {'summary': summary, 'sha256': image_hashes(directory)})
        summaries.append(summary)
        print(f'[{index}/{limit or len(r2_records)}] {summary["relative_image"]}: {summary["instances"]} instances, '
              f'{summary["ordinary_duplicate_pixels"]} core tail duplicate pixels; {summary["runtime_seconds"]:.2f}s', flush=True)
    demo = json.loads((baseline/'demo_manifest.json').read_text())
    write_contact_sheet(output, r1, r2, summaries, demo)
    write_index(output, summaries, fixtures, demo)
    integrity = verify_snapshot(provenance, PROJECT_ROOT)
    integrity['changed_frozen_evidence'] = [name for name, digest in assets.items() if sha256_file(Path(name)) != digest]
    integrity['passed'] = integrity['passed'] and not integrity['changed_frozen_evidence']
    write_metadata_atomic(output/'integrity.json', integrity)
    completion = {'stage': 'R3', 'processed_images': len(summaries), 'available_images': len(r2_records),
                  'full_inventory_run': len(summaries) == len(r2_records), 'integrity_passed': integrity['passed'],
                  'invocation_wall_seconds': time.monotonic()-started, 'resumed': resume, 'images': summaries,
                  'scope': 'Conditional logical-mask reconstruction, preserving selected-path limitations; no real-image accuracy estimate.'}
    write_metadata_atomic(output/'completion.json', completion)
    if not integrity['passed']:
        raise RuntimeError('Source, inputs, configuration or frozen evidence changed during execution')
    return completion


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=PROJECT_ROOT/'configs/overlap_r3.json')
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--limit', type=int, help='Development-only partial run')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    result = run(args.config, args.output_root, args.limit, args.resume)
    print(json.dumps({key: value for key, value in result.items() if key != 'images'}, indent=2))
