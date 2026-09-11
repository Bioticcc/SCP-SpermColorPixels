"""Build an R1 route-proposal report using the frozen R0 image evidence."""
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

from experiments.integrity import verify_snapshot
from experiments.provenance import build_provenance, dependency_metadata, sha256_file, write_metadata_atomic
from experiments.r1_adapter import analyze_image
from experiments.r1_benchmark import run_r1_fixtures
from experiments.r1_viewer import write_image_viewer, write_index


def evidence_paths(predictions: Path, records: list[dict]) -> list[Path]:
    paths = {predictions/'summary.json', *list((predictions/'json').glob('*.json'))}
    for row in records:
        paths.update(Path(row['outputs'][key]) for key in ('head_mask', 'tail_mask'))
        paths.update(Path(c['skeleton_mask']) for c in row['head_connected_postprocessing']['crop_candidates_detail'])
    if any(predictions.resolve() not in path.resolve().parents for path in paths):
        raise ValueError('Saved evidence references files outside its prediction directory')
    return sorted(paths)


def validate_config(config: dict) -> None:
    required = {'schema_version', 'stage', 'baseline_run', 'k', 'diagnostic_k',
                'max_expansions', 'fixture_seed', 'head_contact_radius',
                'centerline_tolerance_px', 'expected_image_count'}
    if set(config) != required or config['stage'] != 'R1' or config['schema_version'] != 'scp.r1.config.v1':
        raise ValueError('Unexpected R1 configuration schema or keys')
    for key in ('k', 'diagnostic_k', 'max_expansions', 'expected_image_count'):
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError(f'{key} must be a positive integer')
    if config['diagnostic_k'] < config['k']:
        raise ValueError('diagnostic_k must be at least k')


def _checkpoint_files(directory: Path) -> dict[str, str]:
    return {name: sha256_file(directory/name) for name in ('routes.json', 'index.html', 'original.jpg', 'baseline.jpg')}


def load_image_checkpoint(directory: Path, relative_image: str) -> dict | None:
    checkpoint = directory/'image_completion.json'
    if not checkpoint.is_file():
        return None
    saved = json.loads(checkpoint.read_text())
    if saved['summary']['relative_image'] != relative_image or saved['sha256'] != _checkpoint_files(directory):
        raise ValueError(f'Image checkpoint changed or has incorrect identity: {directory}')
    return saved['summary']


def run(config_path: Path, output: Path, limit: int | None = None, resume: bool = False) -> dict:
    start = time.monotonic()
    config_path, output = config_path.resolve(), output.resolve()
    config = json.loads(config_path.read_text())
    validate_config(config)
    baseline = (PROJECT_ROOT/config['baseline_run']).resolve()
    predictions = baseline/'predictions'
    experiments = (SCP_ROOT/'outputs/experiments').resolve()
    if experiments not in output.parents or output == baseline or baseline in output.parents:
        raise ValueError('Use a new experiment directory outside the frozen baseline')
    original = json.loads((baseline/'provenance.json').read_text())
    snapshot = dict(original)
    snapshot['source_files'] = [row for row in original['source_files']
                                if row['path'] == 'Algorithmic_Pixel_Mask_For_Tails/algorithmic_tail_mask.py']
    if len(snapshot['source_files']) != 1 or not verify_snapshot(snapshot, PROJECT_ROOT)['passed']:
        raise ValueError('Frozen predictor/input/configuration integrity does not match R0')
    if dependency_metadata() != original['dependencies']:
        raise ValueError('Runtime environment differs from R0; record a new baseline before comparison')
    baseline_config = json.loads((baseline/'config.json').read_text())
    if config['head_contact_radius'] != baseline_config['arguments']['head_contact_radius']:
        raise ValueError('R1 must preserve baseline head-contact settings')
    records = [json.loads(p.read_text()) for p in sorted((predictions/'json').glob('*.json'))]
    expected = {r['relative_image'] for r in original['input_manifest']['images']}
    if {r['relative_image'] for r in records} != expected or len(records) != config['expected_image_count']:
        raise ValueError('Frozen image inventory is incomplete or changed')
    if limit is not None and limit <= 0:
        raise ValueError('Smoke-test limit must be positive')
    assets = evidence_paths(predictions, records)
    fingerprints = {str(p): sha256_file(p) for p in assets}
    code = [SCP_ROOT/'algorithmic_tail_mask.py']
    for folder in ('experiments', 'graph_construction', 'junction_transitions', 'path_hypotheses', 'tests/fixtures'):
        code.extend(sorted((SCP_ROOT/folder).glob('*.py')))
    if resume:
        provenance = json.loads((output/'provenance.json').read_text())
        if (output/'completion.json').exists():
            raise ValueError('Run is already complete; no resume is necessary')
        if json.loads((output/'config.json').read_text()) != config:
            raise ValueError('Resume configuration differs from original run')
        if provenance['effective_args'] != {**config, 'limit': limit}:
            raise ValueError('Resume arguments differ from original run')
        if not verify_snapshot(provenance, PROJECT_ROOT)['passed'] or provenance['frozen_evidence_sha256'] != fingerprints:
            raise ValueError('Resume requires unchanged source, configuration, inputs, and frozen evidence')
        saved_fixture = json.loads((output/'fixture_completion.json').read_text())
        if sha256_file(output/'fixture_report.json') != saved_fixture['report_sha256']:
            raise ValueError('Fixture checkpoint changed')
        fixtures = json.loads((output/'fixture_report.json').read_text())
    else:
        output.mkdir(parents=True, exist_ok=False)
        provenance = build_provenance(input_root=Path(original['input_manifest']['input_root']),
            project_root=PROJECT_ROOT, code_paths=code, config_paths=[config_path, baseline/'demo_manifest.json'],
            namespace=argparse.Namespace(**config, limit=limit), command=list(sys.argv))
        provenance['scope'] = 'R1 graph/search stage on frozen R0 masks; production decisions are retained.'
        provenance['frozen_evidence_sha256'] = fingerprints
        provenance['baseline_provenance_sha256'] = sha256_file(baseline/'provenance.json')
        write_metadata_atomic(output/'provenance.json', provenance)
        write_metadata_atomic(output/'config.json', config)
        fixture_config = {**config, 'baseline_run': str(baseline)}
        fixtures = run_r1_fixtures(output, fixture_config)
        write_metadata_atomic(output/'fixture_completion.json', {'report_sha256': sha256_file(output/'fixture_report.json')})
    summaries = []
    for index, record in enumerate(records[:limit] if limit else records, 1):
        directory = output/'images'/Path(record['outputs']['tail_mask']).name.removesuffix('_tail_mask.png')
        if resume:
            summary = load_image_checkpoint(directory, record['relative_image'])
            if summary:
                summaries.append(summary)
                print(f"[{index}/{limit or len(records)}] verified checkpoint: {record['relative_image']}", flush=True)
                continue
            if directory.exists():
                raise ValueError(f'Incomplete image artifacts need inspection before retry: {directory}')
        tick = time.monotonic()
        result = analyze_image(record, predictions, config)
        directory.mkdir(parents=True)
        write_metadata_atomic(directory/'routes.json', result)
        summary = write_image_viewer(directory, record, result, config)
        summary['runtime_seconds'] = time.monotonic()-tick
        write_metadata_atomic(directory/'image_completion.json', {'summary': summary, 'sha256': _checkpoint_files(directory)})
        summaries.append(summary)
        print(f"[{index}/{limit or len(records)}] {record['relative_image']}: {summary['heads']} anchored searches, {summary['limited_searches']} limited; {summary['runtime_seconds']:.1f}s", flush=True)
    demo = json.loads((baseline/'demo_manifest.json').read_text())
    write_index(output, summaries, fixtures, demo)
    integrity = verify_snapshot(provenance, PROJECT_ROOT)
    integrity['changed_frozen_evidence'] = [str(p) for p in assets if sha256_file(p) != fingerprints[str(p)]]
    integrity['passed'] = integrity['passed'] and not integrity['changed_frozen_evidence']
    write_metadata_atomic(output/'integrity.json', integrity)
    completion = {'stage': 'R1', 'processed_images': len(summaries), 'available_images': len(records),
                  'full_inventory_run': len(summaries) == len(records), 'invocation_wall_seconds': time.monotonic()-start,
                  'image_processing_seconds': sum(row['runtime_seconds'] for row in summaries), 'resumed': resume,
                  'integrity_passed': integrity['passed'], 'images': summaries,
                  'scope': 'Experimental route availability and engineering evidence only; no real accuracy claim.'}
    write_metadata_atomic(output/'completion.json', completion)
    if not integrity['passed']:
        raise RuntimeError('Code, configuration, input, or frozen evidence changed during R1 execution')
    return completion


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=PROJECT_ROOT/'configs/overlap_r1.json')
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--limit', type=int, help='Partial development smoke only; release requires all images')
    parser.add_argument('--resume', action='store_true', help='Verify and continue an interrupted run with unchanged code/config/evidence')
    args = parser.parse_args()
    result = run(args.config, args.output_root, args.limit, args.resume)
    print(json.dumps({k: v for k, v in result.items() if k != 'images'}, indent=2))
