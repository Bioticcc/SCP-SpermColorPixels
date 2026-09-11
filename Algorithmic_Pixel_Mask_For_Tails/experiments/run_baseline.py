#!/usr/bin/env python3
"""Freeze an unchanged full SCP run and build the R0 comparison demonstration."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

SCP_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SCP_ROOT.parent
if str(SCP_ROOT) not in sys.path:
    sys.path.insert(0, str(SCP_ROOT))

import algorithmic_tail_mask as atm
from experiments.configuration import baseline_arguments
from experiments.provenance import build_provenance, compare_runs, sha256_file, write_metadata_atomic
from experiments.benchmark import run_fixture_baseline
from experiments.viewer import write_baseline_viewer
from experiments.integrity import verify_snapshot


def run_baseline(config_path: Path, output_root: Path) -> dict:
    config_path, output_root = config_path.resolve(), output_root.resolve()
    config = json.loads(config_path.read_text())
    input_root = PROJECT_ROOT / config['input_root']
    reference_root = PROJECT_ROOT / config['reference_root']
    demo_path = PROJECT_ROOT / config['demo_manifest']
    demo = json.loads(demo_path.read_text())
    if not reference_root.is_dir():
        raise FileNotFoundError(reference_root)
    # Reject writes inside sources/canonical outputs, including paths via symlinks.
    protected = [input_root.resolve(), reference_root.resolve(), (SCP_ROOT / 'annotations').resolve()]
    if any(output_root == path or path in output_root.parents for path in protected):
        raise ValueError('Experiment root must be separate from source, annotation, and canonical data')
    images = atm.collect_images(input_root, None)
    if len(images) != config['expected_image_count']:
        raise ValueError(f"Expected {config['expected_image_count']} inputs; found {len(images)}")
    image_ids = {p.relative_to(input_root).as_posix() for p in images}
    demo_ids = [row['relative_image'] for row in demo['images']]
    if len(demo_ids) != 6 or len(set(demo_ids)) != 6 or not set(demo_ids) <= image_ids:
        raise ValueError('Demo manifest must reference six distinct existing input images')
    argv, args = baseline_arguments(config, input_root, output_root / 'predictions')
    command = [sys.executable, str(SCP_ROOT / 'algorithmic_tail_mask.py'), *argv]
    code_paths = [SCP_ROOT / 'algorithmic_tail_mask.py', *sorted((SCP_ROOT / 'experiments').glob('*.py')),
                  *sorted((SCP_ROOT / 'tests/fixtures').glob('*.py'))]
    provenance = build_provenance(input_root=input_root, project_root=PROJECT_ROOT,
                                  code_paths=code_paths, config_paths=[config_path, demo_path],
                                  namespace=args, command=command)
    provenance['mode_settings'] = atm.mode_settings(args.mode, args.min_component_area)
    provenance['baseline_id'] = config['baseline_id']
    provenance['reference_metadata_hashes'] = {
        p.relative_to(reference_root).as_posix(): sha256_file(p)
        for p in [reference_root / 'summary.json', *sorted((reference_root / 'json').glob('*.json'))]}
    output_root.mkdir(parents=True, exist_ok=False)
    write_metadata_atomic(output_root / 'provenance.json', provenance)
    write_metadata_atomic(output_root / 'input_manifest.json', provenance['input_manifest'])
    write_metadata_atomic(output_root / 'config.json', config)
    write_metadata_atomic(output_root / 'demo_manifest.json', demo)
    started = time.monotonic()
    print(f'Running unchanged baseline on {len(images)} images; log: {output_root / "prediction_stdout.log"}', flush=True)
    with (output_root / 'prediction_stdout.log').open('w') as log:
        process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
    if process.returncode:
        write_metadata_atomic(output_root / 'failure.json', {'stage': 'predictions', 'exit_code': process.returncode})
        raise RuntimeError(f'Baseline subprocess failed; see {log.name}')
    changed_inputs = [row['relative_image'] for row in provenance['input_manifest']['images']
                      if sha256_file(input_root / row['relative_image']) != row['sha256']]
    changed_code = [row['path'] for row in provenance['source_files']
                    if sha256_file(PROJECT_ROOT / row['path']) != row['sha256']]
    comparison = compare_runs(reference_root, output_root / 'predictions')
    comparison['inputs_changed_during_run'] = changed_inputs
    comparison['code_changed_during_run'] = changed_code
    write_metadata_atomic(output_root / 'comparison.json', comparison)
    fixtures = run_fixture_baseline(output_root, args, seed=config['fixture_seed'])
    write_baseline_viewer(output_root, input_root, reference_root, output_root / 'predictions', demo, comparison, fixtures)
    integrity = verify_snapshot(provenance, PROJECT_ROOT)
    write_metadata_atomic(output_root / 'integrity.json', integrity)
    result = {'stage': 'R0', 'images': len(images), 'fixtures': len(fixtures['cases']),
              'prediction_comparison_has_differences': comparison['has_differences'],
              'inputs_changed_during_run': changed_inputs, 'code_changed_during_run': changed_code,
              'final_integrity_passed': integrity['passed'],
              'elapsed_seconds': round(time.monotonic()-started, 3), 'viewer': str(output_root / 'index.html')}
    write_metadata_atomic(output_root / 'completion.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=PROJECT_ROOT / 'configs/overlap_demo_baseline.json')
    parser.add_argument('--output-root', type=Path, required=True, help='New directory, resolved relative to current working directory. Never overwritten.')
    args = parser.parse_args()
    result = run_baseline(args.config, args.output_root)
    print(json.dumps(result, indent=2))
    if result['prediction_comparison_has_differences'] or not result['final_integrity_passed']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
