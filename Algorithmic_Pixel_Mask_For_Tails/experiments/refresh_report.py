"""Rebuild diagnostics from a completed prediction run without rerunning SCP."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

SCP_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SCP_ROOT.parent
if str(SCP_ROOT) not in sys.path:
    sys.path.insert(0, str(SCP_ROOT))

from experiments.benchmark import run_fixture_baseline
from experiments.configuration import baseline_arguments
from experiments.integrity import verify_snapshot
from experiments.provenance import build_provenance, compare_runs, dependency_metadata, sha256_file, write_metadata_atomic
from experiments.viewer import write_baseline_viewer


def refresh_report(run_root: Path, report_root: Path) -> dict:
    run_root, report_root = run_root.resolve(), report_root.resolve()
    original = json.loads((run_root/'provenance.json').read_text())
    completion = json.loads((run_root/'completion.json').read_text())
    if completion.get('inputs_changed_during_run') or completion.get('code_changed_during_run'):
        raise ValueError('Cannot reuse a run that changed during prediction execution')
    config = json.loads((run_root/'config.json').read_text())
    input_root = Path(original['input_manifest']['input_root'])
    if any(report_root == path or path in report_root.parents for path in [input_root.resolve(), (SCP_ROOT/'annotations').resolve(), (PROJECT_ROOT/config['reference_root']).resolve(), (run_root/'predictions').resolve()]):
        raise ValueError('Report must be separate from protected image and prediction directories')
    # Reporting helpers may evolve; the actual predictor, settings, inputs,
    # and environment must still correspond to the recorded prediction run.
    execution_snapshot = dict(original)
    execution_snapshot['source_files'] = [row for row in original['source_files']
                                          if row['path'] == 'Algorithmic_Pixel_Mask_For_Tails/algorithmic_tail_mask.py']
    if len(execution_snapshot['source_files']) != 1:
        raise ValueError('Missing predictor source fingerprint')
    if not verify_snapshot(execution_snapshot, PROJECT_ROOT)['passed']:
        raise ValueError('Predictor, input inventory, or configuration no longer matches the saved run')
    if dependency_metadata() != original['dependencies']:
        raise ValueError('Runtime dependencies differ from the recorded prediction run')
    reference_root = PROJECT_ROOT/config['reference_root']
    for path, fingerprint in original['reference_metadata_hashes'].items():
        if sha256_file(reference_root/path) != fingerprint:
            raise ValueError(f'Historical reference metadata changed: {path}')
    predictions = run_root/'predictions'
    actual_images = {json.loads(p.read_text())['relative_image'] for p in (predictions/'json').glob('*.json')}
    expected_images = {row['relative_image'] for row in original['input_manifest']['images']}
    if actual_images != expected_images:
        raise ValueError('Saved prediction image inventory is incomplete or unexpected')
    report_root.mkdir(parents=True, exist_ok=False)
    _, args = baseline_arguments(config, input_root, predictions)
    code = [SCP_ROOT/'algorithmic_tail_mask.py', *sorted((SCP_ROOT/'experiments').glob('*.py')),
            *sorted((SCP_ROOT/'tests/fixtures').glob('*.py'))]
    provenance = build_provenance(input_root=input_root, project_root=PROJECT_ROOT, code_paths=code,
                                  config_paths=[run_root/'config.json', run_root/'demo_manifest.json'],
                                  namespace=args, command=list(sys.argv))
    provenance['scope'] = 'Reporting/fixture rerun only. Existing full-image predictions reused without modification.'
    provenance['prediction_provenance'] = str(run_root/'provenance.json')
    provenance['prediction_provenance_sha256'] = sha256_file(run_root/'provenance.json')
    provenance['prediction_metadata_sha256'] = {p.relative_to(predictions).as_posix(): sha256_file(p)
                                               for p in [predictions/'summary.json', *sorted((predictions/'json').glob('*.json'))]}
    write_metadata_atomic(report_root/'provenance.json', provenance)
    write_metadata_atomic(report_root/'input_manifest.json', original['input_manifest'])
    comparison = compare_runs(reference_root, predictions)
    write_metadata_atomic(report_root/'comparison.json', comparison)
    fixtures = run_fixture_baseline(report_root, args, config['fixture_seed'])
    demo = json.loads((run_root/'demo_manifest.json').read_text())
    write_baseline_viewer(report_root, input_root, reference_root, predictions, demo, comparison, fixtures)
    integrity = verify_snapshot(provenance, PROJECT_ROOT)
    integrity['changed_prediction_metadata'] = [path for path, fingerprint in provenance['prediction_metadata_sha256'].items()
                                               if sha256_file(predictions/path) != fingerprint]
    integrity['passed'] = integrity['passed'] and not integrity['changed_prediction_metadata']
    write_metadata_atomic(report_root/'integrity.json', integrity)
    if not integrity['passed']:
        raise RuntimeError('Report integrity check failed')
    result = {'viewer': str(report_root/'index.html'), 'images': len(actual_images), 'fixtures': len(fixtures['cases']),
              'historical_comparison_has_differences': comparison['has_differences'], 'integrity_passed': True}
    write_metadata_atomic(report_root/'completion.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--report-root', type=Path, required=True, help='New directory; never overwrite an existing report')
    args = parser.parse_args()
    print(json.dumps(refresh_report(args.run_root, args.report_root), indent=2))
