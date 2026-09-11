"""Check provenance at artifact completion, including input inventory changes."""
from pathlib import Path

from experiments.provenance import collect_input_manifest, sha256_file


def verify_snapshot(provenance: dict, project_root: Path) -> dict:
    initial = provenance['input_manifest']
    current = collect_input_manifest(Path(initial['input_root']), initial.get('limit'))
    old = {row['relative_image']: row['sha256'] for row in initial['images']}
    new = {row['relative_image']: row['sha256'] for row in current['images']}
    result = {
        'added_inputs': sorted(set(new)-set(old)),
        'missing_inputs': sorted(set(old)-set(new)),
        'changed_inputs': sorted(key for key in set(old)&set(new) if old[key] != new[key]),
    }
    for name in ['source_files', 'config_files']:
        result[f'changed_{name}'] = [row['path'] for row in provenance[name]
                                    if not (project_root / row['path']).is_file()
                                    or sha256_file(project_root / row['path']) != row['sha256']]
    result['passed'] = not any(result.values())
    return result
