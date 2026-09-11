"""Verify R3 source-coordinate exports and conditional ownership invariants."""
from __future__ import annotations

import argparse
from html.parser import HTMLParser
from itertools import combinations
import json
from pathlib import Path
import sys
from urllib.parse import unquote, urlsplit

SCP_ROOT = Path(__file__).resolve().parents[1]
if str(SCP_ROOT) not in sys.path:
    sys.path.insert(0, str(SCP_ROOT))

import cv2
import numpy as np
from PIL import Image
from experiments.provenance import sha256_file
from experiments.r1_adapter import restore_component_labels
from experiments.r3_instance_checks import check_instance
from experiments.run_r2 import load_checkpoint
from experiments.run_r3 import tree_hashes


def _binary(path: Path) -> np.ndarray:
    pixels = np.asarray(Image.open(path))
    if pixels.ndim != 2 or not np.isin(pixels, (0, 255)).all():
        raise ValueError(f'Not a binary mask: {path}')
    return pixels > 0


def _intersection(a, b):
    box = [max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])]
    return box if box[0] < box[2] and box[1] < box[3] else None


def _cut(mask, source_roi, target_roi):
    x0, y0, x1, y1 = target_roi
    return mask[y0-source_roi[1]:y1-source_roi[1], x0-source_roi[0]:x1-source_roi[0]]


def check_pairwise_ownership(directory: Path, payload: dict, mask_key: str = 'instance_mask') -> dict:
    """Check actual logical masks, including heads, against pairwise permissions."""
    permissions = {}
    for component in payload['components']:
        roi = component['source_roi_xyxy']
        for region in component['crossing_regions']:
            mask = _binary(directory/region['mask_file'])
            if mask.shape != (roi[3]-roi[1], roi[2]-roi[0]):
                raise ValueError('Crossing region dimensions disagree with source ROI')
            for pair in combinations(sorted(region['track_ids']), 2):
                permissions.setdefault(pair, []).append((roi, mask))
    items = {item['id']: (item['roi_xyxy'], _binary(directory/item['files'][mask_key])) for item in payload['instances']}
    forbidden = shared = 0
    details = []
    for a, b in combinations(sorted(items), 2):
        ar, am = items[a]; br, bm = items[b]
        overlap_roi = _intersection(ar, br)
        if overlap_roi is None:
            continue
        overlap = _cut(am, ar, overlap_roi) & _cut(bm, br, overlap_roi)
        if not overlap.any():
            continue
        allowed = np.zeros_like(overlap)
        for rr, rm in permissions.get((a, b), []):
            cut_roi = _intersection(rr, overlap_roi)
            if cut_roi:
                _cut(allowed, overlap_roi, cut_roi)[:] |= _cut(rm, rr, cut_roi)
        invalid = int((overlap & ~allowed).sum())
        shared += int(overlap.sum()); forbidden += invalid
        details.append({'instances': [a, b], 'shared_pair_pixels': int(overlap.sum()), 'forbidden_pair_pixels': invalid})
    return {'shared_pair_pixels': shared, 'forbidden_pair_pixels': forbidden, 'pairs': details}


class _Links(HTMLParser):
    def __init__(self):
        super().__init__(); self.links = []

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        self.links.extend(attributes[key] for key in ('href', 'src') if attributes.get(key))


def audit(root: Path) -> dict:
    root = root.resolve()
    errors, images = [], []
    counts = {'images': 0, 'instances': 0, 'null_choices': 0, 'local_links': 0,
              'forbidden_pair_pixels': 0, 'shared_pair_pixels': 0,
              'alternative_forbidden_pair_pixels': 0, 'head_tail_evidence_conflict_pixels': 0,
              'tail_growth_pixels_removed_at_foreign_heads': 0}
    try:
        completion = json.loads((root/'completion.json').read_text())
        integrity = json.loads((root/'integrity.json').read_text())
        provenance = json.loads((root/'provenance.json').read_text())
        config = json.loads((root/'config.json').read_text())
        fixtures = json.loads((root/'fixture_completion.json').read_text())
        if not completion['full_inventory_run'] or not integrity['passed']:
            errors.append('Incomplete inventory or failed final integrity')
        if fixtures['files'] != tree_hashes(root/'fixtures') or fixtures['report_sha256'] != sha256_file(root/'fixture_report.json'):
            errors.append('Fixture checkpoint changed')
        frozen = provenance['frozen_evidence_sha256']
        input_root = Path(provenance['input_manifest']['input_root'])
        for image in provenance['input_manifest']['images']:
            if sha256_file(input_root/image['relative_image']) != image['sha256']:
                errors.append(f'Raw source image changed: {image["relative_image"]}')
        for filename, digest in frozen.items():
            if not Path(filename).is_file() or sha256_file(Path(filename)) != digest:
                errors.append(f'Frozen evidence changed: {filename}')
        source_records = {}
        r1_paths, r2_paths = {}, {}
        for filename in frozen:
            path = Path(filename)
            if path.suffix == '.json' and path.parent.name == 'json' and path.parent.parent.name == 'predictions':
                record = json.loads(path.read_text()); source_records[record['relative_image']] = record
            if path.name == 'routes.json' and path.parent.parent.name == 'images':
                r1_paths[path.parent.name] = path
            if path.name == 'assignment.json' and path.parent.parent.name == 'images':
                r2_paths[path.parent.name] = path
        identities = set()
        for directory in sorted((root/'images').iterdir()):
            payload = json.loads((directory/'reconstruction.json').read_text())
            identity = payload['relative_image']
            if identity in identities:
                errors.append('Duplicate source image')
            identities.add(identity); counts['images'] += 1
            if load_checkpoint(directory, identity) is None:
                errors.append(f'Missing image checkpoint: {identity}')
            source = source_records[identity]
            r1_path, r2_path = r1_paths[directory.name], r2_paths[directory.name]
            if payload['r1_routes_sha256'] != sha256_file(r1_path) or payload['r2_assignment_sha256'] != sha256_file(r2_path):
                errors.append('Frozen route or assignment identity changed')
            r1, r2 = json.loads(r1_path.read_text()), json.loads(r2_path.read_text())
            if r1['relative_image'] != identity or r2['relative_image'] != identity:
                errors.append('Source and frozen assignment identities differ')
            rgb = np.asarray(Image.open(source['image']).convert('RGB'))
            head_mask = _binary(Path(source['outputs']['head_mask']))
            tail_mask = _binary(Path(source['outputs']['tail_mask']))
            tails = restore_component_labels(tail_mask, source['tails'])
            _, heads = cv2.connectedComponents(head_mask.astype(np.uint8), connectivity=8)
            if (payload['height'], payload['width']) != rgb.shape[:2]:
                errors.append('Image dimensions changed')
            by_id = {item['id']: item for item in r2['hypotheses']}
            chosen = r2['assignment']['selected_by_head']
            selected = {item: by_id[item] for item in chosen.values() if by_id[item]['termination'] != 'null'}
            nulls = {head: item for head, item in chosen.items() if by_id[item]['termination'] == 'null'}
            if {item['id'] for item in payload['instances']} != set(selected) or len(payload['instances']) != len(selected):
                errors.append('Exported instance identities differ from selected R2 non-null hypotheses')
            if {item['head_id']: item['selected_id'] for item in payload['null_choices']} != nulls or len(payload['null_choices']) != len(nulls):
                errors.append('Null-choice inventory differs from R2')
            components = {str(item['tail_id']): item for item in r1['components']}
            v2 = payload.get('schema_version') == 'scp.r3.reconstruction.v2'
            selected_heads = {int(item['head_id']): key for key, item in selected.items()}
            pin_counts = np.zeros(heads.shape, np.uint16)
            expected_conflicts = []
            for instance in payload['instances']:
                x0, y0, x1, y1 = instance['roi_xyxy']
                line = _binary(directory/instance['files']['centerline'])
                pin_counts[y0:y1, x0:x1] += line
                local_heads = heads[y0:y1, x0:x1]
                for foreign in sorted((set(np.unique(local_heads[line])) & set(selected_heads)) - {int(instance['head_id'])}):
                    yy, xx = np.nonzero(line & (local_heads == foreign))
                    expected_conflicts.append({'reason': 'head_tail_evidence_conflict',
                        'head_instance_id': selected_heads[foreign], 'tail_instance_id': instance['id'],
                        'source_pixels_xy': np.column_stack((xx+x0, yy+y0)).tolist(),
                        'default_owner': instance['id'], 'alternative_owner': selected_heads[foreign]})
            if v2:
                key = lambda row: (row['head_instance_id'], row['tail_instance_id'])
                if sorted(payload['head_tail_evidence_conflicts'], key=key) != sorted(expected_conflicts, key=key):
                    errors.append('Head/tail evidence conflict metadata differs from exported frozen centerlines')
                counts['head_tail_evidence_conflict_pixels'] += sum(len(row['source_pixels_xy']) for row in expected_conflicts)
            for instance in payload['instances']:
                measurements = check_instance(directory, instance, rgb, heads, tails, selected[instance['id']], components[str(instance['tail_id'])], **({'selected_centerline_counts': pin_counts, 'selected_head_ids': set(selected_heads)} if v2 else {}))
                if measurements['centerline_pixels'] != instance['measurements']['supported_centerline_pixels']:
                    errors.append('Reported centerline support count differs from exported mask')
                if any(measurements[key] != instance['measurements'][key] for key in ('tail_pixels', 'head_pixels')):
                    errors.append('Reported head/tail pixel count differs from exported masks')
                if v2:
                    for key in ('head_evidence_pixels', 'pinned_tail_pixels_in_foreign_head'):
                        if measurements[key] != instance['measurements'][key]:
                            errors.append('Reported evidence-conflict pixel count differs from masks')
                    expected_local = [row for row in expected_conflicts if instance['id'] in (row['head_instance_id'], row['tail_instance_id'])]
                    if sorted(instance['uncertainty']['head_tail_evidence_conflicts'], key=lambda row: (row['head_instance_id'], row['tail_instance_id'])) != sorted(expected_local, key=lambda row: (row['head_instance_id'], row['tail_instance_id'])):
                        errors.append('Per-instance uncertainty omits or alters source conflicts')
                    counts['tail_growth_pixels_removed_at_foreign_heads'] += instance['measurements']['tail_growth_pixels_removed_at_foreign_heads']
            ownership = check_pairwise_ownership(directory, payload)
            if v2:
                alternative = check_pairwise_ownership(directory, payload, 'head_priority_instance_mask')
                counts['alternative_forbidden_pair_pixels'] += alternative['forbidden_pair_pixels']
                if alternative['forbidden_pair_pixels']:
                    errors.append(f'Forbidden head-priority ownership: {identity}')
            counts['forbidden_pair_pixels'] += ownership['forbidden_pair_pixels']
            counts['shared_pair_pixels'] += ownership['shared_pair_pixels']
            counts['instances'] += len(selected); counts['null_choices'] += len(nulls)
            if ownership['forbidden_pair_pixels']:
                errors.append(f'Forbidden logical-mask ownership: {identity}')
            if any(c['core_diagnostics']['ordinary_duplicate_pixels'] for c in payload['components']):
                errors.append('Reconstruction reported forbidden ordinary duplicate pixels')
            if payload['source_comparisons']['production_statuses_changed'] is not False:
                errors.append('Production status preservation is not recorded')
            images.append({'relative_image': identity, 'instances': len(selected), **ownership})
        expected = {row['relative_image'] for row in provenance['input_manifest']['images']}
        if identities != expected or len(identities) != config['expected_image_count']:
            errors.append('Full source-image inventory differs from required manifest')
        for page in root.rglob('*.html'):
            parser = _Links(); parser.feed(page.read_text())
            for link in parser.links:
                parsed = urlsplit(link)
                if not parsed.scheme and parsed.path:
                    counts['local_links'] += 1
                    if not (page.parent/unquote(parsed.path)).is_file():
                        errors.append(f'Missing local viewer target: {link}')
    except (OSError, KeyError, ValueError, TypeError) as exc:
        errors.append(f'Invalid or missing required artifact: {exc}')
    return {'passed': not errors, 'errors': sorted(set(errors)), 'counts': counts, 'images': images,
            'scope': 'Source-coordinate exports, frozen selection/evidence preservation, and pairwise logical ownership; no biological accuracy claim.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_root', type=Path)
    args = parser.parse_args()
    result = audit(args.run_root)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['passed'] else 1)
