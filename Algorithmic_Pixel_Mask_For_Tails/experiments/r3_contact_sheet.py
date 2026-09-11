"""Fixed real-image panel for inspecting selected centerlines and logical masks."""
import json
from pathlib import Path

from PIL import Image, ImageDraw

from experiments.r2_contact_sheet import _overlay, PALETTE


def write_contact_sheet(output: Path, r1_root: Path, r2_root: Path, summaries: list[dict], demo: dict) -> Path:
    inventory = {row['relative_image']: row for row in summaries}
    rows = []
    for example in demo['images']:
        summary = inventory.get(example['relative_image'])
        if summary is None:
            continue
        folder = summary['directory']
        r1 = json.loads((r1_root/'images'/folder/'routes.json').read_text())
        r2 = json.loads((r2_root/'images'/folder/'assignment.json').read_text())
        r3 = json.loads((output/'images'/folder/'reconstruction.json').read_text())
        original = Image.open(r1_root/'images'/folder/'original.jpg').convert('RGB')
        original.thumbnail((350, 265))
        baseline = Image.open(r1_root/'images'/folder/'baseline.jpg').convert('RGB').resize(original.size)
        selected = _overlay(original, r1, r2['hypotheses'], r2['assignment']['selected_by_head'])
        colors = {head: PALETTE[index % len(PALETTE)] for index, head in enumerate(sorted(r2['assignment']['selected_by_head']))}
        composite = original.convert('RGBA')
        for index, instance in enumerate(r3['instances']):
            mask = Image.open(output/'images'/folder/instance['files']['instance_mask']).convert('L')
            x0, y0, x1, y1 = instance['roi_xyxy']
            left, right = round(x0/r3['width']*original.width), round(x1/r3['width']*original.width)
            top, bottom = round(y0/r3['height']*original.height), round(y1/r3['height']*original.height)
            if left == right or top == bottom:
                continue
            alpha = mask.resize((right-left, bottom-top), Image.Resampling.BOX).point(lambda v: round(v*0.8))
            layer = Image.new('RGBA', alpha.size, colors[instance['head_id']])
            layer.putalpha(alpha)
            composite.alpha_composite(layer, (left, top))
        row = Image.new('RGB', (1400, original.height+48), 'white')
        draw = ImageDraw.Draw(row)
        draw.text((5, 3), f'{example["magnification"]} {Path(example["relative_image"]).name} | {len(r3["instances"])} selected non-null proposals', fill='black')
        for index, (label, image) in enumerate(zip(('Original', 'Frozen baseline', 'Selected R2 tracks', 'R3 logical masks'), (original, baseline, selected, composite.convert('RGB')))):
            draw.text((index*350+5, 24), label, fill='black')
            row.paste(image, (index*350, 48))
        rows.append(row)
    result = Image.new('RGB', (1400, sum(row.height for row in rows)+32), 'white')
    ImageDraw.Draw(result).text((5, 5), 'Fixed panel. Reconstruction preserves selected-path limitations; these are unvalidated instance proposals.', fill='black')
    y = 32
    for row in rows:
        result.paste(row, (0, y)); y += row.height
    path = output/'fixed_panel_contact_sheet.png'
    if path.exists():
        raise FileExistsError(path)
    result.save(path)
    return path
