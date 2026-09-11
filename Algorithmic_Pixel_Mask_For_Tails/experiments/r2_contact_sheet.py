"""Measured centerline overlays for inspecting R2 without a browser runtime."""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw

PALETTE = ('#00b9e8', '#ee5090', '#75cf36', '#ad79ff', '#edb628', '#ee6a25')


def _overlay(original: Image.Image, record: dict, hypotheses: list[dict], selected: dict) -> Image.Image:
    result = original.copy()
    draw = ImageDraw.Draw(result)
    sx, sy = result.width/record['width'], result.height/record['height']
    by_id = {item['id']: item for item in hypotheses}
    for index, head in enumerate(sorted(selected)):
        item = by_id[selected[head]]
        metadata = item['metadata']
        roi = metadata.get('roi_xyxy') or [0, 0]
        color = PALETTE[index % len(PALETTE)]
        ordered = metadata.get('points_xy_ordered', [])
        points = [((x+roi[0])*sx, (y+roi[1])*sy) for x, y in ordered]
        if len(points) > 1:
            draw.line(points, fill=color, width=2)
        elif points:
            draw.point(points, fill=color)
        for x, y in metadata.get('pixels_xy_unordered', []):
            draw.point(((x+roi[0])*sx, (y+roi[1])*sy), fill=color)
    return result


def write_contact_sheet(output: Path, r1_root: Path, summaries: list[dict], demo: dict) -> Path:
    """Use fixed R0 panel order and original-image coordinates; no invented pixels."""
    selected = {row['relative_image']: row for row in summaries}
    rows = []
    cell_width = 350
    for example in demo['images']:
        if example['relative_image'] not in selected:
            continue
        summary = selected[example['relative_image']]
        directory = summary['directory']
        record = json.loads((r1_root/'images'/directory/'routes.json').read_text())
        payload = json.loads((output/'images'/directory/'assignment.json').read_text())
        original = Image.open(r1_root/'images'/directory/'original.jpg').convert('RGB')
        original.thumbnail((cell_width, 265))
        baseline = Image.open(r1_root/'images'/directory/'baseline.jpg').convert('RGB').resize(original.size)
        assignment = payload['assignment']
        panels = [original, baseline,
                  _overlay(original, record, payload['hypotheses'], assignment['independent_by_head']),
                  _overlay(original, record, payload['hypotheses'], assignment['selected_by_head'])]
        row = Image.new('RGB', (cell_width*4, original.height+48), 'white')
        draw = ImageDraw.Draw(row)
        title = f'{example["magnification"]} {Path(example["relative_image"]).name} | {summary["changed_from_unary"]} choices changed by constraints'
        draw.text((5, 3), title, fill='black')
        for index, (label, panel) in enumerate(zip(('Original', 'Frozen baseline overlay', 'R2 unary independent', 'R2 joint proposals'), panels)):
            draw.text((index*cell_width+5, 24), label, fill='black')
            row.paste(panel, (index*cell_width, 48))
        rows.append(row)
    height = sum(row.height for row in rows) + 32
    sheet = Image.new('RGB', (cell_width*4, height), 'white')
    draw = ImageDraw.Draw(sheet)
    draw.text((5, 5), 'Fixed panel: changed proposals, not measured biological accuracy. Colors identify heads within each image.', fill='black')
    y = 32
    for row in rows:
        sheet.paste(row, (0, y))
        y += row.height
    path = output/'fixed_panel_contact_sheet.png'
    if path.exists():
        raise FileExistsError(path)
    sheet.save(path)
    return path
