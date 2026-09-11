"""Run the experimental graph on frozen SCP evidence, retaining baseline routes.

Coordinates in component records are ROI-local XY with an explicit source
offset. This adapter does not change any production assignment or status.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import algorithmic_tail_mask as atm
from graph_construction import build_segment_graph
from path_hypotheses import search_hypotheses


def restore_component_labels(mask: np.ndarray, records: list[dict]) -> np.ndarray:
    """Match saved component identities by exact bounding box and area.

    OpenCV label ordering can differ with image dimensions/backend. An explicit
    check avoids silently associating a saved head or tail with another object.
    """
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8), connectivity=8)
    by_geometry = {}
    for label in range(1, count):
        key = tuple(int(v) for v in stats[label])
        by_geometry.setdefault(key, []).append(label)
    output = np.zeros(mask.shape, np.int32)
    used = set()
    for row in records:
        key = (*row['bbox_xywh'], row['area_px'])
        matches = by_geometry.get(key, [])
        if len(matches) != 1 or matches[0] in used:
            raise ValueError(f"Cannot uniquely restore component {row['component_id']}: {key}")
        used.add(matches[0])
        output[labels == matches[0]] = int(row['component_id'])
    if len(used) != count - 1:
        raise ValueError('Saved mask contains components absent from metadata')
    return output


def source_boundary_graph(graph, offset_xy, source_shape):
    """A component ROI border is not an image-boundary truncation."""
    x0, y0 = offset_xy
    height, width = source_shape
    nodes = {}
    for key, node in graph.nodes.items():
        x, y = node.xy
        border = x + x0 <= 1 or y + y0 <= 1 or x + x0 >= width - 2 or y + y0 >= height - 2
        terminal = len(graph.outgoing(key)) <= 1 and key not in graph.anchors.values()
        kind = 'boundary' if terminal and border else node.kind
        if kind in {'boundary', 'boundary_port'} and not border:
            kind = 'endpoint' if terminal else 'ordinary'
        nodes[key] = replace(node, kind=kind)
    return replace(graph, nodes=nodes)


def baseline_record(row: dict, offset_xy: tuple[int, int], predictions: Path) -> dict:
    path = Path(row['skeleton_mask']).resolve()
    if predictions.resolve() not in path.parents:
        raise ValueError('Baseline skeleton must belong to the frozen prediction run')
    mask = np.asarray(Image.open(path).convert('L')) > 0
    bx0, by0, bx1, by1 = row['bbox_xyxy']
    if mask.shape != (by1 - by0, bx1 - bx0):
        raise ValueError('Baseline skeleton dimensions disagree with its saved crop')
    yy, xx = np.nonzero(mask)
    x0, y0 = offset_xy
    return {'crop_id': row['crop_id'], 'head_id': str(row['head_label_id']),
            'status': row['candidate_status'], 'skeleton_file': str(path),
            'pixels_xy': np.column_stack((xx + bx0 - x0, yy + by0 - y0)).tolist(),
            'retention': 'Exact saved raster; unranked alongside experimental alternatives.'}


def analyze_image(record: dict, predictions: Path, config: dict) -> dict:
    rgb = np.asarray(Image.open(record['image']).convert('RGB'))
    stem = Path(record['outputs']['tail_mask']).name.removesuffix('_tail_mask.png')
    tail_mask = np.asarray(Image.open(predictions/'masks'/f'{stem}_tail_mask.png').convert('L'))
    head_mask = np.asarray(Image.open(predictions/'masks'/f'{stem}_head_mask.png').convert('L'))
    tails = restore_component_labels(tail_mask, record['tails'])
    # Production assignment labels heads with connectedComponents on the saved
    # head mask, independently of detect_heads' metadata ordering.
    _, heads, head_stats, _ = cv2.connectedComponentsWithStats((head_mask > 0).astype(np.uint8), connectivity=8)
    if len(head_stats) - 1 != len(record['heads']):
        raise ValueError('Head component inventory differs from frozen metadata')
    height, width = tail_mask.shape
    scale = atm.image_scale_factor(width, height)
    contact = max(1, int(round(config['head_contact_radius'] * scale)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * contact + 1,) * 2)
    candidates = record['head_connected_postprocessing']['crop_candidates_detail']
    components = []
    for tail in record['tails']:
        tail_id = tail['component_id']
        x, y, w, h = tail['bbox_xywh']
        padding = max(2, int(round(contact + 10 * scale)))
        x0, y0, x1, y1 = max(0, x-padding), max(0, y-padding), min(width, x+w+padding), min(height, y+h+padding)
        component = tails[y0:y1, x0:x1] == tail_id
        head_roi = heads[y0:y1, x0:x1]
        touching = sorted(int(i) for i in np.unique(head_roi[cv2.dilate(component.astype(np.uint8), kernel) > 0]) if i)
        skeleton = atm.skeletonize_binary(component.astype(np.uint8)*255)
        coords = np.argwhere(skeleton > 0)
        anchors, excluded = {}, []
        existing = [row for row in candidates if row['tail_id'] == tail_id]
        existing_by_head = {row['head_label_id']: row for row in existing}
        areas = {i: int(head_stats[i, cv2.CC_STAT_AREA]) for i in touching}
        _, min_area = atm.head_area_threshold_for_shared_tail(touching, head_roi, scale, head_areas=areas)
        for head_id in touching:
            row = existing_by_head.get(head_id)
            saved_anchor = row.get('anchor_xy') if row else None
            if saved_anchor is not None:
                anchors[str(head_id)] = (saved_anchor[0]-x0, saved_anchor[1]-y0)
                continue
            idx, distance, xy = atm.nearest_skeleton_anchor(coords, head_roi == head_id)
            if areas[head_id] < min_area or idx is None or distance > max(contact+10*scale, 18*scale):
                excluded.append({'head_id': str(head_id), 'reason': 'baseline_anchor_eligibility',
                                 'area_px': areas[head_id], 'distance_px': distance})
            else:
                anchors[str(head_id)] = tuple(xy)
        # This is existing RGB evidence sampled on already-positive SCP pixels,
        # not a new faint-pixel detector (that belongs to R4).
        local_rgb = rgb[y0:y1, x0:x1]
        hsv = cv2.cvtColor(local_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
        evidence = np.clip(hsv[:, :, 1] / 100., 0, 1) * component
        graph = build_segment_graph(skeleton, tail_mask=component, anchors_xy=anchors, evidence=evidence)
        graph = source_boundary_graph(graph, (x0, y0), (height, width))
        searches = {key: search_hypotheses(graph, key, k=config['diagnostic_k'],
                                          max_expansions=config['max_expansions']) for key in sorted(graph.anchors)}
        main_searches = {key: search_hypotheses(graph, key, k=config['k'],
                                              max_expansions=config['max_expansions']) for key in sorted(graph.anchors)}
        components.append({'tail_id': tail_id, 'roi_xyxy': [x0, y0, x1, y1],
                           'touching_heads': touching, 'excluded_anchors': excluded,
                           'graph': graph.to_dict(), 'searches': main_searches,
                           'diagnostic_searches': searches,
                           'baseline': [baseline_record(row, (x0, y0), predictions) for row in existing],
                           'production_statuses_changed': False})
    return {'relative_image': record['relative_image'], 'width': width, 'height': height,
            'scope': 'R1 proposals from frozen SCP masks and baseline head-anchor rules; no joint assignment.',
            'components': components, 'baseline_candidates': len(candidates)}
