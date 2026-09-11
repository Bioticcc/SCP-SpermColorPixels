"""Head-boundary attachment proposals on an unchanged source-coordinate graph."""
from __future__ import annotations

import math
import cv2
import numpy as np
from tail_evidence.gaps import _sample


def local_segment_direction(points_xy, radius_px):
    """Unit displacement at a fixed local arclength, independent of vertex density."""
    points = np.asarray(points_xy, dtype=float)
    if len(points) < 2:
        return None
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    target = min(max(3.0, 3.0 * radius_px), float(lengths.sum()))
    remaining = target
    point = points[0]
    for start, end, length in zip(points[:-1], points[1:], lengths):
        if length <= 0:
            continue
        if remaining <= length:
            point = start + (end - start) * (remaining / length)
            break
        remaining -= length
        point = end
    delta = point - points[0]
    norm = float(np.linalg.norm(delta))
    return (delta / norm).tolist() if norm > 0 else None


def propose_head_anchors(graph, head_mask, score, baseline_node_ids,
                         max_distance_px=18, max_additional=3):
    """Keep baseline nodes and rank nearby, evidence-supported endpoint alternatives.

    Head pixels contribute neither color support nor unsupported distance to a
    neck proposal. Tangent alignment is a soft cost; hooked/abnormal geometry is
    retained. Inside-head and direct-touch locations have unknown alignment.
    """
    mask = np.asarray(head_mask, dtype=bool)
    field = np.asarray(score)
    if (mask.ndim != 2 or not mask.size or field.shape != mask.shape
            or field.dtype != np.float32 or not np.isfinite(field).all()
            or np.any((field < 0) | (field > 1))):
        raise ValueError('invalid masks')
    if (type(max_additional) is not int or max_additional < 0
            or type(max_distance_px) not in (int, float)
            or not math.isfinite(max_distance_px) or max_distance_px < 0):
        raise ValueError('invalid limits')
    baseline = list(dict.fromkeys(str(x) for x in baseline_node_ids))
    if any(key not in graph.nodes for key in baseline):
        raise ValueError('unknown baseline node')
    if not mask.any():
        return []
    h, w = mask.shape
    eroded = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8),
                       borderType=cv2.BORDER_CONSTANT, borderValue=0)
    yy, xx = np.nonzero(mask & ~eroded.astype(bool))
    adjacency = {key: [] for key in graph.nodes}
    for key, segment in graph.segments.items():
        adjacency[segment.start].append((key, True))
        adjacency[segment.end].append((key, False))

    def valid_position(node):
        x, y = node.xy
        return math.isfinite(x) and math.isfinite(y) and 0 <= x <= w-1 and 0 <= y <= h-1

    def record(node, is_baseline):
        if not valid_position(node):
            raise ValueError('attachment node outside source image')
        x, y = node.xy
        index = int(np.argmin((xx - x)**2 + (yy - y)**2))
        boundary = np.array([xx[index], yy[index]], dtype=float)
        delta = np.array([x, y]) - boundary
        distance = float(np.linalg.norm(delta))
        inside = bool(mask[int(round(y)), int(round(x))])
        directions = []
        for key, forward in adjacency[node.id]:
            segment = graph.segments[key]
            radii = segment.radii
            radius = float(radii[0 if forward else -1]) if radii else 1.0
            direction = local_segment_direction(graph.points(key, forward), radius)
            if direction is not None:
                directions.append(direction)
        direction = directions[0] if len(directions) == 1 else None
        alignment = (float(np.dot(delta / distance, direction))
                     if distance > 0 and not inside and direction is not None else None)
        n = max(1, int(math.ceil(distance * 2)))
        samples = boundary + delta * ((np.arange(n) + 0.5) / n)[:, None]
        rounded = np.rint(samples).astype(int)
        outside = ~mask[rounded[:, 1], rounded[:, 0]]
        values = _sample(field, samples)
        support = float(values[outside].mean()) if outside.any() else None
        step = distance / n
        run = longest = 0.0
        for is_outside, value in zip(outside, values):
            run = run + step if is_outside and value < 0.15 else 0.0
            longest = max(longest, run)
        rank = (distance / max(max_distance_px, 1)
                + (0.25 * (1-alignment) if alignment is not None else 0.25)
                + 1 - (support if support is not None else 0))
        return {'node_id': node.id, 'point_xy': [x, y],
                'nearest_head_boundary_xy': boundary.tolist(), 'distance_px': distance,
                'tail_direction': direction, 'tangent_options': directions,
                'alignment': alignment, 'mean_color_support': support,
                'longest_unsupported_run_px': longest,
                'uncertainty': 'inside_head_or_direct_touch' if inside or distance == 0 else None,
                'score': rank, 'baseline': is_baseline}

    output = [record(graph.nodes[key], True) for key in baseline]
    extras = []
    xmin, xmax = xx.min()-max_distance_px, xx.max()+max_distance_px
    ymin, ymax = yy.min()-max_distance_px, yy.max()+max_distance_px
    for key, node in graph.nodes.items():
        if (key in baseline or node.kind in ('boundary', 'boundary_port')
                or len(adjacency[key]) != 1 or not valid_position(node)):
            continue
        x, y = node.xy
        if not (xmin <= x <= xmax and ymin <= y <= ymax):
            continue
        item = record(node, False)
        support = item['mean_color_support']
        if (item['distance_px'] <= max_distance_px
                and (item['distance_px'] == 0 or (support is not None and support >= 0.25
                     and item['longest_unsupported_run_px'] <= 2.0 + 1e-9))):
            extras.append(item)
    extras.sort(key=lambda item: (item['score'], item['node_id']))
    return output + extras[:max_additional]
