"""Represent head/tail evidence disagreement without altering frozen evidence."""
import numpy as np


def compose_head_and_tail(tail, centerline, head_evidence, foreign_heads, foreign_centerlines):
    arrays = [np.asarray(value, dtype=bool) for value in (tail, centerline, head_evidence, foreign_heads, foreign_centerlines)]
    tail, centerline, head_evidence, foreign_heads, foreign_centerlines = arrays
    if any(value.shape != tail.shape for value in arrays) or tail.ndim != 2:
        raise ValueError('Head/tail ownership masks must have one 2D shape')
    if np.any(centerline & ~tail):
        raise ValueError('Supported selected centerline is absent before head composition')
    trimmed = tail & foreign_heads & ~centerline
    deferred = head_evidence & foreign_centerlines
    tail_result = tail & ~trimmed
    head_result = head_evidence & ~deferred
    uncertainty = deferred | (centerline & foreign_heads)
    return {'tail': tail_result, 'head': head_result, 'head_evidence': head_evidence.copy(),
            'uncertainty': uncertainty,
            'head_priority_instance': head_evidence | (tail_result & ~foreign_heads),
            'growth_pixels_removed': int(trimmed.sum()), 'head_pixels_deferred': int(deferred.sum()),
            'pinned_tail_pixels_in_foreign_head': int((centerline & foreign_heads).sum())}
