"""Deterministic, image-local segment graphs for SCP skeletons.

This package deliberately contains representation and extraction only.  It does
not choose a biological continuation or assign a path to a head.
"""

from .segment_graph import (
    Node,
    Segment,
    SegmentGraph,
    build_segment_graph,
    from_explicit_graph,
)

__all__ = [
    "Node",
    "Segment",
    "SegmentGraph",
    "build_segment_graph",
    "from_explicit_graph",
]
