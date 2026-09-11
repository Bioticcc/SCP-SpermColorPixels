"""Evidence-supported local-width masks around selected ordered centerlines."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class Track:
    id: str
    head_id: str
    points_xy: Any
    ordered: bool = True


@dataclass(frozen=True)
class CrossingRegion:
    id: str
    mask: Any
    track_ids: tuple[str, ...]


_DEFAULTS = {"min_radius_px": 0.5, "crossing_exclusion_px": 2, "tie_epsilon": 1e-9}


def _points(track: Track) -> np.ndarray:
    value = np.asarray(track.points_xy, dtype=float)
    if value.size == 0:
        return np.empty((0, 2), dtype=float)
    if value.ndim != 2 or value.shape[1] != 2 or not np.isfinite(value).all():
        raise ValueError(f"track {track.id!r} points_xy must be finite Nx2")
    return value


def _dense(points: np.ndarray, ordered: bool) -> np.ndarray:
    if not len(points): return np.empty((0, 2), dtype=int)
    if not ordered: return np.unique(np.rint(points).astype(int), axis=0)
    points = np.rint(points).astype(int)
    result = [points[0]]
    for a, b in zip(points, points[1:]):
        if np.max(np.abs(b-a)) <= 1:
            if np.any(a != b): result.append(b)
            continue
        # Use the same LINE_8 raster convention as the evidence renderer. Linear
        # interpolation plus rounding loses every other tie pixel on slope 1/2.
        offset = np.minimum(a, b)
        size = np.abs(b-a)+1
        raster = np.zeros((size[1], size[0]), np.uint8)
        cv2.line(raster, tuple(a-offset), tuple(b-offset), 1, 1, lineType=cv2.LINE_8)
        yy, xx = np.nonzero(raster)
        part = np.column_stack((xx, yy)) + offset
        order = np.argsort((part-a) @ (b-a), kind='stable')
        result.extend(part[order][1:])
    return np.asarray(result, dtype=int)


def reconstruct_tracks(evidence: Any, tracks: list[Track], crossings: list[CrossingRegion] | None = None, config: dict | None = None) -> dict:
    """Grow supported tubes; choose ordinary owners and pairwise-compatible sharing.

    Width is sampled outside declared junction neighborhoods. Ordered paths use
    arclength interpolation; unordered controls use nearest supported samples.
    Where no unexcluded width evidence exists, the conservative radius floor is
    explicitly marked unknown. It is never replaced by the union's crossing width.
    """
    if config is not None and (not isinstance(config, dict) or set(config) - set(_DEFAULTS)):
        raise ValueError("unknown reconstruction configuration")
    settings = {**_DEFAULTS, **(config or {})}
    for key in ("min_radius_px", "tie_epsilon"):
        value = settings[key]
        if isinstance(value, bool) or not isinstance(value, (int, float, np.number)) or not math.isfinite(value):
            raise ValueError("invalid reconstruction configuration")
        settings[key] = float(value)
    exclusion_px = settings["crossing_exclusion_px"]
    if (isinstance(exclusion_px, bool) or not isinstance(exclusion_px, (int, np.integer)) or exclusion_px < 0
            or settings["min_radius_px"] <= 0 or settings["tie_epsilon"] < 0):
        raise ValueError("invalid reconstruction configuration")
    ev = np.asarray(evidence, dtype=bool)
    if ev.ndim != 2 or not all(ev.shape):
        raise ValueError("evidence must be a nonempty 2D mask")
    ids = [track.id for track in tracks]
    if any(not isinstance(item, str) or not item for item in ids) or len(set(ids)) != len(ids):
        raise ValueError("track ids must be unique non-empty strings")
    ids = sorted(ids)
    h, w = ev.shape
    zero = np.zeros_like(ev)
    allowed: dict[frozenset[str], np.ndarray] = {}
    exclusion = np.zeros_like(ev)
    region_ids = set()
    for region in crossings or []:
        mask = np.asarray(region.mask, dtype=bool)
        if (mask.shape != ev.shape or not region.id or region.id in region_ids or not region.track_ids
                or len(set(region.track_ids)) != len(region.track_ids) or not set(region.track_ids) <= set(ids)):
            raise ValueError("invalid crossing region")
        region_ids.add(region.id)
        kernel = np.ones((2 * exclusion_px + 1,) * 2, np.uint8)
        exclusion |= cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)
        for a in region.track_ids:
            for b in region.track_ids:
                if a < b:
                    pair = frozenset((a, b))
                    allowed[pair] = allowed.get(pair, zero) | mask
    # Explicit zero boundary prevents the huge OpenCV sentinel on all-true ROIs.
    distance = cv2.distanceTransform(np.pad(ev.astype(np.uint8), 1), cv2.DIST_L2, 5)[1:-1, 1:-1]
    internal_distance = cv2.distanceTransform(ev.astype(np.uint8), cv2.DIST_L2, 5)
    # Image borders can truncate a thick tail. They must not become positive
    # evidence for a smaller width; extrapolate from available interior samples.
    border_truncated = internal_distance > distance + 1e-6
    centerlines, radii, scores, radius_points = {}, {}, {}, {}
    unsupported, outside, interpolated, unknown, unknown_crossing, truncated_width = {}, {}, {}, {}, {}, {}
    for track in tracks:
        path = _dense(_points(track), track.ordered)
        inside = ((path[:, 0] >= 0) & (path[:, 0] < w) & (path[:, 1] >= 0) & (path[:, 1] < h)) if len(path) else np.zeros(0, bool)
        valid = path[inside]
        outside[track.id] = int((~inside).sum())
        raw = np.zeros_like(ev)
        if len(valid):
            raw[valid[:, 1], valid[:, 0]] = True
        centerlines[track.id] = raw & ev
        unsupported[track.id] = int((raw & ~ev).sum())
        known = np.full(len(valid), settings["min_radius_px"], dtype=float)
        supported = ev[valid[:, 1], valid[:, 0]] if len(valid) else np.zeros(0, bool)
        good = supported & ~(exclusion | border_truncated)[valid[:, 1], valid[:, 0]] if len(valid) else np.zeros(0, bool)
        # DT measures distance to an outside pixel CENTER. A half-pixel boundary
        # convention preserves the discrete width of an odd-width straight line.
        values = np.maximum(0.5, distance[valid[:, 1], valid[:, 0]] - 0.5) if len(valid) else np.zeros(0)
        unknown[track.id] = int((supported & ~good).sum()) if not good.any() else 0
        unknown_crossing[track.id] = int((supported & exclusion[valid[:, 1], valid[:, 0]]).sum()) if len(valid) and not good.any() else 0
        truncated_width[track.id] = int((supported & border_truncated[valid[:, 1], valid[:, 0]]).sum()) if len(valid) else 0
        interpolated[track.id] = int((supported & ~good).sum()) if good.any() else 0
        if good.any():
            if track.ordered:
                arc = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))))[inside]
                known = np.interp(arc, arc[good], values[good])
            else:
                known[good] = values[good]
                good_points = valid[good]
                for index in np.flatnonzero(~good):
                    nearest = np.argmin(np.square(good_points - valid[index]).sum(axis=1))
                    known[index] = values[good][nearest]
        known = np.maximum(settings["min_radius_px"], known)
        radii[track.id] = known.tolist()
        radius_points[track.id] = valid.tolist()
        score = np.full(ev.shape, np.inf, dtype=float)
        # Dense track samples bound disk work by local width; unsupported samples
        # do not seed tubes across missing evidence.
        for (x, y), radius in zip(valid[supported], known[supported]):
            r = max(1, int(math.ceil(radius)))
            ya, yb, xa, xb = max(0, y-r), min(h, y+r+1), max(0, x-r), min(w, x+r+1)
            yy, xx = np.ogrid[ya:yb, xa:xb]
            local = np.hypot(xx-x, yy-y) / radius
            np.minimum(score[ya:yb, xa:xb], local, out=score[ya:yb, xa:xb])
        scores[track.id] = score
    for index, a in enumerate(ids):
        for b in ids[index+1:]:
            if np.any(centerlines[a] & centerlines[b] & ~allowed.get(frozenset((a, b)), zero)):
                raise ValueError(f"tracks {a!r} and {b!r} duplicate a centerline outside a crossing region")
    # Pin centers first. Allocate ordinary pixels to the nearest normalized tube,
    # using IDs only for numerical ties, independently of input track order.
    masks = {item: centerlines[item].copy() for item in ids}
    claimed = np.logical_or.reduce(list(masks.values())) if ids else zero.copy()
    free = ev & ~claimed
    best = np.minimum.reduce([scores[item] for item in ids]) if ids else np.full(ev.shape, np.inf)
    for item in ids:
        take = free & (scores[item] <= 1.0) & (scores[item] <= best + settings["tie_epsilon"])
        masks[item] |= take
        free &= ~take
    # Grow additional owners only if compatible with EVERY existing owner.
    # This preserves full unequal-width tubes at a two-track crossing without
    # accidentally authorizing A/C sharing through separate A/B and B/C regions.
    for item in ids:
        eligible = ev & (scores[item] <= 1.0) & ~masks[item]
        for other in ids:
            if other != item:
                eligible &= ~masks[other] | allowed.get(frozenset((item, other)), zero)
        masks[item] |= eligible
    ownership = np.zeros_like(ev, dtype=np.int32)
    for mask in masks.values():
        ownership += mask
    forbidden = np.zeros_like(ev)
    for index, a in enumerate(ids):
        for b in ids[index+1:]:
            forbidden |= masks[a] & masks[b] & ~allowed.get(frozenset((a, b)), zero)
    if forbidden.any():
        raise RuntimeError("reconstruction introduced forbidden pairwise ownership")
    return {"masks": masks, "centerlines": centerlines, "radii": radii, "radius_points_xy": radius_points,
            "diagnostics": {"evidence_pixels": int(ev.sum()), "unsupported_centerline_pixels": unsupported,
                "out_of_bounds_centerline_samples": outside, "width_unknown_samples": unknown,
                "width_unknown_crossing_samples": unknown_crossing, "border_truncated_width_samples": truncated_width,
                "width_interpolated_samples": interpolated,
                "shared_pixels": int((ownership > 1).sum()), "ordinary_duplicate_pixels": int(forbidden.sum()),
                "crossing_exclusion_pixels": int(exclusion.sum()),
                "ownership_rule": "Pinned centers; nearest normalized tube; lexicographic pairwise-compatible extra owners inside declared regions.",
                "width_interpolation": "DT minus half a pixel; arclength for ordered tracks; nearest supported sample for unordered controls; border-truncated samples excluded; radius floor if no width sample is available.",
                "orientation_rationale": "Local tubes follow supplied centerlines; no RGB orientation inference is claimed."}}
