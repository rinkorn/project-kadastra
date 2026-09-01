"""Per-cell water-land share for the cell valuation layer (ADR-0029
addendum).

The Слой 1 grid covers the full agglomeration bbox, water bodies
included — so the EBM price appears on the Volga, where no land object
can exist. This module measures how much of each hexagon's area is
water and exposes it twice:

- ``cell_water_share`` (float 0..1) — the raw area fraction. Kept as
  data so the display threshold can be revisited without a rebuild.
- ``on_water`` (bool) — ``share >= ON_WATER_SHARE_THRESHOLD`` (0.5,
  i.e. the cell is mostly water and cannot host a land object). A
  display-domain marker, NOT a model input.

Shoreline cells with a minority water share stay untouched: a price
there is meaningful (embankments, waterfront districts), and an
any-intersection criterion would erase them.

Area ratios are computed in WGS84 degrees. For a res-10 cell
(~15 000 m²) the latitude scale factor is effectively constant across
the cell, so the degree-area ratio matches the true area ratio closely
enough for a 0.5 threshold.
"""

from __future__ import annotations

from collections import defaultdict

import h3
import numpy as np
import shapely
from shapely import STRtree
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry

# A cell mostly covered by water cannot host a land object — its price
# is meaningless on the map (ADR-0029 addendum).
ON_WATER_SHARE_THRESHOLD = 0.5


def _cell_polygon(h3_index: str) -> Polygon:
    """H3 cell boundary as a WGS84 shapely Polygon (lng, lat order)."""
    ring = h3.cell_to_boundary(h3_index)
    return Polygon([(lng, lat) for lat, lng in ring])


def compute_cell_water_share(
    h3_indices: list[str],
    water_geoms: list[BaseGeometry],
) -> np.ndarray:
    """Fraction of each cell's area covered by water polygons.

    Returns one float per input cell. Overlapping water polygons are
    unioned per cell before intersecting, so a riverbank inside a
    reservoir polygon does not inflate the share. No water geometries →
    all zeros.
    """
    n = len(h3_indices)
    if n == 0:
        return np.array([], dtype=np.float64)
    if not water_geoms:
        return np.zeros(n, dtype=np.float64)

    cells = [_cell_polygon(h) for h in h3_indices]
    tree = STRtree(water_geoms)
    pairs = tree.query(cells, predicate="intersects")

    # Group intersecting water geometries per cell, union, intersect.
    hits: dict[int, list[int]] = defaultdict(list)
    for k in range(pairs.shape[1]):
        hits[int(pairs[0, k])].append(int(pairs[1, k]))

    shares = np.zeros(n, dtype=np.float64)
    for i, js in hits.items():
        cell = cells[i]
        water = shapely.union_all([water_geoms[j] for j in js])
        intersection = cell.intersection(water)
        if not intersection.is_empty and cell.area > 0:
            shares[i] = intersection.area / cell.area
    return shares
