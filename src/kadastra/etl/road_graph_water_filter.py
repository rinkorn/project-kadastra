"""Water-crossing filter for road-graph edges (ADR-0030).

OSM pedestrian extracts contain phantom crossings: ways whose geometry
runs straight across river channels where no bridge or ferry exists in
reality (digitizing artifacts, winter ice roads, abandoned piers). Once
such edges land in the graph, Dijkstra happily routes pedestrians across
open water — e.g. a ~2.3 km chain across the Volga channel to Verkhny
Uslon that produced garbage walk distances for the whole right bank.

The filter drops every edge whose segment intersects the water polygons
(``kazan-agg-water.geojsonseq``) for more than ``max_crossing_m`` meters,
unless the source way is a bridge or a tunnel (``bridge``/``tunnel`` tag
present and not ``"no"``). The 30 m threshold tolerates the fuzzy OSM
water-polygon shoreline (real embankment streets hug the waterline and
clip a few meters of the polygon) while any genuine open-water crossing
is an order of magnitude longer.

Spatial work follows the ADR-0014 pattern: water polygons are projected
to UTM 39N, unioned and indexed in an ``STRtree``; per-edge lengths come
from vectorized ``shapely.intersection`` + ``shapely.length`` over the
tree hits, rolled back per edge with ``np.bincount``.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import shapely
from pyproj import Transformer
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as shapely_transform
from shapely.ops import unary_union

DEFAULT_MAX_WATER_CROSSING_M = 30.0

# UTM zone 39N — same projection used by the other spatial ETL modules;
# minimal length distortion (≤ 0.1 %) at Kazan latitude.
_TO_UTM = Transformer.from_crs("EPSG:4326", "EPSG:32639", always_xy=True)


def _project_lonlat(geom: BaseGeometry) -> BaseGeometry:
    return shapely_transform(lambda x, y, z=None: _TO_UTM.transform(x, y), geom)


def _polygon_parts(geom: BaseGeometry) -> list[BaseGeometry]:
    """Flatten a (Multi)Polygon union into disjoint polygon parts."""
    if geom.is_empty:
        return []
    geoms = getattr(geom, "geoms", None)
    if geoms is None:
        return [geom]
    parts: list[BaseGeometry] = []
    for sub in geoms:
        parts.extend(_polygon_parts(sub) if hasattr(sub, "geoms") else [sub])
    return [p for p in parts if not p.is_empty]


def _bridge_or_tunnel_mask(edges: pl.DataFrame) -> np.ndarray:
    """True for edges whose source way is a bridge or a tunnel.

    A tag counts when present and not ``"no"``. Old artifacts without
    the tag columns (pre-ADR-0030) exempt nothing."""

    def _tag_active(col: str) -> pl.Expr:
        return pl.col(col).is_not_null() & (pl.col(col) != "no")

    exprs = [_tag_active(c) for c in ("bridge", "tunnel") if c in edges.columns]
    if not exprs:
        return np.zeros(edges.height, dtype=bool)
    combined = exprs[0]
    for e in exprs[1:]:
        combined = combined | e
    result: np.ndarray = edges.select(combined.alias("exempt"))["exempt"].to_numpy()
    return result


def filter_water_crossing_edges(
    edges: pl.DataFrame,
    water_polygons: list[BaseGeometry],
    *,
    max_crossing_m: float = DEFAULT_MAX_WATER_CROSSING_M,
) -> pl.DataFrame:
    """Drop edges crossing water polygons by more than ``max_crossing_m``.

    Bridges and tunnels are exempt (see module docstring). Returns the
    input frame minus the phantom crossings; schema unchanged.
    """
    if edges.is_empty() or not water_polygons:
        return edges

    merged = unary_union([_project_lonlat(p) for p in water_polygons])
    parts = _polygon_parts(merged)
    if not parts:
        return edges
    parts_arr = np.asarray(parts, dtype=object)
    tree = shapely.STRtree(parts_arr)

    from_x, from_y = _TO_UTM.transform(
        edges["from_lon"].to_numpy(),
        edges["from_lat"].to_numpy(),
    )
    to_x, to_y = _TO_UTM.transform(
        edges["to_lon"].to_numpy(),
        edges["to_lat"].to_numpy(),
    )
    coords = np.stack(
        [
            np.column_stack([from_x, from_y]),
            np.column_stack([to_x, to_y]),
        ],
        axis=1,
    )
    segments = np.asarray(shapely.linestrings(coords))

    pairs = tree.query(segments, predicate="intersects")
    if pairs.shape[1] == 0:
        return edges
    edge_ids, part_ids = pairs[0], pairs[1]
    crossings = shapely.intersection(segments[edge_ids], parts_arr[part_ids])
    crossing_m = np.bincount(
        edge_ids,
        weights=shapely.length(crossings),
        minlength=edges.height,
    )

    exempt = _bridge_or_tunnel_mask(edges)
    drop = (crossing_m > max_crossing_m) & ~exempt
    return edges.filter(pl.Series(~drop))
