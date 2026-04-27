"""ADR-0019: per-object distance to the nearest geometry of each layer.

Mirror of ``compute_object_polygon_features`` (share-in-buffer) for
absolute-distance signals. Same projection (EPSG:32639 UTM-39N) so the
two feature blocks are dimensionally compatible.

Geometry-agnostic: a layer is a list of arbitrary Shapely geometries
(Polygon, LineString, Point, plus their Multi/Collection forms). One
STRtree → ``shapely.distance`` pipeline serves all three so that point
POIs (school, bus_stop), linear externalities (powerline, railway) and
polygonal layers (water, park, landfill) reuse the same code path.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import shapely
from pyproj import Transformer
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as shapely_transform
from shapely.ops import unary_union

# UTM-39N — same projection as the share-feature pipeline.
_TO_UTM = Transformer.from_crs("EPSG:4326", "EPSG:32639", always_xy=True)


def _project_lonlat(geom: BaseGeometry) -> BaseGeometry:
    return shapely_transform(lambda x, y, z=None: _TO_UTM.transform(x, y), geom)


def _flatten_to_parts(geom: BaseGeometry) -> list[BaseGeometry]:
    """Flatten any geometry into its atomic non-empty parts.

    Required because ``unary_union`` of mixed-type input may produce a
    GeometryCollection, and STRtree needs flat leaves to index. Atomic
    parts are anything without ``.geoms`` (Point, LineString, LinearRing,
    Polygon); collection types (Multi*, GeometryCollection) are recursed.
    """
    if geom.is_empty:
        return []
    sub_geoms = getattr(geom, "geoms", None)
    if sub_geoms is None:
        return [geom]
    parts: list[BaseGeometry] = []
    for sub in sub_geoms:
        parts.extend(_flatten_to_parts(sub))
    return parts


def compute_object_geom_distance_features(
    objects: pl.DataFrame,
    *,
    geometries_by_layer: dict[str, list[BaseGeometry]],
) -> pl.DataFrame:
    """Append ``dist_to_<layer>_m`` columns: distance in metres
    (EPSG:32639 UTM-39N) from each object to the nearest geometry of
    each layer. Distance is 0.0 when the object is inside a polygon
    or sits exactly on a line; positive otherwise.

    Empty layers produce all-null columns. Empty input frames produce
    empty columns with the expected names so downstream schema is
    stable across asset-class slices.
    """
    if not geometries_by_layer:
        return objects

    n = objects.height

    if n == 0:
        return objects.with_columns(
            [pl.lit(None, dtype=pl.Float64).alias(f"dist_to_{layer}_m") for layer in geometries_by_layer]
        )

    obj_lats = objects["lat"].to_numpy()
    obj_lons = objects["lon"].to_numpy()
    obj_xs, obj_ys = _TO_UTM.transform(obj_lons, obj_lats)
    points = shapely.points(np.asarray(obj_xs), np.asarray(obj_ys))

    new_columns: list[pl.Series] = []
    for layer, polys in geometries_by_layer.items():
        col_name = f"dist_to_{layer}_m"
        if not polys:
            new_columns.append(
                pl.Series(
                    col_name,
                    [None] * n,
                    dtype=pl.Float64,
                )
            )
            continue
        projected = [_project_lonlat(p) for p in polys]
        merged = unary_union(projected)
        parts = _flatten_to_parts(merged)
        if not parts:
            new_columns.append(pl.Series(col_name, [None] * n, dtype=pl.Float64))
            continue
        parts_arr = np.asarray(parts, dtype=object)
        tree = shapely.STRtree(parts_arr)
        nearest_idx = tree.nearest(points)
        distances = shapely.distance(points, parts_arr[nearest_idx])
        new_columns.append(pl.Series(col_name, np.asarray(distances, dtype=np.float64)))

    return objects.with_columns(new_columns)
