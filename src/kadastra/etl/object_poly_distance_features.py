"""ADR-0019: per-object distance to the nearest polygon of each layer.

Mirror of ``compute_object_polygon_features`` (share-in-buffer) for
absolute-distance signals. Same projection (EPSG:32639 UTM-39N) so the
two feature blocks are dimensionally compatible.

Implementation: per-layer ``unary_union`` → flatten parts → STRtree;
per-object ``tree.nearest(point)`` returns the nearest part, then
``shapely.distance`` gives the metric. Inside a polygon the distance
is naturally 0. Empty layer → null column.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import shapely
from pyproj import Transformer
from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as shapely_transform
from shapely.ops import unary_union

# UTM-39N — same projection as the share-feature pipeline.
_TO_UTM = Transformer.from_crs("EPSG:4326", "EPSG:32639", always_xy=True)


def _project_lonlat(geom: BaseGeometry) -> BaseGeometry:
    return shapely_transform(lambda x, y, z=None: _TO_UTM.transform(x, y), geom)


def _flatten_to_parts(geom: BaseGeometry) -> list[BaseGeometry]:
    if geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, MultiPolygon):
        return [p for p in geom.geoms if not p.is_empty]
    parts: list[BaseGeometry] = []
    for sub in getattr(geom, "geoms", []):
        if isinstance(sub, (Polygon, MultiPolygon)):
            parts.extend(_flatten_to_parts(sub))
    return parts


def compute_object_poly_distance_features(
    objects: pl.DataFrame,
    *,
    polygons_by_layer: dict[str, list[BaseGeometry]],
) -> pl.DataFrame:
    """Append ``dist_to_<layer>_m`` columns: distance in metres
    (EPSG:32639 UTM-39N) from each object to the nearest polygon of
    each layer. Distance is 0.0 when the object is inside a polygon.

    Empty layers produce all-null columns. Empty input frames produce
    empty columns with the expected names so downstream schema is
    stable across asset-class slices.
    """
    if not polygons_by_layer:
        return objects

    n = objects.height

    if n == 0:
        return objects.with_columns(
            [
                pl.lit(None, dtype=pl.Float64).alias(f"dist_to_{layer}_m")
                for layer in polygons_by_layer
            ]
        )

    obj_lats = objects["lat"].to_numpy()
    obj_lons = objects["lon"].to_numpy()
    obj_xs, obj_ys = _TO_UTM.transform(obj_lons, obj_lats)
    points = shapely.points(np.asarray(obj_xs), np.asarray(obj_ys))

    new_columns: list[pl.Series] = []
    for layer, polys in polygons_by_layer.items():
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
            new_columns.append(
                pl.Series(col_name, [None] * n, dtype=pl.Float64)
            )
            continue
        parts_arr = np.asarray(parts, dtype=object)
        tree = shapely.STRtree(parts_arr)
        nearest_idx = tree.nearest(points)
        distances = shapely.distance(points, parts_arr[nearest_idx])
        new_columns.append(
            pl.Series(col_name, np.asarray(distances, dtype=np.float64))
        )

    return objects.with_columns(new_columns)
