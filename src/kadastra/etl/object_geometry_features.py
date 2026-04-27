"""Per-object geometric ZOFs from polygon_wkt_3857 (ADR-0018).

Reads ``polygon_wkt_3857`` (EPSG:3857 WKT) and appends 7 columns:

- ``polygon_area_m2``         (Float64) — shapely.area
- ``polygon_perimeter_m``     (Float64) — shapely.length
- ``polygon_compactness``     (Float64) — Polsby–Popper 4·π·A / P²
- ``polygon_convexity``       (Float64) — A / area(convex_hull)
- ``bbox_aspect_ratio``       (Float64) — long / short side of min rotated bbox
- ``polygon_orientation_deg`` (Float64) — long-axis angle of min rotated
                                          bbox, [0°, 180°)
- ``polygon_n_vertices``      (Int64)   — exterior vertex count

Null WKT → all 7 features null on that row.
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import shapely
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry

_NEW_FLOAT_COLUMNS = (
    "polygon_area_m2",
    "polygon_perimeter_m",
    "polygon_compactness",
    "polygon_convexity",
    "bbox_aspect_ratio",
    "polygon_orientation_deg",
)
_NEW_INT_COLUMN = "polygon_n_vertices"


def compute_object_geometry_features(objects: pl.DataFrame) -> pl.DataFrame:
    if "polygon_wkt_3857" not in objects.columns:
        raise KeyError(
            "compute_object_geometry_features requires column 'polygon_wkt_3857' (EPSG:3857 WKT) — it is not present"
        )

    n = objects.height
    if n == 0:
        return objects.with_columns(
            *[pl.lit(None, dtype=pl.Float64).alias(c) for c in _NEW_FLOAT_COLUMNS],
            pl.lit(None, dtype=pl.Int64).alias(_NEW_INT_COLUMN),
        )

    wkts = objects["polygon_wkt_3857"].to_list()

    area: list[float | None] = [None] * n
    perim: list[float | None] = [None] * n
    compact: list[float | None] = [None] * n
    convex: list[float | None] = [None] * n
    aspect: list[float | None] = [None] * n
    orient: list[float | None] = [None] * n
    nverts: list[int | None] = [None] * n

    for i, wkt in enumerate(wkts):
        if wkt is None:
            continue
        geom: BaseGeometry = shapely.from_wkt(wkt)
        if geom.is_empty:
            continue

        a = float(shapely.area(geom))
        p = float(shapely.length(geom))
        area[i] = a
        perim[i] = p
        compact[i] = (4.0 * math.pi * a / (p * p)) if p > 0 else None

        hull = shapely.convex_hull(geom)
        hull_area = float(shapely.area(hull))
        convex[i] = (a / hull_area) if hull_area > 0 else None

        bbox = geom.minimum_rotated_rectangle
        side_long, side_short, angle_deg = _bbox_long_short_angle(bbox)
        aspect[i] = (side_long / side_short) if side_short > 0 else None
        orient[i] = angle_deg if not math.isnan(angle_deg) else None

        # shapely's get_num_coordinates counts the closing repeat in a
        # Polygon's exterior ring, so a square reports 5 — subtract one
        # to get unique-vertex count.
        nverts[i] = int(shapely.get_num_coordinates(geom)) - 1

    return objects.with_columns(
        pl.Series("polygon_area_m2", area, dtype=pl.Float64),
        pl.Series("polygon_perimeter_m", perim, dtype=pl.Float64),
        pl.Series("polygon_compactness", compact, dtype=pl.Float64),
        pl.Series("polygon_convexity", convex, dtype=pl.Float64),
        pl.Series("bbox_aspect_ratio", aspect, dtype=pl.Float64),
        pl.Series("polygon_orientation_deg", orient, dtype=pl.Float64),
        pl.Series(_NEW_INT_COLUMN, nverts, dtype=pl.Int64),
    )


def _bbox_long_short_angle(bbox: BaseGeometry) -> tuple[float, float, float]:
    """Extract long-side length, short-side length, and the long-side
    angle (in degrees, normalized to [0°, 180°)) from a minimum-rotated
    bounding rectangle. shapely returns it as a Polygon with 5 exterior
    points (4 corners + closing repeat) ordered along the rectangle.

    For an empty / degenerate bbox returns NaNs and 0°."""
    if not isinstance(bbox, Polygon):
        return float("nan"), float("nan"), float("nan")
    coords = np.asarray(bbox.exterior.coords)
    if coords.shape[0] < 4:
        return float("nan"), float("nan"), float("nan")
    # The first 4 points are the rectangle corners in order.
    p0, p1, p2 = coords[0], coords[1], coords[2]
    side_a_vec = p1 - p0
    side_b_vec = p2 - p1
    side_a_len = float(np.linalg.norm(side_a_vec))
    side_b_len = float(np.linalg.norm(side_b_vec))
    if side_a_len >= side_b_len:
        long_len, short_len, long_vec = side_a_len, side_b_len, side_a_vec
    else:
        long_len, short_len, long_vec = side_b_len, side_a_len, side_b_vec
    angle_rad = math.atan2(long_vec[1], long_vec[0])
    angle_deg = math.degrees(angle_rad) % 180.0
    return long_len, short_len, angle_deg
