"""Poly-area buffer features.

For each (polygon layer, radius) pair, attach a column
``{layer}_share_{R}m`` whose value is the share of the buffer-circle
area at the object covered by polygons of that layer (range [0, 1]).

ADR-0014 explains the rationale (UTM 39N projection, four starter
layers water/park/industrial/cemetery, and how within-layer overlapping
polygons are unioned so the per-layer share never exceeds 1.0 while
across-layer sums may).
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl
from pyproj import Transformer
from shapely.geometry import Point
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as shapely_transform
from shapely.ops import unary_union

# UTM zone 39N — same projection used by the agglomeration boundary
# build script; minimal area distortion (≤ 0.1 %) at Kazan latitude.
_TO_UTM = Transformer.from_crs("EPSG:4326", "EPSG:32639", always_xy=True)


def _project_lonlat(geom: BaseGeometry) -> BaseGeometry:
    return shapely_transform(lambda x, y, z=None: _TO_UTM.transform(x, y), geom)


def compute_object_polygon_features(
    objects: pl.DataFrame,
    *,
    polygons_by_layer: dict[str, list[BaseGeometry]],
    radii_m: list[int],
) -> pl.DataFrame:
    if not polygons_by_layer or not radii_m:
        return objects

    radii_sorted = sorted({int(r) for r in radii_m})
    n = objects.height

    if n == 0:
        return objects.with_columns(
            [
                pl.lit(None, dtype=pl.Float64).alias(f"{layer}_share_{r}m")
                for layer in polygons_by_layer
                for r in radii_sorted
            ]
        )

    # Project objects once.
    obj_lats = objects["lat"].to_numpy()
    obj_lons = objects["lon"].to_numpy()
    obj_xs = np.empty(n, dtype=np.float64)
    obj_ys = np.empty(n, dtype=np.float64)
    for i in range(n):
        x, y = _TO_UTM.transform(float(obj_lons[i]), float(obj_lats[i]))
        obj_xs[i] = x
        obj_ys[i] = y

    new_columns: list[pl.Series] = []

    for layer, polys in polygons_by_layer.items():
        # Union the layer's polygons in UTM space so overlapping polygons
        # don't double-count (cap per-layer share at 1.0). For an empty
        # list this just yields zeros.
        if polys:
            projected = [_project_lonlat(p) for p in polys]
            merged = unary_union(projected)
        else:
            merged = None

        per_radius = {r: np.zeros(n, dtype=np.float64) for r in radii_sorted}

        if merged is not None and not merged.is_empty:
            merged_bounds = merged.bounds  # (minx, miny, maxx, maxy)
            for i in range(n):
                ox = obj_xs[i]
                oy = obj_ys[i]
                # Quick-reject: object's max-radius envelope must touch
                # the layer's bbox to do any expensive work.
                max_r = radii_sorted[-1]
                if (
                    ox + max_r < merged_bounds[0]
                    or ox - max_r > merged_bounds[2]
                    or oy + max_r < merged_bounds[1]
                    or oy - max_r > merged_bounds[3]
                ):
                    continue
                for r in radii_sorted:
                    circle = Point(ox, oy).buffer(r)
                    inter_area = merged.intersection(circle).area
                    per_radius[r][i] = inter_area / (math.pi * r * r)

        for r in radii_sorted:
            new_columns.append(
                pl.Series(f"{layer}_share_{r}m", per_radius[r])
            )

    return objects.with_columns(new_columns)
