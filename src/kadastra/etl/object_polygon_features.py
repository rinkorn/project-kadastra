"""Poly-area buffer features.

For each (polygon layer, radius) pair, attach a column
``{layer}_share_{R}m`` whose value is the share of the buffer-circle
area at the object covered by polygons of that layer (range [0, 1]).

ADR-0014 explains the rationale (UTM 39N projection, four starter
layers water/park/industrial/cemetery, and how within-layer overlapping
polygons are unioned so the per-layer share never exceeds 1.0 while
across-layer sums may).

Implementation uses the **shapely 2.0 array API + STRtree + threads**:

1. Per layer we ``unary_union`` projected polygons once and split the
   result into a flat list of non-overlapping parts (so summed
   intersection areas per buffer cannot double-count within a layer)
   and build a single ``STRtree`` over those parts.
2. Per radius we issue one ``shapely.buffer`` over all N objects.
3. The 16 (layer, radius) pairs run on a ``ThreadPoolExecutor``: each
   task does ``tree.query(buffers, predicate='intersects')``, then
   ``shapely.intersection`` + ``shapely.area`` over the (typically
   K << N·M) hit pairs, and ``np.bincount`` rolls per-pair areas back
   to per-buffer totals. shapely 2 releases the GIL during these
   batch C calls, so threading gives real multi-core speedup with
   no per-task pickling cost.

This avoids the O(N · vertices_in_merged) cost of intersecting each
buffer against a single huge multipolygon — the slow path that the
naive vectorization fell into for layers like water/industrial.

The denominator for the share is ``shapely.area(buffers)``, not
``π·r²``. They differ by the polygon-discretization error of the
buffer (≈ 0.6 % at the default ``quad_segs=8``). Using the actual
buffer area makes the result independent of that discretization —
``share == 1.0`` exactly when the buffer is fully inside the layer.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import polars as pl
import shapely
from pyproj import Transformer
from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as shapely_transform
from shapely.ops import unary_union

# UTM zone 39N — same projection used by the agglomeration boundary
# build script; minimal area distortion (≤ 0.1 %) at Kazan latitude.
_TO_UTM = Transformer.from_crs("EPSG:4326", "EPSG:32639", always_xy=True)


def _project_lonlat(geom: BaseGeometry) -> BaseGeometry:
    return shapely_transform(lambda x, y, z=None: _TO_UTM.transform(x, y), geom)


def _flatten_to_parts(geom: BaseGeometry) -> list[BaseGeometry]:
    """Split a (Multi)Polygon into a list of disjoint Polygon parts.

    After ``unary_union`` the parts are guaranteed non-overlapping, so
    summed intersection areas per buffer cannot double-count within
    a layer. Anything that's empty or non-polygonal is dropped."""
    if geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, MultiPolygon):
        return [p for p in geom.geoms if not p.is_empty]
    # GeometryCollection or unexpected — keep only polygonal parts.
    parts: list[BaseGeometry] = []
    for sub in getattr(geom, "geoms", []):
        if isinstance(sub, (Polygon, MultiPolygon)):
            parts.extend(_flatten_to_parts(sub))
    return parts


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

    # Project objects once, vectorized via pyproj.
    obj_lats = objects["lat"].to_numpy()
    obj_lons = objects["lon"].to_numpy()
    obj_xs, obj_ys = _TO_UTM.transform(obj_lons, obj_lats)
    points = shapely.points(np.asarray(obj_xs), np.asarray(obj_ys))

    # Per-radius buffers + areas — shared across all layer threads.
    buffers_by_r: dict[int, np.ndarray] = {}
    buffer_areas_by_r: dict[int, np.ndarray] = {}
    for r in radii_sorted:
        buffers = np.asarray(shapely.buffer(points, r))
        buffers_by_r[r] = buffers
        buffer_areas_by_r[r] = np.asarray(shapely.area(buffers))

    # Per-layer STRtree (None when the layer has no polygons).
    trees_by_layer: dict[str, tuple[shapely.STRtree, np.ndarray] | None] = {}
    for layer, polys in polygons_by_layer.items():
        if not polys:
            trees_by_layer[layer] = None
            continue
        projected = [_project_lonlat(p) for p in polys]
        merged = unary_union(projected)
        parts = _flatten_to_parts(merged)
        if not parts:
            trees_by_layer[layer] = None
            continue
        parts_arr = np.asarray(parts, dtype=object)
        trees_by_layer[layer] = (shapely.STRtree(parts_arr), parts_arr)

    def _share_for_pair(layer: str, r: int) -> tuple[str, int, np.ndarray]:
        state = trees_by_layer[layer]
        buffer_areas = buffer_areas_by_r[r]
        if state is None:
            return layer, r, np.zeros(n, dtype=np.float64)
        tree, parts_arr = state
        buffers = buffers_by_r[r]
        pairs = tree.query(buffers, predicate="intersects")
        if pairs.shape[1] == 0:
            return layer, r, np.zeros(n, dtype=np.float64)
        buf_ids, part_ids = pairs[0], pairs[1]
        inter = shapely.intersection(buffers[buf_ids], parts_arr[part_ids])
        inter_areas = shapely.area(inter)
        summed = np.bincount(buf_ids, weights=inter_areas, minlength=n)
        return layer, r, np.minimum(summed / buffer_areas, 1.0)

    pairs_to_run = [(layer, r) for layer in polygons_by_layer for r in radii_sorted]
    max_workers = min(len(pairs_to_run), os.cpu_count() or 1)
    results: dict[tuple[str, int], np.ndarray] = {}
    if max_workers <= 1 or len(pairs_to_run) <= 1:
        for layer, r in pairs_to_run:
            _, _, shares = _share_for_pair(layer, r)
            results[(layer, r)] = shares
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            def _run_pair(lr: tuple[str, int]) -> tuple[str, int, np.ndarray]:
                return _share_for_pair(*lr)

            for layer, r, shares in ex.map(_run_pair, pairs_to_run):
                results[(layer, r)] = shares

    new_columns: list[pl.Series] = [
        pl.Series(f"{layer}_share_{r}m", results[(layer, r)])
        for layer in polygons_by_layer
        for r in radii_sorted
    ]
    return objects.with_columns(new_columns)
