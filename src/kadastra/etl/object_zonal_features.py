"""Zonal density features at multiple radii.

For each (point layer, radius) pair, attach a column
``{layer}_within_{R}m`` with the number of layer points within R
meters of each object (haversine distance). ADR-0013 explains
the radii choice (100/300/500/800 m), why haversine instead of
graph distance, and how self-exclusion works for self-class
density layers.
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl
from scipy.spatial import cKDTree

from kadastra.etl.haversine import EARTH_RADIUS_METERS

# Conservative degrees-per-meter on the lat axis. The kd-tree lives in
# (lat, lon) degree space, so we over-estimate the bbox radius enough
# to cover any coordinate that could fall within `max_radius_m`.
_DEG_PER_METER_LAT = 1.0 / 111_000.0


def _haversine_one_to_many(
    lat: float, lon: float, lats: np.ndarray, lons: np.ndarray
) -> np.ndarray:
    rlat1 = math.radians(lat)
    rlon1 = math.radians(lon)
    rlat2 = np.radians(lats)
    rlon2 = np.radians(lons)
    dlat = rlat2 - rlat1
    dlon = rlon2 - rlon1
    a = (
        np.sin(dlat / 2) ** 2
        + math.cos(rlat1) * np.cos(rlat2) * np.sin(dlon / 2) ** 2
    )
    return 2 * EARTH_RADIUS_METERS * np.arcsin(np.sqrt(a))


def _bbox_radius_deg(max_radius_m: int, ref_lat: float) -> float:
    # The lon-axis is shorter at high latitudes, but the kd-tree query is on
    # raw degrees in both axes, so we must use the *worst* (smallest) lon
    # spacing. The 1.5x slack handles that and rounding error in the kd-tree.
    cos_lat = max(math.cos(math.radians(ref_lat)), 0.1)
    return float(max_radius_m) * _DEG_PER_METER_LAT / cos_lat * 1.5


def compute_object_zonal_features(
    objects: pl.DataFrame,
    *,
    layers: dict[str, pl.DataFrame],
    radii_m: list[int],
) -> pl.DataFrame:
    if not layers or not radii_m:
        return objects

    radii_sorted = sorted({int(r) for r in radii_m})
    n = objects.height

    if n == 0:
        return objects.with_columns(
            [
                pl.lit(None, dtype=pl.Int64).alias(f"{layer}_within_{r}m")
                for layer in layers
                for r in radii_sorted
            ]
        )

    obj_lats = objects["lat"].to_numpy()
    obj_lons = objects["lon"].to_numpy()
    obj_xy = np.column_stack([obj_lats, obj_lons])
    obj_ids = (
        objects["object_id"].to_list() if "object_id" in objects.columns else None
    )

    max_r = max(radii_sorted)
    ref_lat = float(obj_lats.mean())
    bbox_r = _bbox_radius_deg(max_r, ref_lat)

    new_columns: list[pl.Series] = []

    for layer_name, layer_df in layers.items():
        if layer_df.is_empty():
            for r in radii_sorted:
                new_columns.append(
                    pl.Series(
                        f"{layer_name}_within_{r}m",
                        np.zeros(n, dtype=np.int64),
                    )
                )
            continue

        layer_lats = layer_df["lat"].to_numpy()
        layer_lons = layer_df["lon"].to_numpy()
        layer_xy = np.column_stack([layer_lats, layer_lons])
        layer_ids = (
            layer_df["object_id"].to_list()
            if "object_id" in layer_df.columns
            else None
        )

        tree = cKDTree(layer_xy)
        candidate_idx = tree.query_ball_point(obj_xy, bbox_r)

        per_radius = {r: np.zeros(n, dtype=np.int64) for r in radii_sorted}

        for i in range(n):
            cand = candidate_idx[i]
            if not cand:
                continue
            cand_arr = np.asarray(cand, dtype=np.int64)
            if layer_ids is not None and obj_ids is not None:
                cur_id = obj_ids[i]
                keep = np.array(
                    [layer_ids[j] != cur_id for j in cand_arr], dtype=bool
                )
                cand_arr = cand_arr[keep]
                if cand_arr.size == 0:
                    continue
            dists = _haversine_one_to_many(
                float(obj_lats[i]),
                float(obj_lons[i]),
                layer_lats[cand_arr],
                layer_lons[cand_arr],
            )
            for r in radii_sorted:
                per_radius[r][i] = int((dists < r).sum())

        for r in radii_sorted:
            new_columns.append(
                pl.Series(f"{layer_name}_within_{r}m", per_radius[r])
            )

    return objects.with_columns(new_columns)
