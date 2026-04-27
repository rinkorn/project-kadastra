from typing import Any

import h3
import numpy as np
import polars as pl

from kadastra.etl.haversine import EARTH_RADIUS_METERS, haversine_meters

_BUCKET_RES = 9


def _haversine_one_to_many(lat: float, lon: float, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    rlat1 = np.radians(lat)
    rlon1 = np.radians(lon)
    rlat2 = np.radians(lats)
    rlon2 = np.radians(lons)
    dlat = rlat2 - rlat1
    dlon = rlon2 - rlon1
    a = np.sin(dlat / 2) ** 2 + np.cos(rlat1) * np.cos(rlat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_METERS * np.arcsin(np.sqrt(a))


def compute_object_road_features(
    objects: pl.DataFrame,
    ways: list[dict[str, Any]],
    radius_m: float,
) -> pl.DataFrame:
    if objects.is_empty():
        return objects.with_columns(pl.lit(None, dtype=pl.Float64).alias("road_length_500m"))

    # Materialize segment midpoints + lengths
    seg_lats: list[float] = []
    seg_lons: list[float] = []
    seg_lens: list[float] = []
    for way in ways:
        geom = way.get("geometry", []) or []
        for i in range(len(geom) - 1):
            lat1, lon1 = geom[i]["lat"], geom[i]["lon"]
            lat2, lon2 = geom[i + 1]["lat"], geom[i + 1]["lon"]
            seg_lats.append((lat1 + lat2) / 2)
            seg_lons.append((lon1 + lon2) / 2)
            seg_lens.append(haversine_meters(lat1, lon1, lat2, lon2))

    if not seg_lats:
        zeros = np.zeros(objects.height, dtype=np.float64)
        return objects.with_columns(pl.Series("road_length_500m", zeros))

    seg_lat_arr = np.asarray(seg_lats, dtype=np.float64)
    seg_lon_arr = np.asarray(seg_lons, dtype=np.float64)
    seg_len_arr = np.asarray(seg_lens, dtype=np.float64)

    seg_by_cell: dict[str, list[int]] = {}
    for idx in range(seg_lat_arr.size):
        cell = h3.latlng_to_cell(float(seg_lat_arr[idx]), float(seg_lon_arr[idx]), _BUCKET_RES)
        seg_by_cell.setdefault(cell, []).append(idx)

    edge_m = h3.average_hexagon_edge_length(_BUCKET_RES, unit="m")
    k = max(1, int(np.ceil(radius_m / edge_m)) + 1)

    obj_coords = objects.select(["lat", "lon"]).to_numpy()
    out = np.zeros(obj_coords.shape[0], dtype=np.float64)
    for i in range(obj_coords.shape[0]):
        olat, olon = float(obj_coords[i, 0]), float(obj_coords[i, 1])
        cell = h3.latlng_to_cell(olat, olon, _BUCKET_RES)
        candidate_idx: list[int] = []
        for nc in h3.grid_disk(cell, k):
            candidate_idx.extend(seg_by_cell.get(nc, []))
        if not candidate_idx:
            continue
        idx_arr = np.asarray(candidate_idx, dtype=np.int64)
        dist = _haversine_one_to_many(olat, olon, seg_lat_arr[idx_arr], seg_lon_arr[idx_arr])
        mask = dist < radius_m
        if mask.any():
            out[i] = float(seg_len_arr[idx_arr][mask].sum())

    return objects.with_columns(pl.Series("road_length_500m", out))
