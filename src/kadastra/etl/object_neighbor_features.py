import h3
import numpy as np
import polars as pl

from kadastra.domain.asset_class import AssetClass
from kadastra.etl.haversine import EARTH_RADIUS_METERS

_BUCKET_RES = 9
_TRACKED_CLASSES = (
    AssetClass.APARTMENT,
    AssetClass.HOUSE,
    AssetClass.COMMERCIAL,
)


_PLURAL = {
    AssetClass.APARTMENT: "apartments",
    AssetClass.HOUSE: "houses",
    AssetClass.COMMERCIAL: "commercial",
}


def _column_name(cls: AssetClass) -> str:
    return f"count_{_PLURAL[cls]}_500m"


def _haversine_one_to_many(
    lat: float, lon: float, lats: np.ndarray, lons: np.ndarray
) -> np.ndarray:
    rlat1 = np.radians(lat)
    rlon1 = np.radians(lon)
    rlat2 = np.radians(lats)
    rlon2 = np.radians(lons)
    dlat = rlat2 - rlat1
    dlon = rlon2 - rlon1
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(rlat1) * np.cos(rlat2) * np.sin(dlon / 2) ** 2
    )
    return 2 * EARTH_RADIUS_METERS * np.arcsin(np.sqrt(a))


def compute_object_neighbor_features(
    objects: pl.DataFrame, radius_m: float
) -> pl.DataFrame:
    column_names = [_column_name(cls) for cls in _TRACKED_CLASSES]

    if objects.is_empty():
        return objects.with_columns(
            [pl.lit(None, dtype=pl.Int64).alias(col) for col in column_names]
        )

    n = objects.height
    obj_classes = objects["asset_class"].to_list()
    obj_coords = objects.select(["lat", "lon"]).to_numpy()

    # Bucket all objects by H3 res=9
    cells = [
        h3.latlng_to_cell(float(obj_coords[i, 0]), float(obj_coords[i, 1]), _BUCKET_RES)
        for i in range(n)
    ]
    by_cell: dict[str, list[int]] = {}
    for i, c in enumerate(cells):
        by_cell.setdefault(c, []).append(i)

    edge_m = h3.average_hexagon_edge_length(_BUCKET_RES, unit="m")
    k = max(1, int(np.ceil(radius_m / edge_m)) + 1)

    counts: dict[AssetClass, np.ndarray] = {
        cls: np.zeros(n, dtype=np.int64) for cls in _TRACKED_CLASSES
    }

    obj_lats = obj_coords[:, 0]
    obj_lons = obj_coords[:, 1]

    for i in range(n):
        olat = float(obj_lats[i])
        olon = float(obj_lons[i])
        candidate_idx: list[int] = []
        for nc in h3.grid_disk(cells[i], k):
            candidate_idx.extend(by_cell.get(nc, []))
        if not candidate_idx:
            continue
        idx_arr = np.asarray(candidate_idx, dtype=np.int64)
        # Exclude self
        idx_arr = idx_arr[idx_arr != i]
        if idx_arr.size == 0:
            continue
        dist = _haversine_one_to_many(olat, olon, obj_lats[idx_arr], obj_lons[idx_arr])
        within = dist < radius_m
        if not within.any():
            continue
        within_idx = idx_arr[within]
        for cls in _TRACKED_CLASSES:
            target_value = cls.value
            cnt = sum(1 for j in within_idx if obj_classes[j] == target_value)
            counts[cls][i] = cnt

    return objects.with_columns(
        [pl.Series(_column_name(cls), counts[cls]) for cls in _TRACKED_CLASSES]
    )
