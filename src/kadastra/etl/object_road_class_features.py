"""Per-object nearest-road-class features (ADR-0024, group 1).

From the OSM road ways of the region (major classes from
``tatarstan_major_roads.json``, minor classes from
``kazan-agg-minor_road_ways.parquet`` — see ADR-0024 «Аудит данных»)
derives, for every valuation object:

- ``nearest_road_class``   (Utf8) — normalized OSM highway class of the
  nearest road way;
- ``dist_to_motorway_m``   (Float64) — distance to the nearest
  motorway/trunk way;
- ``dist_to_primary_m``    (Float64);
- ``dist_to_secondary_m``  (Float64);
- ``dist_to_residential_m``(Float64) — landplot «подъезд» signal;
- ``dist_to_pedestrian_m`` (Float64) — pedestrian infrastructure union
  (``pedestrian`` + ``living_street`` + ``footway``; the two former are
  nearly absent in the region, so they are merged to keep the feature
  informative — ADR-0024 «Аудит данных», п. 2).

Distances are true point-to-polyline distances in metres, computed in
EPSG:32639 (UTM-39N) — same projection and STRtree pattern as
``object_geom_distance_features`` (ADR-0019). The ADR's KDTree sketch
was upgraded to exact line distances; see ADR-0024 «Отличия реализации».

The builder (``scripts/build_nearest_road_features.py``) materializes a
per-object silver table; ``BuildObjectFeatures`` then LEFT JOINs it via
:func:`join_road_class_features` after the RAW_OBJECT_SCHEMA reset, so
the columns are recomputed from silver on every run (ADR-0022/0023
pattern).
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import polars as pl
import shapely
from pyproj import Transformer
from shapely.geometry import LineString

# UTM-39N — same projection as the other distance-feature pipelines.
_TO_UTM = Transformer.from_crs("EPSG:4326", "EPSG:32639", always_xy=True)

# Normalized highway classes that may appear as ``nearest_road_class``.
# Raw OSM classes are folded by :func:`normalize_highway_class`:
# ``*_link`` → base class, ``footway``/``living_street`` → ``pedestrian``.
NEAREST_ROAD_CLASSES: tuple[str, ...] = (
    "motorway",
    "trunk",
    "primary",
    "secondary",
    "tertiary",
    "residential",
    "service",
    "unclassified",
    "pedestrian",
)

# Distance feature column → normalized classes indexed for it.
DIST_CLASS_GROUPS: dict[str, tuple[str, ...]] = {
    "dist_to_motorway_m": ("motorway", "trunk"),
    "dist_to_primary_m": ("primary",),
    "dist_to_secondary_m": ("secondary",),
    "dist_to_residential_m": ("residential",),
    "dist_to_pedestrian_m": ("pedestrian",),
}

ROAD_CLASS_FEATURE_COLUMNS: tuple[str, ...] = (
    "nearest_road_class",
    *DIST_CLASS_GROUPS.keys(),
)

_OUTPUT_SCHEMA: dict[str, type[pl.DataType] | pl.DataType] = {
    "object_id": pl.Utf8,
    "nearest_road_class": pl.Utf8,
    **{col: pl.Float64 for col in DIST_CLASS_GROUPS},
}


def normalize_highway_class(highway: str | None) -> str | None:
    """Fold a raw OSM ``highway=*`` value into a normalized class.

    Returns ``None`` for classes outside :data:`NEAREST_ROAD_CLASSES`
    (e.g. ``cycleway``, ``track``, ``construction``) — such ways are not
    used by this feature block.
    """
    if not highway:
        return None
    base = highway[: -len("_link")] if highway.endswith("_link") else highway
    if base in ("footway", "living_street"):
        return "pedestrian"
    return base if base in NEAREST_ROAD_CLASSES else None


def compute_nearest_road_features(
    objects: pl.DataFrame,
    *,
    ways_by_class: dict[str, list[list[tuple[float, float]]]],
) -> pl.DataFrame:
    """Per-object road-class features from road ways.

    ``objects`` must carry ``object_id``, ``lat``, ``lon``.
    ``ways_by_class`` maps a *normalized* class to its ways; each way is
    a list of ``(lat, lon)`` vertices. Returns a frame keyed by
    ``object_id`` with :data:`ROAD_CLASS_FEATURE_COLUMNS`. Objects with
    null coordinates — and distance columns whose class group has no
    ways — come out null.
    """
    if objects.height == 0:
        return pl.DataFrame(schema=_OUTPUT_SCHEMA)

    lines_by_class: dict[str, list[LineString]] = {}
    for cls, ways in ways_by_class.items():
        lines: list[LineString] = []
        for way in ways:
            if len(way) < 2:
                continue
            arr = np.asarray(way, dtype=np.float64)  # (lat, lon) pairs
            xs, ys = _TO_UTM.transform(arr[:, 1], arr[:, 0])
            lines.append(LineString(np.column_stack([xs, ys])))
        lines_by_class[cls] = lines

    all_lines: list[LineString] = []
    all_labels: list[str] = []
    for cls in NEAREST_ROAD_CLASSES:  # stable order → deterministic ties
        for line in lines_by_class.get(cls, []):
            all_lines.append(line)
            all_labels.append(cls)

    n = objects.height
    nearest_class: list[str | None] = [None] * n
    dist_values: dict[str, list[float | None]] = {col: [None] * n for col in DIST_CLASS_GROUPS}

    lat_arr = objects["lat"].to_numpy()
    lon_arr = objects["lon"].to_numpy()
    valid = ~(np.isnan(lat_arr) | np.isnan(lon_arr))
    if valid.any():
        obj_xs, obj_ys = _TO_UTM.transform(lon_arr[valid], lat_arr[valid])
        points = shapely.points(np.asarray(obj_xs), np.asarray(obj_ys))
        valid_idx = np.flatnonzero(valid)

        if all_lines:
            all_arr = np.asarray(all_lines, dtype=object)
            nearest = cast(np.ndarray[Any, Any], shapely.STRtree(all_arr).nearest(points))
            nearest_idx = [int(i) for i in nearest]
            for row_i, line_i in zip(valid_idx.tolist(), nearest_idx, strict=True):
                nearest_class[row_i] = all_labels[line_i]

        for col, classes in DIST_CLASS_GROUPS.items():
            group_lines = [line for cls in classes for line in lines_by_class.get(cls, [])]
            if not group_lines:
                continue
            group_arr = np.asarray(group_lines, dtype=object)
            nearest_idx = shapely.STRtree(group_arr).nearest(points)
            distances = shapely.distance(points, group_arr[nearest_idx])
            for row_i, dist in zip(valid_idx.tolist(), distances.tolist(), strict=True):
                dist_values[col][row_i] = float(dist)

    return pl.DataFrame(
        {
            "object_id": objects["object_id"],
            "nearest_road_class": pl.Series(nearest_class, dtype=pl.Utf8),
            **{col: pl.Series(vals, dtype=pl.Float64) for col, vals in dist_values.items()},
        }
    )


def join_road_class_features(objects: pl.DataFrame, road_features: pl.DataFrame) -> pl.DataFrame:
    """LEFT JOIN the silver road-class table onto objects by ``object_id``.

    Idempotent: pre-existing feature columns on ``objects`` are dropped
    first (the store is read-write; a rerun reads its own enriched
    output). Objects absent from ``road_features`` get nulls; an empty
    ``road_features`` yields all-null feature columns.
    """
    drop_existing = [c for c in ROAD_CLASS_FEATURE_COLUMNS if c in objects.columns]
    if drop_existing:
        objects = objects.drop(drop_existing)

    if road_features.is_empty():
        return objects.with_columns(
            [
                pl.lit(None, dtype=pl.Utf8).alias("nearest_road_class"),
                *[pl.lit(None, dtype=pl.Float64).alias(col) for col in DIST_CLASS_GROUPS],
            ]
        )

    right = road_features.select(["object_id", *ROAD_CLASS_FEATURE_COLUMNS])
    return objects.join(right, on="object_id", how="left")
