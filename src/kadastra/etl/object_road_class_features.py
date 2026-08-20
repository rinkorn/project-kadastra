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

import polars as pl

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


def normalize_highway_class(highway: str | None) -> str | None:
    """Fold a raw OSM ``highway=*`` value into a normalized class.

    Returns ``None`` for classes outside :data:`NEAREST_ROAD_CLASSES`
    (e.g. ``cycleway``, ``track``, ``construction``) — such ways are not
    used by this feature block.
    """
    raise NotImplementedError


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
    raise NotImplementedError


def join_road_class_features(objects: pl.DataFrame, road_features: pl.DataFrame) -> pl.DataFrame:
    """LEFT JOIN the silver road-class table onto objects by ``object_id``.

    Idempotent: pre-existing feature columns on ``objects`` are dropped
    first (the store is read-write; a rerun reads its own enriched
    output). Objects absent from ``road_features`` get nulls; an empty
    ``road_features`` yields all-null feature columns.
    """
    raise NotImplementedError
