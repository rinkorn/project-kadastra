"""Per-object ЗОУИТ features (ADR-0025, п. 3).

Source reality (ADR-0025 «Аудит данных»): the ADR's hypothetical
``attrs.zouit_intersection`` NSPD field does not exist; instead there is
a full bulk dump of НСПД layer 36302 (ЗОУИТ) for Tatarstan —
``data/raw/nspd/zouit-tatarstan/page-*.json``, 177 663 zone polygons in
EPSG:3857 with ``properties.options.type_zone`` (human-readable zone
kind). The implementation therefore does a real spatial join (object
point inside zone polygon) instead of reading a pre-baked flag.

Two-stage silver pipeline:

1. ``scripts/build_zouit_silver.py`` parses the page dump into the zone
   layer (``zouit_silver_path``), filtering to the region bbox (the dump
   covers all of Tatarstan) and mapping ``type_zone`` to a
   :data:`ZOUIT_CATEGORIES` category via :func:`categorize_zouit_type`.
2. ``scripts/build_zouit_features.py`` materializes the per-object
   table (``zouit_features_path``) with
   :func:`compute_object_zouit_features`; ``BuildObjectFeatures`` then
   LEFT JOINs it via :func:`join_zouit_features` after the
   RAW_OBJECT_SCHEMA reset (ADR-0022/0023/0024 pattern).

All spatial work in EPSG:32639 (UTM-39N); zone geometries are stored in
the silver layer as EPSG:3857 WKT (as they arrive from НСПД).
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import polars as pl
import shapely
import shapely.wkt
from pyproj import Transformer
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as shapely_transform

ZOUIT_CATEGORIES: tuple[str, ...] = (
    "water_protection",
    "sanitary",
    "heritage_buffer",
    "power_line",
    "pipeline",
    "aerodrome",
    "other",
)

ZOUIT_ZONE_SCHEMA: dict[str, type[pl.DataType] | pl.DataType] = {
    "zouit_id": pl.Utf8,
    "type_zone": pl.Utf8,
    "category": pl.Utf8,
    # WKT in EPSG:3857 (web mercator), as the НСПД dump ships it.
    "geometry_wkt_3857": pl.Utf8,
}

ZOUIT_FEATURE_COLUMNS: tuple[str, ...] = (
    "inside_zouit",
    "zouit_types",
    "inside_water_protection",
)

_ZOUIT_OUTPUT_SCHEMA: dict[str, type[pl.DataType] | pl.DataType] = {
    "object_id": pl.Utf8,
    "inside_zouit": pl.Int64,
    "zouit_types": pl.Utf8,
    "inside_water_protection": pl.Int64,
}

# EPSG:3857 (dump CRS) → UTM-39N (all spatial work, same as the other
# distance/containment pipelines).
_3857_TO_UTM = Transformer.from_crs("EPSG:3857", "EPSG:32639", always_xy=True)
_WGS84_TO_UTM = Transformer.from_crs("EPSG:4326", "EPSG:32639", always_xy=True)

# Ordered (needle, category) rules; first match wins. Substring match on
# the lowercased type_zone — the dump's raw values are long legal
# formulations, so exact matching would be brittle.
_CATEGORY_RULES: tuple[tuple[str, str], ...] = (
    ("культурного наследия", "heritage_buffer"),
    ("электроэнергетик", "power_line"),
    ("электросетев", "power_line"),
    ("гидроэнергетическ", "power_line"),
    ("трубопровод", "pipeline"),
    ("санитарн", "sanitary"),
    ("водоохранн", "water_protection"),
    ("прибрежная защитная", "water_protection"),
    ("затоплени", "water_protection"),
    ("приаэродромн", "aerodrome"),
)


def categorize_zouit_type(type_zone: str | None) -> str:
    """Map a raw ``type_zone`` string to a :data:`ZOUIT_CATEGORIES` value.

    Substring rules over the lowercased string, ordered by specificity;
    anything unmatched (including null/empty) falls to ``other``.
    """
    if not type_zone:
        return "other"
    lowered = type_zone.lower()
    for needle, category in _CATEGORY_RULES:
        if needle in lowered:
            return category
    return "other"


def parse_zouit_feature(feature: dict[str, Any]) -> dict[str, Any] | None:
    """Parse one НСПД page-dump feature into a silver zone row.

    Returns ``None`` for features without geometry. The output dict
    follows :data:`ZOUIT_ZONE_SCHEMA` (category already mapped).
    """
    geom_dict = feature.get("geometry")
    if geom_dict is None:
        return None
    geom = shape(geom_dict)
    if geom.is_empty:
        return None
    props = feature.get("properties") or {}
    options = props.get("options") or {}
    type_zone = options.get("type_zone")
    zouit_id = props.get("externalKey") or options.get("reg_numb_border")
    return {
        "zouit_id": str(zouit_id) if zouit_id is not None else str(feature.get("id")),
        "type_zone": str(type_zone) if type_zone else None,
        "category": categorize_zouit_type(type_zone if isinstance(type_zone, str) else None),
        "geometry_wkt_3857": geom.wkt,
    }


def _project_3857_to_utm(geom: BaseGeometry) -> BaseGeometry:
    return shapely_transform(lambda x, y, z=None: _3857_TO_UTM.transform(x, y), geom)


def compute_object_zouit_features(
    objects: pl.DataFrame,
    *,
    zones: pl.DataFrame,
) -> pl.DataFrame:
    """Per-object ЗОУИТ features from the silver zone layer.

    ``objects`` must carry ``object_id``, ``lat``, ``lon``; ``zones``
    follows :data:`ZOUIT_ZONE_SCHEMA`. Returns a frame keyed by
    ``object_id`` with :data:`ZOUIT_FEATURE_COLUMNS`:

    - ``inside_zouit`` (Int64) — object point inside any zone polygon;
    - ``zouit_types`` (Utf8) — sorted ``;``-joined categories of the
      intersecting zones, null when the object is outside all zones;
    - ``inside_water_protection`` (Int64) — subset flag, separate
      because of frequency (ADR-0025 п. 3).

    Objects with null coordinates — and every object when the zone layer
    is empty — get null feature values.
    """
    if objects.height == 0:
        return pl.DataFrame(schema=_ZOUIT_OUTPUT_SCHEMA)

    n = objects.height
    inside_values: list[int | None] = [None] * n
    types_values: list[str | None] = [None] * n
    water_values: list[int | None] = [None] * n

    zone_geoms: list[BaseGeometry] = []
    zone_categories: list[str] = []
    for wkt_str, category in zones.select(["geometry_wkt_3857", "category"]).iter_rows():
        if wkt_str is None or category is None:
            continue
        geom = shapely.wkt.loads(wkt_str)
        if geom.is_empty:
            continue
        zone_geoms.append(_project_3857_to_utm(geom))
        zone_categories.append(str(category))

    obj_lats = objects["lat"].to_numpy()
    obj_lons = objects["lon"].to_numpy()
    valid = ~(np.isnan(obj_lats) | np.isnan(obj_lons))

    if zone_geoms and valid.any():
        obj_xs, obj_ys = _WGS84_TO_UTM.transform(obj_lons[valid], obj_lats[valid])
        points = cast(np.ndarray[Any, Any], shapely.points(np.asarray(obj_xs), np.asarray(obj_ys)))
        valid_idx = np.flatnonzero(valid)

        zone_arr = np.asarray(zone_geoms, dtype=object)
        hits = shapely.STRtree(zone_arr).query(points, predicate="intersects")
        # point position → sorted category set of intersecting zones.
        cats_by_pos: dict[int, set[str]] = {}
        if hits.size:
            for pos, zone_i in zip(hits[0].tolist(), hits[1].tolist(), strict=True):
                cats_by_pos.setdefault(int(pos), set()).add(zone_categories[int(zone_i)])
        for pos, row_i in enumerate(valid_idx.tolist()):
            cats = cats_by_pos.get(pos)
            if cats:
                inside_values[row_i] = 1
                types_values[row_i] = ";".join(sorted(cats))
                water_values[row_i] = 1 if "water_protection" in cats else 0
            else:
                inside_values[row_i] = 0
                water_values[row_i] = 0

    return pl.DataFrame(
        {
            "object_id": objects["object_id"],
            "inside_zouit": pl.Series(inside_values, dtype=pl.Int64),
            "zouit_types": pl.Series(types_values, dtype=pl.Utf8),
            "inside_water_protection": pl.Series(water_values, dtype=pl.Int64),
        }
    )


def join_zouit_features(objects: pl.DataFrame, zouit_features: pl.DataFrame) -> pl.DataFrame:
    """LEFT JOIN the silver per-object ЗОУИТ table by ``object_id``.

    Idempotent: pre-existing feature columns on ``objects`` are dropped
    first (the store is read-write; a rerun reads its own enriched
    output). Objects absent from ``zouit_features`` get nulls; an empty
    ``zouit_features`` yields all-null feature columns.
    """
    drop_existing = [c for c in ZOUIT_FEATURE_COLUMNS if c in objects.columns]
    if drop_existing:
        objects = objects.drop(drop_existing)

    if zouit_features.is_empty():
        return objects.with_columns(
            [
                pl.lit(None, dtype=pl.Int64).alias("inside_zouit"),
                pl.lit(None, dtype=pl.Utf8).alias("zouit_types"),
                pl.lit(None, dtype=pl.Int64).alias("inside_water_protection"),
            ]
        )

    right = zouit_features.select(["object_id", *ZOUIT_FEATURE_COLUMNS])
    return objects.join(right, on="object_id", how="left")
