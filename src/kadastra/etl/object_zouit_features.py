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

from typing import Any

import polars as pl

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


def categorize_zouit_type(type_zone: str | None) -> str:
    """Map a raw ``type_zone`` string to a :data:`ZOUIT_CATEGORIES` value.

    Substring rules over the lowercased string, ordered by specificity;
    anything unmatched (including null/empty) falls to ``other``.
    """
    raise NotImplementedError


def parse_zouit_feature(feature: dict[str, Any]) -> dict[str, Any] | None:
    """Parse one НСПД page-dump feature into a silver zone row.

    Returns ``None`` for features without geometry. The output dict
    follows :data:`ZOUIT_ZONE_SCHEMA` (category already mapped).
    """
    raise NotImplementedError


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
    raise NotImplementedError


def join_zouit_features(objects: pl.DataFrame, zouit_features: pl.DataFrame) -> pl.DataFrame:
    """LEFT JOIN the silver per-object ЗОУИТ table by ``object_id``.

    Idempotent: pre-existing feature columns on ``objects`` are dropped
    first (the store is read-write; a rerun reads its own enriched
    output). Objects absent from ``zouit_features`` get nulls; an empty
    ``zouit_features`` yields all-null feature columns.
    """
    raise NotImplementedError
