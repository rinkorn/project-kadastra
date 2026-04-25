"""Walk a directory of NSPD page-NNNN.json files and assemble a polars frame.

This is the bulk-of-the-work step that sits between
``scripts/download_nspd_layer.py`` (which produces the JSON pages) and
the rest of the pipeline (which expects polars frames). The output
schema for buildings and landplots is fixed and explicit so polars
doesn't have to guess types from row 0.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import polars as pl

from kadastra.etl.parse_nspd_feature import (
    parse_nspd_building_feature,
    parse_nspd_landplot_feature,
)

_BUILDINGS_SCHEMA: dict[str, pl.DataType] = {
    "geom_data_id": pl.Int64,
    "cad_num": pl.Utf8,
    "purpose": pl.Utf8,
    "asset_class": pl.Utf8,
    "lat": pl.Float64,
    "lon": pl.Float64,
    "area_m2": pl.Float64,
    "cost_value_rub": pl.Float64,
    "cost_index_rub_per_m2": pl.Float64,
    "year_built": pl.Int64,
    "floors": pl.Int64,
    "underground_floors": pl.Int64,
    "materials": pl.Utf8,
    "ownership_type": pl.Utf8,
    "registration_date": pl.Utf8,
    "readable_address": pl.Utf8,
    "polygon_wkt_3857": pl.Utf8,
}

_LANDPLOTS_SCHEMA: dict[str, pl.DataType] = {
    "geom_data_id": pl.Int64,
    "cad_num": pl.Utf8,
    "asset_class": pl.Utf8,
    "lat": pl.Float64,
    "lon": pl.Float64,
    "area_m2": pl.Float64,
    "cost_value_rub": pl.Float64,
    "cost_index_rub_per_m2": pl.Float64,
    "land_record_category_type": pl.Utf8,
    "land_record_subtype": pl.Utf8,
    "ownership_type": pl.Utf8,
    "registration_date": pl.Utf8,
    "readable_address": pl.Utf8,
    "polygon_wkt_3857": pl.Utf8,
}


def read_nspd_buildings_dir(directory: Path) -> pl.DataFrame:
    raise NotImplementedError


def read_nspd_landplots_dir(directory: Path) -> pl.DataFrame:
    raise NotImplementedError
