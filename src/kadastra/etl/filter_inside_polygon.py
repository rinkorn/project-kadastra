"""Spatial postfilter for polars frames with lat/lon columns."""

from __future__ import annotations

import polars as pl
from shapely.geometry.base import BaseGeometry


def filter_inside_polygon(df: pl.DataFrame, polygon: BaseGeometry) -> pl.DataFrame:
    raise NotImplementedError
