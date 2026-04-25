"""Spatial postfilter for polars frames with lat/lon columns."""

from __future__ import annotations

import polars as pl
from shapely.geometry import Point
from shapely.geometry.base import BaseGeometry
from shapely.prepared import prep


def filter_inside_polygon(df: pl.DataFrame, polygon: BaseGeometry) -> pl.DataFrame:
    if df.is_empty():
        return df

    prepared = prep(polygon)
    lats = df["lat"].to_list()
    lons = df["lon"].to_list()

    mask = [
        lat is not None and lon is not None and prepared.contains(Point(lon, lat))
        for lat, lon in zip(lats, lons, strict=True)
    ]
    return df.filter(pl.Series(mask))
