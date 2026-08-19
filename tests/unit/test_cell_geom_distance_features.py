"""Unit tests for ADR-0027 — distance ЦОФ measured at cell centres.

The cell helper must produce the same distances as the object helper
measured at the cell centre, but keyed by ``h3_index`` instead of a
per-object lat/lon.
"""

from __future__ import annotations

import h3
import polars as pl
from shapely.geometry import Polygon, box

from kadastra.etl.cell_geom_distance_features import (
    compute_cell_geom_distance_features,
)
from kadastra.etl.h3_coverage import h3_cells_to_latlng
from kadastra.etl.object_geom_distance_features import (
    compute_object_geom_distance_features,
)

_KAZAN_LAT = 55.7905
_KAZAN_LON = 49.1142


def _cell_frame(cell: str) -> pl.DataFrame:
    return pl.DataFrame({"h3_index": [cell]})


def _polygon_around(lat: float, lon: float, radius_deg: float) -> Polygon:
    return box(lon - radius_deg, lat - radius_deg, lon + radius_deg, lat + radius_deg)


def test_cell_distance_equals_object_distance_at_centre() -> None:
    cell = h3.latlng_to_cell(_KAZAN_LAT, _KAZAN_LON, 10)
    lat, lon = h3_cells_to_latlng([cell])[0]
    poly = _polygon_around(_KAZAN_LAT, _KAZAN_LON + 0.01, 0.001)  # ~110 m square east

    cell_df = compute_cell_geom_distance_features(_cell_frame(cell), geometries_by_layer={"water": [poly]})
    obj_df = compute_object_geom_distance_features(
        pl.DataFrame({"lat": [lat], "lon": [lon]}),
        geometries_by_layer={"water": [poly]},
    )

    assert abs(float(cell_df["dist_to_water_m"][0]) - float(obj_df["dist_to_water_m"][0])) < 1e-6


def test_empty_layer_yields_null_column() -> None:
    cell = h3.latlng_to_cell(_KAZAN_LAT, _KAZAN_LON, 10)

    df = compute_cell_geom_distance_features(_cell_frame(cell), geometries_by_layer={"water": []})

    assert "dist_to_water_m" in df.columns
    assert df["dist_to_water_m"][0] is None


def test_drops_latlon_and_keeps_h3_index() -> None:
    cell = h3.latlng_to_cell(_KAZAN_LAT, _KAZAN_LON, 10)
    poly = _polygon_around(_KAZAN_LAT, _KAZAN_LON, 0.001)

    df = compute_cell_geom_distance_features(_cell_frame(cell), geometries_by_layer={"water": [poly]})

    assert df.columns == ["h3_index", "dist_to_water_m"]


def test_empty_cells_frame_emits_null_columns() -> None:
    empty = pl.DataFrame({"h3_index": []}, schema={"h3_index": pl.Utf8})

    df = compute_cell_geom_distance_features(
        empty, geometries_by_layer={"water": [_polygon_around(_KAZAN_LAT, _KAZAN_LON, 0.001)]}
    )

    assert "dist_to_water_m" in df.columns
    assert df.height == 0


def test_no_layers_returns_unchanged_frame() -> None:
    cell = h3.latlng_to_cell(_KAZAN_LAT, _KAZAN_LON, 10)
    frame = _cell_frame(cell)

    df = compute_cell_geom_distance_features(frame, geometries_by_layer={})

    assert df.columns == frame.columns
