"""Unit tests for ADR-0027 — share ЦОФ measured at cell centres."""

from __future__ import annotations

import h3
import polars as pl
from shapely.geometry import box

from kadastra.etl.cell_polygon_features import compute_cell_polygon_features

_KAZAN_LAT = 55.7905
_KAZAN_LON = 49.1142


def _cell_frame(cell: str) -> pl.DataFrame:
    return pl.DataFrame({"h3_index": [cell]})


def test_share_at_cell_centre_full_cover_is_one() -> None:
    cell = h3.latlng_to_cell(_KAZAN_LAT, _KAZAN_LON, 10)
    poly = box(_KAZAN_LON - 0.1, _KAZAN_LAT - 0.1, _KAZAN_LON + 0.1, _KAZAN_LAT + 0.1)

    df = compute_cell_polygon_features(_cell_frame(cell), polygons_by_layer={"water": [poly]}, radii_m=[500])

    assert "water_share_500m" in df.columns
    assert float(df["water_share_500m"][0]) > 0.99


def test_empty_layer_yields_zero_share() -> None:
    cell = h3.latlng_to_cell(_KAZAN_LAT, _KAZAN_LON, 10)

    df = compute_cell_polygon_features(_cell_frame(cell), polygons_by_layer={"water": []}, radii_m=[500])

    assert "water_share_500m" in df.columns
    # Empty layer → no polygons to cover the buffer → share 0.0 (matches
    # compute_object_polygon_features, unlike distance's null-for-empty).
    assert float(df["water_share_500m"][0]) == 0.0


def test_drops_latlon_and_keeps_h3_index() -> None:
    cell = h3.latlng_to_cell(_KAZAN_LAT, _KAZAN_LON, 10)
    poly = box(_KAZAN_LON - 0.1, _KAZAN_LAT - 0.1, _KAZAN_LON + 0.1, _KAZAN_LAT + 0.1)

    df = compute_cell_polygon_features(_cell_frame(cell), polygons_by_layer={"water": [poly]}, radii_m=[500])

    assert df.columns == ["h3_index", "water_share_500m"]


def test_no_layers_returns_unchanged_frame() -> None:
    cell = h3.latlng_to_cell(_KAZAN_LAT, _KAZAN_LON, 10)
    frame = _cell_frame(cell)

    df = compute_cell_polygon_features(frame, polygons_by_layer={}, radii_m=[500])

    assert df.columns == frame.columns
