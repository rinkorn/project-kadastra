"""Unit tests for ADR-0027 — self-free zonal density at cell centres."""

from __future__ import annotations

import h3
import polars as pl

from kadastra.etl.cell_zonal_features import compute_cell_zonal_features

_KAZAN_LAT = 55.7905
_KAZAN_LON = 49.1142


def _cell_frame(cell: str) -> pl.DataFrame:
    return pl.DataFrame({"h3_index": [cell]})


def test_counts_layer_points_within_radius() -> None:
    cell = h3.latlng_to_cell(_KAZAN_LAT, _KAZAN_LON, 10)
    layer = pl.DataFrame({"lat": [_KAZAN_LAT, _KAZAN_LAT + 1.0], "lon": [_KAZAN_LON, _KAZAN_LON]})

    df = compute_cell_zonal_features(_cell_frame(cell), layers={"stations": layer}, radii_m=[500])

    assert "stations_within_500m" in df.columns
    assert int(df["stations_within_500m"][0]) == 1  # only the nearby one


def test_self_free_ignores_object_id() -> None:
    """A cell has no self to exclude — a point layer carrying object_id
    must still count all points (no self-exclusion)."""
    cell = h3.latlng_to_cell(_KAZAN_LAT, _KAZAN_LON, 10)
    layer = pl.DataFrame({"object_id": ["apt-1"], "lat": [_KAZAN_LAT], "lon": [_KAZAN_LON]})

    df = compute_cell_zonal_features(_cell_frame(cell), layers={"apartments": layer}, radii_m=[500])

    assert int(df["apartments_within_500m"][0]) == 1  # counted, not excluded


def test_drops_latlon_and_keeps_h3_index() -> None:
    cell = h3.latlng_to_cell(_KAZAN_LAT, _KAZAN_LON, 10)
    layer = pl.DataFrame({"lat": [_KAZAN_LAT], "lon": [_KAZAN_LON]})

    df = compute_cell_zonal_features(_cell_frame(cell), layers={"stations": layer}, radii_m=[500])

    assert df.columns == ["h3_index", "stations_within_500m"]


def test_empty_layer_yields_zero_counts() -> None:
    cell = h3.latlng_to_cell(_KAZAN_LAT, _KAZAN_LON, 10)
    empty = pl.DataFrame({"lat": [], "lon": []}, schema={"lat": pl.Float64, "lon": pl.Float64})

    df = compute_cell_zonal_features(_cell_frame(cell), layers={"stations": empty}, radii_m=[500])

    assert "stations_within_500m" in df.columns
    assert int(df["stations_within_500m"][0]) == 0


def test_no_layers_returns_unchanged_frame() -> None:
    cell = h3.latlng_to_cell(_KAZAN_LAT, _KAZAN_LON, 10)
    frame = _cell_frame(cell)

    df = compute_cell_zonal_features(frame, layers={}, radii_m=[500])

    assert df.columns == frame.columns
