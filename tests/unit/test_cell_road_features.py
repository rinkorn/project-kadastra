"""Unit tests for ADR-0027 — road density measured at cell centres."""

from __future__ import annotations

from typing import Any

import h3
import polars as pl

from kadastra.etl.cell_road_features import compute_cell_road_features

_KAZAN_LAT = 55.7905
_KAZAN_LON = 49.1142


def _cell_frame(cell: str) -> pl.DataFrame:
    return pl.DataFrame({"h3_index": [cell]})


def _way(coords: list[tuple[float, float]]) -> dict[str, Any]:
    return {"type": "way", "geometry": [{"lat": lat, "lon": lon} for lat, lon in coords]}


def test_road_length_within_buffer() -> None:
    cell = h3.latlng_to_cell(_KAZAN_LAT, _KAZAN_LON, 10)
    ways = [_way([(_KAZAN_LAT, _KAZAN_LON), (_KAZAN_LAT, _KAZAN_LON + 0.001)])]

    df = compute_cell_road_features(_cell_frame(cell), ways=ways, radius_m=500)

    assert "road_length_500m" in df.columns
    assert float(df["road_length_500m"][0]) > 0.0


def test_no_roads_yields_zero() -> None:
    cell = h3.latlng_to_cell(_KAZAN_LAT, _KAZAN_LON, 10)

    df = compute_cell_road_features(_cell_frame(cell), ways=[], radius_m=500)

    assert "road_length_500m" in df.columns
    assert float(df["road_length_500m"][0]) == 0.0


def test_drops_latlon_and_keeps_h3_index() -> None:
    cell = h3.latlng_to_cell(_KAZAN_LAT, _KAZAN_LON, 10)
    ways = [_way([(_KAZAN_LAT, _KAZAN_LON), (_KAZAN_LAT, _KAZAN_LON + 0.001)])]

    df = compute_cell_road_features(_cell_frame(cell), ways=ways, radius_m=500)

    assert df.columns == ["h3_index", "road_length_500m"]
