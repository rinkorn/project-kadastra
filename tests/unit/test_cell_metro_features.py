"""Unit tests for ADR-0027 — metro accessibility measured at cell centres."""

from __future__ import annotations

import h3
import numpy as np
import polars as pl

from kadastra.etl.cell_metro_features import compute_cell_metro_features

_KAZAN_LAT = 55.7905
_KAZAN_LON = 49.1142


class _FakeRoadGraph:
    def distance_matrix_m(
        self,
        from_coords: list[tuple[float, float]],
        to_coords: list[tuple[float, float]],
    ) -> np.ndarray:
        return np.full((len(from_coords), len(to_coords)), 300.0, dtype=np.float64)

    def nearest_distance_m(
        self,
        from_coords: list[tuple[float, float]],
        to_coords: list[tuple[float, float]],
    ) -> np.ndarray:
        return self.distance_matrix_m(from_coords, to_coords).min(axis=1)


def _cell_frame(cell: str) -> pl.DataFrame:
    return pl.DataFrame({"h3_index": [cell]})


def _points() -> pl.DataFrame:
    return pl.DataFrame({"lat": [_KAZAN_LAT], "lon": [_KAZAN_LON]})


def test_cell_metro_uses_graph_distance() -> None:
    cell = h3.latlng_to_cell(_KAZAN_LAT, _KAZAN_LON, 10)

    df = compute_cell_metro_features(_cell_frame(cell), _points(), _points(), road_graph=_FakeRoadGraph())

    assert df.columns == [
        "h3_index",
        "dist_metro_m",
        "dist_entrance_m",
        "count_stations_1km",
        "count_entrances_500m",
    ]
    assert float(df["dist_metro_m"][0]) == 300.0
    assert float(df["dist_entrance_m"][0]) == 300.0
    assert int(df["count_stations_1km"][0]) == 1
    assert int(df["count_entrances_500m"][0]) == 1


def test_empty_cells_emits_null_columns() -> None:
    empty = pl.DataFrame({"h3_index": []}, schema={"h3_index": pl.Utf8})

    df = compute_cell_metro_features(empty, _points(), _points(), road_graph=_FakeRoadGraph())

    assert df.height == 0
    assert {"dist_metro_m", "dist_entrance_m", "count_stations_1km", "count_entrances_500m"} <= set(df.columns)
