"""Unit tests for ADR-0027 — BuildCellMetroFeatures usecase."""

from __future__ import annotations

import h3
import numpy as np
import polars as pl

from kadastra.usecases.build_cell_metro_features import BuildCellMetroFeatures

_KAZAN_LAT = 55.7887
_KAZAN_LON = 49.1221


class FakeCoverageReader:
    def __init__(self, by_resolution: dict[int, pl.DataFrame]) -> None:
        self._by_resolution = by_resolution

    def load(self, region_code: str, resolution: int) -> pl.DataFrame:
        return self._by_resolution[resolution]


class FakeFeatureStore:
    def __init__(self) -> None:
        self.saved: list[tuple[str, int, str, pl.DataFrame]] = []

    def save(self, region_code: str, resolution: int, feature_set: str, df: pl.DataFrame) -> None:
        self.saved.append((region_code, resolution, feature_set, df))


class FakeRawData:
    def __init__(self, payloads: dict[str, bytes]) -> None:
        self._payloads = payloads

    def read_bytes(self, key: str) -> bytes:
        return self._payloads[key]

    def list_keys(self, prefix: str) -> list[str]:
        return [k for k in self._payloads if k.startswith(prefix)]


class FakeRoadGraph:
    def distance_matrix_m(
        self,
        from_coords: list[tuple[float, float]],
        to_coords: list[tuple[float, float]],
    ) -> np.ndarray:
        return np.full((len(from_coords), len(to_coords)), 300.0, dtype=np.float64)


def _coverage(resolution: int) -> pl.DataFrame:
    cell = h3.latlng_to_cell(_KAZAN_LAT, _KAZAN_LON, resolution)
    return pl.DataFrame({"h3_index": [cell], "resolution": [resolution]})


def _stations_csv() -> bytes:
    return pl.DataFrame({"lat": [_KAZAN_LAT], "lon": [_KAZAN_LON]}).write_csv().encode()


def test_execute_saves_metro_keyed_by_h3_index() -> None:
    coverage_reader = FakeCoverageReader({10: _coverage(10)})
    feature_store = FakeFeatureStore()
    raw_data = FakeRawData({"stations": _stations_csv(), "entrances": _stations_csv()})

    usecase = BuildCellMetroFeatures(
        coverage_reader=coverage_reader,
        feature_store=feature_store,
        raw_data=raw_data,
        road_graph=FakeRoadGraph(),
        stations_key="stations",
        entrances_key="entrances",
    )
    usecase.execute("RU-KAZAN-AGG", resolution=10)

    assert len(feature_store.saved) == 1
    region, resolution, feature_set, df = feature_store.saved[0]
    assert region == "RU-KAZAN-AGG"
    assert resolution == 10
    assert feature_set == "metro"
    assert {
        "h3_index",
        "resolution",
        "dist_metro_m",
        "dist_entrance_m",
        "count_stations_1km",
        "count_entrances_500m",
    } <= set(df.columns)
    assert "lat" not in df.columns and "lon" not in df.columns
    assert float(df["dist_metro_m"][0]) == 300.0
