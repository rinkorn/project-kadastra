"""Unit tests for ADR-0027 — BuildCellGraphDistanceFeatures usecase."""

from __future__ import annotations

import json
from pathlib import Path

import h3
import numpy as np
import polars as pl
from shapely.geometry import Point, mapping

from kadastra.usecases.build_cell_graph_distance_features import (
    BuildCellGraphDistanceFeatures,
)

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


class FakeRoadGraph:
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


def _coverage(resolution: int) -> pl.DataFrame:
    cell = h3.latlng_to_cell(_KAZAN_LAT, _KAZAN_LON, resolution)
    return pl.DataFrame({"h3_index": [cell], "resolution": [resolution]})


def _write_point_layer(tmp_path: Path, name: str, points: list[tuple[float, float]]) -> str:
    path = tmp_path / name
    with path.open("w", encoding="utf-8") as f:
        for lat, lon in points:
            f.write(json.dumps({"geometry": mapping(Point(lon, lat))}) + "\n")
    return str(path)


def _make_usecase(coverage_reader: FakeCoverageReader, feature_store: FakeFeatureStore, tmp_path: Path):
    school_path = _write_point_layer(tmp_path, "school.geojsonseq", [(_KAZAN_LAT, _KAZAN_LON)])
    clinic_path = _write_point_layer(tmp_path, "clinic.geojsonseq", [(_KAZAN_LAT, _KAZAN_LON)])
    return BuildCellGraphDistanceFeatures(
        coverage_reader=coverage_reader,
        feature_store=feature_store,
        road_graph=FakeRoadGraph(),
        layer_paths={"school": school_path, "clinic": clinic_path},
        layer_names=["school", "clinic"],
    )


def test_execute_saves_walk_dist_keyed_by_h3_index(tmp_path: Path) -> None:
    coverage_reader = FakeCoverageReader({10: _coverage(10)})
    feature_store = FakeFeatureStore()
    usecase = _make_usecase(coverage_reader, feature_store, tmp_path)

    usecase.execute("RU-KAZAN-AGG", resolution=10)

    assert len(feature_store.saved) == 1
    region, resolution, feature_set, df = feature_store.saved[0]
    assert region == "RU-KAZAN-AGG"
    assert resolution == 10
    assert feature_set == "walk_dist"
    assert {"h3_index", "resolution", "walk_dist_to_school_m", "walk_dist_to_clinic_m"} <= set(df.columns)
    assert "lat" not in df.columns and "lon" not in df.columns
    assert float(df["walk_dist_to_school_m"][0]) == 300.0


def test_execute_missing_layer_yields_null_column(tmp_path: Path) -> None:
    coverage_reader = FakeCoverageReader({10: _coverage(10)})
    feature_store = FakeFeatureStore()
    usecase = BuildCellGraphDistanceFeatures(
        coverage_reader=coverage_reader,
        feature_store=feature_store,
        road_graph=FakeRoadGraph(),
        layer_paths={"school": str(tmp_path / "missing.geojsonseq")},
        layer_names=["school"],
    )

    usecase.execute("RU-KAZAN-AGG", resolution=10)

    _, _, _, df = feature_store.saved[0]
    assert "walk_dist_to_school_m" in df.columns
    assert df["walk_dist_to_school_m"][0] is None
