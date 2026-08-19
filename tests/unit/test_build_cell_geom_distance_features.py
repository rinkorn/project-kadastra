"""Unit tests for ADR-0027 — BuildCellGeomDistanceFeatures usecase.

Verifies the Слой 1 grid-distance builder: loads coverage, computes
``dist_to_<layer>_m`` at cell centres, and stores the result keyed by
``h3_index`` under feature_set ``geom_distance``.
"""

from __future__ import annotations

import json
from pathlib import Path

import h3
import polars as pl
from shapely.geometry import box, mapping
from shapely.geometry.base import BaseGeometry

from kadastra.usecases.build_cell_geom_distance_features import (
    BuildCellGeomDistanceFeatures,
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


def _coverage(resolution: int) -> pl.DataFrame:
    cell = h3.latlng_to_cell(_KAZAN_LAT, _KAZAN_LON, resolution)
    return pl.DataFrame({"h3_index": [cell], "resolution": [resolution]})


def _write_layer(tmp_path: Path, name: str, geometries: list[BaseGeometry]) -> str:
    path = tmp_path / name
    with path.open("w", encoding="utf-8") as f:
        for geom in geometries:
            f.write(json.dumps({"geometry": mapping(geom)}) + "\n")
    return str(path)


def _make_usecase(coverage_reader: FakeCoverageReader, feature_store: FakeFeatureStore, tmp_path: Path):
    # Polygon covering the Kazan cell centre → distance 0.0 (inside).
    water_path = _write_layer(tmp_path, "water.geojsonseq", [box(49.0, 55.0, 49.5, 56.0)])
    return BuildCellGeomDistanceFeatures(
        coverage_reader=coverage_reader,
        feature_store=feature_store,
        geom_distance_layer_paths={"water": water_path},
    )


def test_execute_saves_geom_distance_keyed_by_h3_index(tmp_path: Path) -> None:
    coverage_reader = FakeCoverageReader({10: _coverage(10)})
    feature_store = FakeFeatureStore()
    usecase = _make_usecase(coverage_reader, feature_store, tmp_path)

    usecase.execute("RU-KAZAN-AGG", resolution=10)

    assert len(feature_store.saved) == 1
    region, resolution, feature_set, df = feature_store.saved[0]
    assert region == "RU-KAZAN-AGG"
    assert resolution == 10
    assert feature_set == "geom_distance"
    # Keyed by h3_index, no lat/lon leakage.
    assert {"h3_index", "resolution", "dist_to_water_m"} <= set(df.columns)
    assert "lat" not in df.columns and "lon" not in df.columns
    assert df.height == 1
    assert df["h3_index"][0] == h3.latlng_to_cell(_KAZAN_LAT, _KAZAN_LON, 10)


def test_execute_cell_inside_polygon_yields_zero_distance(tmp_path: Path) -> None:
    coverage_reader = FakeCoverageReader({10: _coverage(10)})
    feature_store = FakeFeatureStore()
    usecase = _make_usecase(coverage_reader, feature_store, tmp_path)

    usecase.execute("RU-KAZAN-AGG", resolution=10)

    _, _, _, df = feature_store.saved[0]
    assert float(df["dist_to_water_m"][0]) == 0.0


def test_execute_missing_layer_yields_null_column(tmp_path: Path) -> None:
    coverage_reader = FakeCoverageReader({10: _coverage(10)})
    feature_store = FakeFeatureStore()
    usecase = BuildCellGeomDistanceFeatures(
        coverage_reader=coverage_reader,
        feature_store=feature_store,
        geom_distance_layer_paths={"water": str(tmp_path / "missing.geojsonseq")},
    )

    usecase.execute("RU-KAZAN-AGG", resolution=10)

    _, _, _, df = feature_store.saved[0]
    assert "dist_to_water_m" in df.columns
    assert df["dist_to_water_m"][0] is None
