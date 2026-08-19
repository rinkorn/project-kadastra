"""Unit tests for ADR-0027 — BuildCellPolygonFeatures usecase."""

from __future__ import annotations

import json
from pathlib import Path

import h3
import polars as pl
from shapely.geometry import box, mapping
from shapely.geometry.base import BaseGeometry

from kadastra.usecases.build_cell_polygon_features import BuildCellPolygonFeatures

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


def test_execute_saves_poly_area_share_keyed_by_h3_index(tmp_path: Path) -> None:
    coverage_reader = FakeCoverageReader({10: _coverage(10)})
    feature_store = FakeFeatureStore()
    water_path = _write_layer(tmp_path, "water.geojsonseq", [box(49.0, 55.0, 49.5, 56.0)])

    usecase = BuildCellPolygonFeatures(
        coverage_reader=coverage_reader,
        feature_store=feature_store,
        poly_area_layer_paths={"water": water_path},
        radii_m=[500],
    )
    usecase.execute("RU-KAZAN-AGG", resolution=10)

    assert len(feature_store.saved) == 1
    region, resolution, feature_set, df = feature_store.saved[0]
    assert region == "RU-KAZAN-AGG"
    assert resolution == 10
    assert feature_set == "poly_area"
    assert {"h3_index", "resolution", "water_share_500m"} <= set(df.columns)
    assert "lat" not in df.columns and "lon" not in df.columns
    assert df.height == 1
