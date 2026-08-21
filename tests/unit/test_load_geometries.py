"""Unit tests for the shared GeoJSON-seq geometry loader."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest
from shapely.geometry import Point, box, mapping
from shapely.geometry.base import BaseGeometry

from kadastra.etl.load_geometries import (
    GeoJsonSeqLayerLoader,
    load_geojsonseq_geometries,
    load_geojsonseq_points,
    points_from_geometries,
)


def _write_geojsonseq(tmp_path: Path, name: str, geometries: list[BaseGeometry]) -> Path:
    path = tmp_path / name
    with path.open("w", encoding="utf-8") as f:
        for geom in geometries:
            f.write(json.dumps({"geometry": mapping(geom)}) + "\n")
    return path


def test_loads_geometries_from_geojsonseq(tmp_path: Path) -> None:
    poly = box(0, 0, 1, 1)
    path = _write_geojsonseq(tmp_path, "water.geojsonseq", [poly])

    layers = load_geojsonseq_geometries({"water": str(path)})

    assert len(layers["water"]) == 1
    assert layers["water"][0].equals(poly)


def test_missing_file_yields_empty_layer(tmp_path: Path) -> None:
    layers = load_geojsonseq_geometries({"water": str(tmp_path / "nope.geojsonseq")})

    assert layers == {"water": []}


def test_polygon_and_point_layers_load_together(tmp_path: Path) -> None:
    poly = box(0, 0, 1, 1)
    point = Point(0.5, 0.5)
    poly_path = _write_geojsonseq(tmp_path, "water.geojsonseq", [poly])
    point_path = _write_geojsonseq(tmp_path, "school.geojsonseq", [point])

    layers = load_geojsonseq_geometries({"water": str(poly_path), "school": str(point_path)})

    assert len(layers["water"]) == 1
    assert len(layers["school"]) == 1
    assert layers["school"][0].equals(point)


def test_empty_paths_dict_yields_empty_layers() -> None:
    assert load_geojsonseq_geometries({}) == {}


def _write_geojsonseq_raw(tmp_path: Path, name: str, lines: list[str]) -> Path:
    path = tmp_path / name
    path.write_text("".join(line + "\n" for line in lines), encoding="utf-8")
    return path


def test_points_from_geometries_point_passthrough_and_centroid() -> None:
    """Point features pass through unchanged; non-Point geometries are
    reduced to their centroid (same CRS — WGS84 lon/lat)."""
    df = points_from_geometries([Point(49.0, 55.0), box(0, 0, 2, 2)])

    assert df.schema == {"lat": pl.Float64, "lon": pl.Float64}
    assert df.row(0, named=True) == {"lat": 55.0, "lon": 49.0}
    assert df.row(1, named=True) == {"lat": 1.0, "lon": 1.0}


def test_points_from_geometries_skips_empty_geometries() -> None:
    df = points_from_geometries([Point(), Point(1.0, 2.0)])

    assert df.height == 1
    assert df.row(0, named=True) == {"lat": 2.0, "lon": 1.0}


def test_points_from_geometries_empty_input_yields_empty_frame() -> None:
    df = points_from_geometries([])

    assert df.is_empty()
    assert df.columns == ["lat", "lon"]


def test_load_geojsonseq_points_semantics(tmp_path: Path) -> None:
    """Baseline contract the loader-sharing refactor must preserve:
    Point as-is, Polygon → centroid, empty and null geometries skipped."""
    path = _write_geojsonseq_raw(
        tmp_path,
        "school.geojsonseq",
        [
            '{"type":"Feature","properties":{},"geometry":{"type":"Point","coordinates":[49.0,55.0]}}',
            '{"type":"Feature","properties":{},"geometry":{"type":"Polygon",'
            '"coordinates":[[[0,0],[2,0],[2,2],[0,2],[0,0]]]}}',
            '{"type":"Feature","properties":{},"geometry":{"type":"Point","coordinates":[]}}',
            '{"type":"Feature","properties":{},"geometry":null}',
        ],
    )

    df = load_geojsonseq_points(str(path))

    assert df.schema == {"lat": pl.Float64, "lon": pl.Float64}
    assert df.rows(named=True) == [
        {"lat": 55.0, "lon": 49.0},
        {"lat": 1.0, "lon": 1.0},
    ]


def test_load_geojsonseq_points_missing_file_yields_typed_empty_frame(tmp_path: Path) -> None:
    df = load_geojsonseq_points(str(tmp_path / "missing.geojsonseq"))

    assert df.is_empty()
    assert df.schema == {"lat": pl.Float64, "lon": pl.Float64}


def test_loader_load_points_shares_parse_with_load(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``load_points`` must transform the cached geometry list, not
    re-read the file: a second consumer of the same path costs zero
    disk parses."""
    import kadastra.etl.load_geometries as mod

    path = _write_geojsonseq(tmp_path, "school.geojsonseq", [Point(49.0, 55.0)])

    parse_calls: list[str] = []
    real_load = mod.load_geojsonseq_geometries

    def _counting_load(paths: dict[str, str]) -> dict[str, list[BaseGeometry]]:
        parse_calls.extend(paths.values())
        return real_load(paths)

    monkeypatch.setattr(mod, "load_geojsonseq_geometries", _counting_load)

    loader = GeoJsonSeqLayerLoader()
    points = loader.load_points(str(path))
    geoms = loader.load(str(path))
    points_again = loader.load_points(str(path))

    assert parse_calls == [str(path)]
    assert points.rows(named=True) == [{"lat": 55.0, "lon": 49.0}]
    assert points_again.rows(named=True) == points.rows(named=True)
    assert len(geoms) == 1


def test_loader_load_points_missing_file_yields_typed_empty_frame(tmp_path: Path) -> None:
    """Missing file → empty Float64 frame (zero counts downstream), same
    contract as ``load_geojsonseq_points``."""
    loader = GeoJsonSeqLayerLoader()

    df = loader.load_points(str(tmp_path / "missing.geojsonseq"))

    assert df.is_empty()
    assert df.schema == {"lat": pl.Float64, "lon": pl.Float64}
