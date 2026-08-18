"""Unit tests for the shared GeoJSON-seq geometry loader."""

from __future__ import annotations

import json
from pathlib import Path

from shapely.geometry import Point, box, mapping
from shapely.geometry.base import BaseGeometry

from kadastra.etl.load_geometries import load_geojsonseq_geometries


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
