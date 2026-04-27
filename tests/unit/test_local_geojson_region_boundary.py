import json
from pathlib import Path
from typing import Any

import pytest

from kadastra.adapters.local_geojson_region_boundary import LocalGeoJsonRegionBoundary


def _square_polygon(min_lon: float, min_lat: float, max_lon: float, max_lat: float) -> dict[str, Any]:
    return {
        "type": "Polygon",
        "coordinates": [
            [
                [min_lon, min_lat],
                [max_lon, min_lat],
                [max_lon, max_lat],
                [min_lon, max_lat],
                [min_lon, min_lat],
            ]
        ],
    }


def _write_geojson(path: Path, features: list[dict[str, Any]]) -> None:
    path.write_text(json.dumps({"type": "FeatureCollection", "features": features}))


def test_get_boundary_returns_geometry_for_matching_region(tmp_path: Path) -> None:
    geojson = tmp_path / "regions.geojson"
    _write_geojson(
        geojson,
        [
            {
                "type": "Feature",
                "properties": {"shapeISO": "RU-TA"},
                "geometry": _square_polygon(49.0, 55.0, 50.0, 56.0),
            },
            {
                "type": "Feature",
                "properties": {"shapeISO": "RU-MO"},
                "geometry": _square_polygon(37.0, 55.0, 38.0, 56.0),
            },
        ],
    )
    adapter = LocalGeoJsonRegionBoundary(geojson)

    boundary = adapter.get_boundary("RU-TA")

    minx, miny, maxx, maxy = boundary.bounds
    assert (minx, miny, maxx, maxy) == (49.0, 55.0, 50.0, 56.0)


def test_get_boundary_raises_keyerror_for_missing_region(tmp_path: Path) -> None:
    geojson = tmp_path / "regions.geojson"
    _write_geojson(
        geojson,
        [
            {
                "type": "Feature",
                "properties": {"shapeISO": "RU-TA"},
                "geometry": _square_polygon(49.0, 55.0, 50.0, 56.0),
            }
        ],
    )
    adapter = LocalGeoJsonRegionBoundary(geojson)

    with pytest.raises(KeyError):
        adapter.get_boundary("RU-XX")


def test_get_boundary_supports_custom_region_code_field(tmp_path: Path) -> None:
    geojson = tmp_path / "regions.geojson"
    _write_geojson(
        geojson,
        [
            {
                "type": "Feature",
                "properties": {"iso_3166_2": "RU-TA"},
                "geometry": _square_polygon(49.0, 55.0, 50.0, 56.0),
            }
        ],
    )
    adapter = LocalGeoJsonRegionBoundary(geojson, region_code_field="iso_3166_2")

    boundary = adapter.get_boundary("RU-TA")

    assert boundary.area > 0
