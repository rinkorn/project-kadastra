"""Unit tests for linear OSM waterway buffering (ADR-0032).

OSM maps most in-city rivers/streams as linear ``waterway=*`` ways, not
polygons. The water layer must include them as buffered polygons so
``dist_to_water_m`` / ``water_share_*`` and the road-graph water filter
(ADR-0030) see them. Buffering happens in UTM 39N; the result is stored
back in WGS84.
"""

from __future__ import annotations

import math
from typing import Any

import pytest
from pyproj import Transformer
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as shapely_transform

from kadastra.etl.water_linear_waterways import (
    buffer_linear_waterway_features,
    parse_width_m,
)

_TO_UTM = Transformer.from_crs("EPSG:4326", "EPSG:32639", always_xy=True)
_TO_WGS84 = Transformer.from_crs("EPSG:32639", "EPSG:4326", always_xy=True)

# Kazan city centre — the latitude the layer is built for.
_LAT = 55.79
_LON0 = 49.10


def _utm(geom: BaseGeometry) -> BaseGeometry:
    return shapely_transform(lambda x, y, z=None: _TO_UTM.transform(x, y), geom)


def _ew_line_coords(length_m: float) -> list[list[float]]:
    """East-west LineString coords of ~``length_m`` at Kazan latitude."""
    x0, y0 = _TO_UTM.transform(_LON0, _LAT)
    lon1, lat1 = _TO_WGS84.transform(x0 + length_m, y0)
    return [[_LON0, _LAT], [lon1, lat1]]


def _line_feature(coords: list[Any], **props: Any) -> dict[str, Any]:
    geom_type = "MultiLineString" if coords and isinstance(coords[0][0], list) else "LineString"
    return {
        "type": "Feature",
        "id": "w123",
        "geometry": {"type": geom_type, "coordinates": coords},
        "properties": props,
    }


def _polygon_feature(**props: Any) -> dict[str, Any]:
    return {
        "type": "Feature",
        "id": "w456",
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[49.10, 55.79], [49.11, 55.79], [49.11, 55.80], [49.10, 55.79]]],
        },
        "properties": props,
    }


def _buffered_half_width_m(feature: dict[str, Any]) -> float:
    """Measured half width: UTM distance from the source line to the
    resulting polygon boundary."""
    out = buffer_linear_waterway_features([feature])
    assert len(out) == 1
    poly = shape(out[0]["geometry"])
    assert poly.geom_type in ("Polygon", "MultiPolygon")
    line = shape(feature["geometry"])
    return _utm(poly).boundary.distance(_utm(line))


# — parse_width_m —


def test_parse_width_plain_number() -> None:
    assert parse_width_m("10") == 10.0


def test_parse_width_decimal() -> None:
    assert parse_width_m("10.5") == 10.5


def test_parse_width_decimal_comma() -> None:
    assert parse_width_m("10,5") == 10.5


def test_parse_width_with_units() -> None:
    assert parse_width_m("6.5 m") == 6.5
    assert parse_width_m("6.5m") == 6.5


@pytest.mark.parametrize("raw", [None, "", "unknown", "wide", "0", "0.0", "-3"])
def test_parse_width_unusable(raw: object) -> None:
    assert parse_width_m(raw) is None


# — buffering —


def test_river_without_width_buffers_to_default_20m_wide() -> None:
    feature = _line_feature(_ew_line_coords(200.0), waterway="river")
    assert _buffered_half_width_m(feature) == pytest.approx(10.0, abs=0.3)


def test_width_tag_overrides_class_default() -> None:
    feature = _line_feature(_ew_line_coords(200.0), waterway="river", width="6")
    assert _buffered_half_width_m(feature) == pytest.approx(3.0, abs=0.15)


def test_width_tag_with_units() -> None:
    feature = _line_feature(_ew_line_coords(200.0), waterway="river", width="6.5 m")
    assert _buffered_half_width_m(feature) == pytest.approx(3.25, abs=0.15)


def test_stream_default_half_width() -> None:
    feature = _line_feature(_ew_line_coords(200.0), waterway="stream")
    assert _buffered_half_width_m(feature) == pytest.approx(3.0, abs=0.15)


def test_canal_default_half_width() -> None:
    feature = _line_feature(_ew_line_coords(200.0), waterway="canal")
    assert _buffered_half_width_m(feature) == pytest.approx(8.0, abs=0.25)


@pytest.mark.parametrize("cls", ["ditch", "drain"])
def test_ditch_drain_default_half_width(cls: str) -> None:
    feature = _line_feature(_ew_line_coords(200.0), waterway=cls)
    assert _buffered_half_width_m(feature) == pytest.approx(1.5, abs=0.1)


def test_round_caps_area() -> None:
    """Round cap/join style: area of a straight segment buffer is
    ``L * 2r + π r²`` — flat caps would miss the π r² term (~14 % here)."""
    length_m = 100.0
    feature = _line_feature(_ew_line_coords(length_m), waterway="river")
    out = buffer_linear_waterway_features([feature])
    area_m2 = _utm(shape(out[0]["geometry"])).area
    expected = length_m * 20.0 + math.pi * 10.0**2
    assert area_m2 == pytest.approx(expected, rel=0.02)


def test_multilinestring_buffered() -> None:
    coords = [_ew_line_coords(100.0), _ew_line_coords(100.0)]
    feature = _line_feature(coords, waterway="stream")
    out = buffer_linear_waterway_features([feature])
    assert shape(out[0]["geometry"]).geom_type in ("Polygon", "MultiPolygon")


def test_result_stays_wgs84_and_keeps_properties() -> None:
    feature = _line_feature(_ew_line_coords(100.0), waterway="river", name="Казанка")
    out = buffer_linear_waterway_features([feature])
    poly = shape(out[0]["geometry"])
    minx, miny, maxx, maxy = poly.bounds
    assert 48.0 < minx < maxx < 51.0
    assert 55.0 < miny < maxy < 57.0
    assert out[0]["properties"]["waterway"] == "river"
    assert out[0]["properties"]["name"] == "Казанка"


# — pass-through —


def test_polygon_features_pass_through_untouched() -> None:
    feature = _polygon_feature(natural="water")
    out = buffer_linear_waterway_features([feature])
    assert len(out) == 1
    assert out[0] == feature


def test_non_waterway_line_passes_through_untouched() -> None:
    feature = _line_feature(_ew_line_coords(100.0), power="line")
    out = buffer_linear_waterway_features([feature])
    assert out[0] == feature


def test_unknown_waterway_class_without_width_passes_through() -> None:
    """Classes without a documented default half-width (e.g. ``weir``)
    are not guessable — left as lines rather than buffered with a made-up
    width."""
    feature = _line_feature(_ew_line_coords(100.0), waterway="weir")
    out = buffer_linear_waterway_features([feature])
    assert out[0] == feature
