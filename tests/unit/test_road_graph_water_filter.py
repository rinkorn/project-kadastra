"""Tests for filter_water_crossing_edges (ADR-0030).

Drops road-graph edges whose segment runs across water polygons for
more than the threshold (default 30 m) unless the source OSM way is a
bridge or a tunnel — phantom ferry-like crossings over river channels
must not survive into the pedestrian graph, while real bridges must.
"""

from __future__ import annotations

import polars as pl
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry

from kadastra.etl.haversine import haversine_meters
from kadastra.etl.road_graph_water_filter import (
    DEFAULT_MAX_WATER_CROSSING_M,
    filter_water_crossing_edges,
)

_LAT = 55.79  # Kazan latitude; 1° lon ≈ 62.6 km here, 1° lat ≈ 111.3 km
_LON = 49.12


def _water_square(lon_west: float, lon_east: float) -> BaseGeometry:
    """A water polygon straddling the east-west test edge (~0.002° tall)."""
    return Polygon(
        [
            (lon_west, _LAT - 0.001),
            (lon_east, _LAT - 0.001),
            (lon_east, _LAT + 0.001),
            (lon_west, _LAT + 0.001),
        ]
    )


def _edges(*, bridge: str | None = None, tunnel: str | None = None) -> pl.DataFrame:
    from_lat, from_lon = _LAT, _LON - 0.01
    to_lat, to_lon = _LAT, _LON + 0.01  # ~1250 m due east
    return pl.DataFrame(
        {
            "from_lat": [from_lat],
            "from_lon": [from_lon],
            "to_lat": [to_lat],
            "to_lon": [to_lon],
            "length_m": [haversine_meters(from_lat, from_lon, to_lat, to_lon)],
            "highway": ["footway"],
            "bridge": [bridge],
            "tunnel": [tunnel],
            "layer": [None],
        },
        schema={
            "from_lat": pl.Float64,
            "from_lon": pl.Float64,
            "to_lat": pl.Float64,
            "to_lon": pl.Float64,
            "length_m": pl.Float64,
            "highway": pl.Utf8,
            "bridge": pl.Utf8,
            "tunnel": pl.Utf8,
            "layer": pl.Utf8,
        },
    )


def test_edge_crossing_water_beyond_threshold_is_dropped() -> None:
    # 0.002° ≈ 125 m of the segment inside the polygon — way over 30 m.
    water = [_water_square(_LON - 0.001, _LON + 0.001)]

    out = filter_water_crossing_edges(_edges(), water)

    assert out.height == 0


def test_bridge_edge_crossing_water_is_kept() -> None:
    water = [_water_square(_LON - 0.001, _LON + 0.001)]

    out = filter_water_crossing_edges(_edges(bridge="yes"), water)

    assert out.height == 1


def test_tunnel_edge_crossing_water_is_kept() -> None:
    water = [_water_square(_LON - 0.001, _LON + 0.001)]

    out = filter_water_crossing_edges(_edges(tunnel="yes"), water)

    assert out.height == 1


def test_bridge_no_tag_does_not_exempt_edge() -> None:
    water = [_water_square(_LON - 0.001, _LON + 0.001)]

    out = filter_water_crossing_edges(_edges(bridge="no"), water)

    assert out.height == 0


def test_short_water_crossing_below_threshold_is_kept() -> None:
    # 0.0003° ≈ 19 m inside the polygon — under the 30 m threshold.
    water = [_water_square(_LON - 0.00015, _LON + 0.00015)]
    assert DEFAULT_MAX_WATER_CROSSING_M == 30.0

    out = filter_water_crossing_edges(_edges(), water)

    assert out.height == 1


def test_edge_away_from_water_is_kept() -> None:
    water = [_water_square(_LON + 0.1, _LON + 0.102)]  # ~6 km east

    out = filter_water_crossing_edges(_edges(), water)

    assert out.height == 1


def test_empty_water_list_returns_edges_unchanged() -> None:
    edges = _edges()

    out = filter_water_crossing_edges(edges, [])

    assert out.height == 1
    assert out.columns == edges.columns


def test_empty_edges_return_empty_frame_with_schema() -> None:
    edges = _edges().head(0)

    out = filter_water_crossing_edges(edges, [_water_square(_LON - 0.001, _LON + 0.001)])

    assert out.height == 0
    assert out.columns == edges.columns
