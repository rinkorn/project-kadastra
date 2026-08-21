"""Tests for build_road_graph_edges_from_overpass.

Pure-function ETL that converts Overpass-API JSON into the edges-table
DataFrame consumed by NetworkxRoadGraph.from_parquet.

Overpass JSON shape (subset we care about): ::

    {
      "elements": [
        {"type": "way", "id": 123, "geometry": [{"lat": ..., "lon": ...}, ...]},
        ...
      ]
    }

The function turns each way into ``len(geometry) - 1`` consecutive
edges with haversine length in meters, and concatenates across ways.
"""

from __future__ import annotations

import math

import polars as pl
import pytest

from kadastra.etl.haversine import haversine_meters
from kadastra.etl.road_graph_edges_from_overpass import (
    build_road_graph_edges_from_overpass,
)


def _way(*coords: tuple[float, float], tags: dict[str, str] | None = None) -> dict[str, object]:
    element: dict[str, object] = {
        "type": "way",
        "id": 1,
        "geometry": [{"lat": lat, "lon": lon} for lat, lon in coords],
    }
    if tags is not None:
        element["tags"] = tags
    return element


def test_single_way_with_three_nodes_yields_two_edges() -> None:
    a = (55.78, 49.12)
    b = (55.79, 49.13)
    c = (55.80, 49.14)

    edges = build_road_graph_edges_from_overpass({"elements": [_way(a, b, c)]})

    assert edges.height == 2
    assert set(edges.columns) == {
        "from_lat",
        "from_lon",
        "to_lat",
        "to_lon",
        "length_m",
        "highway",
        "bridge",
        "tunnel",
        "layer",
    }
    rows = edges.rows()
    assert rows[0][:4] == (a[0], a[1], b[0], b[1])
    assert rows[1][:4] == (b[0], b[1], c[0], c[1])
    assert rows[0][4] == pytest.approx(haversine_meters(*a, *b), rel=1e-9)
    assert rows[1][4] == pytest.approx(haversine_meters(*b, *c), rel=1e-9)


def test_multiple_ways_concatenated() -> None:
    way1 = _way((0.0, 0.0), (0.001, 0.0))
    way2 = _way((1.0, 1.0), (1.001, 1.0), (1.002, 1.0))

    edges = build_road_graph_edges_from_overpass({"elements": [way1, way2]})

    # 1 + 2 = 3 edges
    assert edges.height == 3


def test_way_without_geometry_skipped() -> None:
    edges = build_road_graph_edges_from_overpass(
        {
            "elements": [
                {"type": "way", "id": 1},  # no 'geometry' key
                _way((0.0, 0.0), (0.001, 0.0)),
            ]
        }
    )
    assert edges.height == 1


def test_way_with_single_node_skipped() -> None:
    edges = build_road_graph_edges_from_overpass(
        {
            "elements": [
                _way((0.0, 0.0)),  # only one node — no edge possible
                _way((0.0, 0.0), (0.001, 0.0)),
            ]
        }
    )
    assert edges.height == 1


def test_node_elements_ignored() -> None:
    edges = build_road_graph_edges_from_overpass(
        {
            "elements": [
                {"type": "node", "id": 1, "lat": 0.0, "lon": 0.0},
                _way((0.0, 0.0), (0.001, 0.0)),
            ]
        }
    )
    assert edges.height == 1


def test_empty_payload_returns_empty_df_with_correct_schema() -> None:
    edges = build_road_graph_edges_from_overpass({"elements": []})

    assert edges.height == 0
    assert set(edges.columns) == {
        "from_lat",
        "from_lon",
        "to_lat",
        "to_lon",
        "length_m",
        "highway",
        "bridge",
        "tunnel",
        "layer",
    }
    assert edges.schema["from_lat"] == pl.Float64
    assert edges.schema["length_m"] == pl.Float64


def test_zero_length_edge_kept_for_traceability() -> None:
    """A way with two coincident nodes is unusual but real (digitization
    artifact). The function shouldn't drop it silently — keeping it as
    a 0 m edge means it doesn't break shortest-path and lets the edge
    show up in counts/audits.
    """
    a = (55.78, 49.12)
    edges = build_road_graph_edges_from_overpass({"elements": [_way(a, a)]})

    assert edges.height == 1
    assert edges["length_m"][0] == 0.0 or math.isclose(edges["length_m"][0], 0.0, abs_tol=1e-6)


def test_way_tags_are_carried_into_tag_columns() -> None:
    """ADR-0030: edges.parquet must keep the source way's highway/bridge/
    tunnel/layer tags so the water filter and ADR-0024 road-class features
    can work off the graph artifact without a separate OSM extract."""
    a = (55.78, 49.12)
    b = (55.79, 49.13)
    way = _way(a, b, tags={"highway": "footway", "bridge": "yes", "layer": "1"})

    edges = build_road_graph_edges_from_overpass({"elements": [way]})

    assert edges.height == 1
    assert edges["highway"][0] == "footway"
    assert edges["bridge"][0] == "yes"
    assert edges["tunnel"][0] is None
    assert edges["layer"][0] == "1"


def test_way_without_tags_yields_null_tag_columns() -> None:
    a = (55.78, 49.12)
    b = (55.79, 49.13)

    edges = build_road_graph_edges_from_overpass({"elements": [_way(a, b)]})

    assert edges.height == 1
    for col in ("highway", "bridge", "tunnel", "layer"):
        assert edges[col][0] is None
        assert edges.schema[col] == pl.Utf8
