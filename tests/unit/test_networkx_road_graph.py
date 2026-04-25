"""Tests for NetworkxRoadGraph — KD-tree snap + Dijkstra over a NetworkX graph.

Covers the contract that compute_object_metro_features and other
graph-aware ZOF use: pairwise shortest-path distances in meters
between arbitrary lat/lon points, snapping each to the nearest
graph node and adding the snap distance.
"""

from __future__ import annotations

import math

import pytest

from kadastra.adapters.networkx_road_graph import NetworkxRoadGraph
from kadastra.etl.haversine import haversine_meters


def test_distance_matrix_returns_graph_distance_through_intermediate_nodes() -> None:
    """Two endpoints with no direct edge — graph distance must route
    through the intermediate node, not take the euclidean shortcut.
    """
    a = (0.0, 0.0)
    b = (0.001, 0.0)  # ~111 m north of A
    c = (0.001, 0.001)  # ~111 m east of B (no direct A-C edge)

    graph = NetworkxRoadGraph.from_edges(
        [
            (a, b, haversine_meters(*a, *b)),
            (b, c, haversine_meters(*b, *c)),
        ]
    )

    out = graph.distance_matrix_m(from_coords=[a], to_coords=[c])

    assert out.shape == (1, 1)
    expected = haversine_meters(*a, *b) + haversine_meters(*b, *c)
    assert out[0, 0] == pytest.approx(expected, rel=1e-3)
    # Sanity: graph distance is strictly larger than the euclidean shortcut.
    assert out[0, 0] > haversine_meters(*a, *c)


def test_distance_matrix_returns_inf_for_disconnected_components() -> None:
    a = (0.0, 0.0)
    b = (0.001, 0.0)
    # A separate, disconnected segment far away.
    c = (10.0, 10.0)
    d = (10.001, 10.0)

    graph = NetworkxRoadGraph.from_edges(
        [
            (a, b, haversine_meters(*a, *b)),
            (c, d, haversine_meters(*c, *d)),
        ]
    )

    out = graph.distance_matrix_m(from_coords=[a], to_coords=[c])

    assert math.isinf(out[0, 0])


def test_distance_matrix_includes_snap_distance_to_nearest_node() -> None:
    """Querying a coord that doesn't sit on the graph: total distance
    must include the haversine snap from the query point to the
    nearest graph node, on both ends.
    """
    a = (0.0, 0.0)
    b = (0.001, 0.0)  # ~111 m
    graph = NetworkxRoadGraph.from_edges([(a, b, haversine_meters(*a, *b))])

    # Query points 0.0001° (~11 m) off the graph endpoints.
    a_off = (0.0001, 0.0)  # 11 m north of A
    b_off = (0.0009, 0.0)  # 11 m south of B (still snaps to B)

    out = graph.distance_matrix_m(from_coords=[a_off], to_coords=[b_off])

    expected_snap_a = haversine_meters(*a_off, *a)
    expected_snap_b = haversine_meters(*b_off, *b)
    expected_total = expected_snap_a + haversine_meters(*a, *b) + expected_snap_b
    assert out[0, 0] == pytest.approx(expected_total, rel=1e-2)


def test_distance_matrix_handles_batch_of_sources_and_targets() -> None:
    """Shape and ordering: out[i, j] is from_coords[i] -> to_coords[j].
    """
    a = (0.0, 0.0)
    b = (0.001, 0.0)
    c = (0.002, 0.0)
    edges = [
        (a, b, haversine_meters(*a, *b)),
        (b, c, haversine_meters(*b, *c)),
    ]
    graph = NetworkxRoadGraph.from_edges(edges)

    out = graph.distance_matrix_m(from_coords=[a, b], to_coords=[b, c])

    assert out.shape == (2, 2)
    # a -> b: ~111
    assert out[0, 0] == pytest.approx(haversine_meters(*a, *b), rel=1e-3)
    # a -> c: ~222
    assert out[0, 1] == pytest.approx(
        haversine_meters(*a, *b) + haversine_meters(*b, *c), rel=1e-3
    )
    # b -> b: 0
    assert out[1, 0] == pytest.approx(0.0, abs=1e-6)
    # b -> c: ~111
    assert out[1, 1] == pytest.approx(haversine_meters(*b, *c), rel=1e-3)


def test_empty_query_returns_zero_sized_matrix() -> None:
    graph = NetworkxRoadGraph.from_edges(
        [((0.0, 0.0), (0.001, 0.0), 111.0)]
    )

    out = graph.distance_matrix_m(from_coords=[], to_coords=[(0.0, 0.0)])
    assert out.shape == (0, 1)

    out = graph.distance_matrix_m(from_coords=[(0.0, 0.0)], to_coords=[])
    assert out.shape == (1, 0)


def test_from_parquet_loads_edges_table_and_builds_equivalent_graph(
    tmp_path,  # type: ignore[no-untyped-def]
) -> None:
    """Edges-table parquet is the persistence format Container loads at
    boot. Schema: (from_lat, from_lon, to_lat, to_lon, length_m) — one
    row per edge.
    """
    import polars as pl

    a = (0.0, 0.0)
    b = (0.001, 0.0)
    c = (0.001, 0.001)

    edges = pl.DataFrame(
        {
            "from_lat": [a[0], b[0]],
            "from_lon": [a[1], b[1]],
            "to_lat": [b[0], c[0]],
            "to_lon": [b[1], c[1]],
            "length_m": [
                haversine_meters(*a, *b),
                haversine_meters(*b, *c),
            ],
        }
    )
    path = tmp_path / "edges.parquet"
    edges.write_parquet(path)

    graph = NetworkxRoadGraph.from_parquet(path)

    out = graph.distance_matrix_m(from_coords=[a], to_coords=[c])
    expected = haversine_meters(*a, *b) + haversine_meters(*b, *c)
    assert out[0, 0] == pytest.approx(expected, rel=1e-3)
