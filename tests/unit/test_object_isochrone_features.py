"""Tests for the ADR-0024 group-2 isochrone cache and its join.

``compute_isochrone_cache`` is exercised on a tiny synthetic
NetworkxRoadGraph: a 4-node chain n0—n1—n2—n3 with 200/200/1000 m
edges, so a 1200 m cutoff reaches n0..n2 from n0's cell and all four
nodes from n2's cell. ``join_isochrone_features`` is tested on a
synthetic cache frame.
"""

import h3
import polars as pl

from kadastra.adapters.networkx_road_graph import NetworkxRoadGraph
from kadastra.etl.object_isochrone_features import (
    ISO15_FEATURE_COLUMNS,
    ISOCHRONE_CACHE_SCHEMA,
    compute_isochrone_cache,
    join_isochrone_features,
)

N0 = (55.7887, 49.1221)
N1 = (55.7887, 49.1250)
N2 = (55.7887, 49.1279)
N3 = (55.7887, 49.1350)

RES = 11
CELL0 = h3.latlng_to_cell(*N0, RES)
CELL1 = h3.latlng_to_cell(*N1, RES)
CELL2 = h3.latlng_to_cell(*N2, RES)
CELL3 = h3.latlng_to_cell(*N3, RES)
# ~88 km from the graph — snap alone exceeds any walking cutoff.
FAR_CELL = h3.latlng_to_cell(55.0, 49.0, RES)

CELL_POPULATION = {CELL0: 100.0, CELL1: 200.0, CELL2: 300.0, CELL3: 400.0, FAR_CELL: 50.0}


def _graph() -> NetworkxRoadGraph:
    return NetworkxRoadGraph.from_edges(
        [
            (N0, N1, 200.0),
            (N1, N2, 200.0),
            (N2, N3, 1000.0),
        ]
    )


def _compute(cells: list[str]) -> pl.DataFrame:
    return compute_isochrone_cache(
        cells,
        road_graph=_graph(),
        cutoff_m=1200.0,
        resolution=RES,
        # Two POIs on the same node both count — we count POIs, not nodes.
        poi_coords=[N1, N1, N3],
        metro_coords=[N3],
        cell_population=CELL_POPULATION,
    )


def test_cache_row_within_cutoff() -> None:
    out = _compute([CELL0])

    row = out.filter(pl.col("h3_index") == CELL0).row(0, named=True)
    # From n0: 200 m to n1, 400 m to n2, 1400 m to n3 — n3 is out.
    assert row["iso15_amenity_count"] == 2
    assert row["iso15_metro_reach"] == 0
    assert row["iso15_pop_count"] == 600.0  # cells of n0+n1+n2


def test_cache_row_reaching_metro() -> None:
    out = _compute([CELL2])

    row = out.filter(pl.col("h3_index") == CELL2).row(0, named=True)
    # From n2: n3 is 1000 m away — inside the 1200 m cutoff.
    assert row["iso15_amenity_count"] == 3
    assert row["iso15_metro_reach"] == 1
    assert row["iso15_pop_count"] == 1000.0  # all four cells


def test_off_graph_cell_degenerates_to_itself() -> None:
    out = _compute([FAR_CELL])

    row = out.filter(pl.col("h3_index") == FAR_CELL).row(0, named=True)
    assert row["iso15_amenity_count"] == 0
    assert row["iso15_metro_reach"] == 0
    assert row["iso15_pop_count"] == 50.0  # own cell only


def test_empty_cells_yield_empty_frame_with_schema() -> None:
    out = _compute([])
    assert out.height == 0
    assert out.schema == pl.Schema(ISOCHRONE_CACHE_SCHEMA)


def _objects() -> pl.DataFrame:
    return pl.DataFrame(
        [
            {"object_id": "a", "asset_class": "apartment", "lat": N0[0], "lon": N0[1]},
            {"object_id": "b", "asset_class": "apartment", "lat": 56.5, "lon": 50.5},
            {"object_id": "c", "asset_class": "apartment", "lat": None, "lon": None},
        ],
        schema={"object_id": pl.Utf8, "asset_class": pl.Utf8, "lat": pl.Float64, "lon": pl.Float64},
    )


def _cache() -> pl.DataFrame:
    return pl.DataFrame(
        [(CELL0, 600.0, 2, 0)],
        schema=ISOCHRONE_CACHE_SCHEMA,
        orient="row",
    )


def test_join_produces_expected_columns() -> None:
    out = join_isochrone_features(_objects(), _cache(), resolution=RES)

    for col in ISO15_FEATURE_COLUMNS:
        assert col in out.columns
    row_a = out.filter(pl.col("object_id") == "a").row(0, named=True)
    assert row_a["iso15_pop_count"] == 600.0
    assert row_a["iso15_amenity_count"] == 2
    assert row_a["iso15_metro_reach"] == 0


def test_join_null_fallbacks() -> None:
    out = join_isochrone_features(_objects(), _cache(), resolution=RES)

    # Cell not present in the cache → nulls.
    row_b = out.filter(pl.col("object_id") == "b").row(0, named=True)
    # Null coordinates → nulls, no crash.
    row_c = out.filter(pl.col("object_id") == "c").row(0, named=True)
    for row in (row_b, row_c):
        for col in ISO15_FEATURE_COLUMNS:
            assert row[col] is None


def test_join_empty_cache_yields_null_columns() -> None:
    empty = pl.DataFrame(schema=ISOCHRONE_CACHE_SCHEMA)
    out = join_isochrone_features(_objects(), empty, resolution=RES)

    assert out.height == 3
    for col in ISO15_FEATURE_COLUMNS:
        assert out[col].null_count() == 3


def test_join_is_idempotent() -> None:
    once = join_isochrone_features(_objects(), _cache(), resolution=RES)
    twice = join_isochrone_features(once, _cache(), resolution=RES)

    assert not any(c.endswith("_right") for c in twice.columns)
    assert twice.columns == once.columns
    row_a = twice.filter(pl.col("object_id") == "a").row(0, named=True)
    assert row_a["iso15_pop_count"] == 600.0
