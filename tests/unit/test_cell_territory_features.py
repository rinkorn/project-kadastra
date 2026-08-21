"""Tests for per-cell territory features (ADR-0029 «OKTMO-пропагация»)."""

from __future__ import annotations

import h3
import polars as pl
from shapely.geometry import Polygon

from kadastra.etl.cell_territory_features import (
    TERRITORY_PROPAGATED_COLUMNS,
    compute_cell_territory_features,
)

# A fixed point inside Kazan and its res-10 cell.
_LAT, _LON = 55.79, 49.11
_CELL = h3.latlng_to_cell(_LAT, _LON, 10)


def _cells_frame(cells: list[str]) -> pl.DataFrame:
    rows = [(c, *h3.cell_to_latlng(c)) for c in cells]
    return pl.DataFrame(
        rows,
        schema={"h3_index": pl.Utf8, "lat": pl.Float64, "lon": pl.Float64},
        orient="row",
    )


def _objects_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "lat": [_LAT, _LAT, _LAT],
            "lon": [_LON, _LON, _LON],
            "oktmo_full": ["92701000001", "92701000001", "92701000002"],
            "settlement_name": ["Казань", "Казань", None],
            "kadnum_quarter": ["16:01:0001", "16:01:0001", "16:01:0002"],
        }
    )


def test_cell_with_objects_gets_modal_value() -> None:
    cells = _cells_frame([_CELL])
    out = compute_cell_territory_features(cells, _objects_frame())
    assert out["oktmo_full"][0] == "92701000001"
    assert out["settlement_name"][0] == "Казань"
    assert out["kadnum_quarter"][0] == "16:01:0001"


def test_cell_without_objects_inherits_parent_mode() -> None:
    # A neighbour cell in the same res-9 parent, itself empty.
    parent = h3.cell_to_parent(_CELL, 9)
    sibling = next(c for c in h3.cell_to_children(parent, 10) if c != _CELL)
    cells = _cells_frame([sibling])
    out = compute_cell_territory_features(cells, _objects_frame())
    assert out["oktmo_full"][0] == "92701000001"


def test_cell_outside_all_populated_parents_gets_null() -> None:
    # A cell on the other side of the region, no objects anywhere near.
    far_cell = h3.latlng_to_cell(55.2, 49.8, 10)
    cells = _cells_frame([far_cell])
    out = compute_cell_territory_features(cells, _objects_frame())
    assert out["oktmo_full"][0] is None


def test_all_propagated_columns_present() -> None:
    out = compute_cell_territory_features(_cells_frame([_CELL]), _objects_frame())
    for col in TERRITORY_PROPAGATED_COLUMNS:
        assert col in out.columns
    # Columns absent from the objects frame propagate as nulls.
    assert out["postal_index"][0] is None


def test_intra_city_raion_via_polygon_join() -> None:
    square = Polygon(
        [(_LON - 0.01, _LAT - 0.01), (_LON + 0.01, _LAT - 0.01), (_LON + 0.01, _LAT + 0.01), (_LON - 0.01, _LAT + 0.01)]
    )
    out = compute_cell_territory_features(
        _cells_frame([_CELL]),
        _objects_frame(),
        raion_polygons=[("Советский", square)],
    )
    assert out["intra_city_raion"][0] == "Советский"


def test_intra_city_raion_null_outside_polygons() -> None:
    square = Polygon([(49.0, 55.0), (49.01, 55.0), (49.01, 55.01), (49.0, 55.01)])
    out = compute_cell_territory_features(
        _cells_frame([_CELL]),
        _objects_frame(),
        raion_polygons=[("Вахитовский", square)],
    )
    assert out["intra_city_raion"][0] is None


def test_empty_cells_frame_returns_empty_with_schema() -> None:
    out = compute_cell_territory_features(_cells_frame([]), _objects_frame())
    assert out.is_empty()
    for col in (*TERRITORY_PROPAGATED_COLUMNS, "intra_city_raion"):
        assert col in out.columns
