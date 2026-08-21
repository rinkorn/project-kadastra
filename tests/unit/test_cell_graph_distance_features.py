"""Unit tests for ADR-0027 — walking-distance ЦОФ measured at cell centres."""

from __future__ import annotations

import h3
import numpy as np
import polars as pl

from kadastra.etl.cell_graph_distance_features import (
    compute_cell_graph_distance_features,
)

_KAZAN_LAT = 55.7905
_KAZAN_LON = 49.1142


class _FakeRoadGraph:
    def distance_matrix_m(
        self,
        from_coords: list[tuple[float, float]],
        to_coords: list[tuple[float, float]],
    ) -> np.ndarray:
        return np.full((len(from_coords), len(to_coords)), 300.0, dtype=np.float64)

    def nearest_distance_m(
        self,
        from_coords: list[tuple[float, float]],
        to_coords: list[tuple[float, float]],
    ) -> np.ndarray:
        return self.distance_matrix_m(from_coords, to_coords).min(axis=1)

    # ADR-0024 topology accessors — no graph behind this fake.
    def snap_node(self, coord: tuple[float, float]) -> tuple[int, float]:
        raise ValueError("fake has no nodes")

    def node_coord(self, node_id: int) -> tuple[float, float]:
        raise ValueError("fake has no nodes")

    def reachable_nodes_within_m(
        self,
        from_coord: tuple[float, float],
        cutoff_m: float,
    ) -> dict[int, float]:
        return {}


def _cell_frame(cell: str) -> pl.DataFrame:
    return pl.DataFrame({"h3_index": [cell]})


def _points() -> pl.DataFrame:
    return pl.DataFrame({"lat": [_KAZAN_LAT], "lon": [_KAZAN_LON]})


def test_computes_walk_distance_per_layer() -> None:
    cell = h3.latlng_to_cell(_KAZAN_LAT, _KAZAN_LON, 10)

    df = compute_cell_graph_distance_features(
        _cell_frame(cell),
        point_layers={"school": _points(), "clinic": _points()},
        road_graph=_FakeRoadGraph(),
    )

    assert df.columns == ["h3_index", "walk_dist_to_school_m", "walk_dist_to_clinic_m"]
    assert float(df["walk_dist_to_school_m"][0]) == 300.0
    assert float(df["walk_dist_to_clinic_m"][0]) == 300.0


def test_empty_layer_yields_null_column() -> None:
    cell = h3.latlng_to_cell(_KAZAN_LAT, _KAZAN_LON, 10)
    empty = pl.DataFrame({"lat": [], "lon": []}, schema={"lat": pl.Float64, "lon": pl.Float64})

    df = compute_cell_graph_distance_features(
        _cell_frame(cell),
        point_layers={"school": empty},
        road_graph=_FakeRoadGraph(),
    )

    assert "walk_dist_to_school_m" in df.columns
    assert df["walk_dist_to_school_m"][0] is None


def test_unreachable_points_yield_null() -> None:
    """ADR-0030: unreachable (no graph path) → null, same semantics as an
    empty layer; the old 1e9 sentinel was read by models as 'very far'."""
    cell = h3.latlng_to_cell(_KAZAN_LAT, _KAZAN_LON, 10)

    class _NoPathRoadGraph:
        def distance_matrix_m(
            self,
            from_coords: list[tuple[float, float]],
            to_coords: list[tuple[float, float]],
        ) -> np.ndarray:
            return np.full((len(from_coords), len(to_coords)), np.inf, dtype=np.float64)

        def nearest_distance_m(
            self,
            from_coords: list[tuple[float, float]],
            to_coords: list[tuple[float, float]],
        ) -> np.ndarray:
            return self.distance_matrix_m(from_coords, to_coords).min(axis=1)

        # ADR-0024 topology accessors — no graph behind this fake.
        def snap_node(self, coord: tuple[float, float]) -> tuple[int, float]:
            raise ValueError("fake has no nodes")

        def node_coord(self, node_id: int) -> tuple[float, float]:
            raise ValueError("fake has no nodes")

        def reachable_nodes_within_m(
            self,
            from_coord: tuple[float, float],
            cutoff_m: float,
        ) -> dict[int, float]:
            return {}

    df = compute_cell_graph_distance_features(
        _cell_frame(cell),
        point_layers={"school": _points()},
        road_graph=_NoPathRoadGraph(),
    )

    assert df["walk_dist_to_school_m"][0] is None


def test_empty_cells_emits_null_columns() -> None:
    empty = pl.DataFrame({"h3_index": []}, schema={"h3_index": pl.Utf8})

    df = compute_cell_graph_distance_features(
        empty,
        point_layers={"school": _points()},
        road_graph=_FakeRoadGraph(),
    )

    assert df.height == 0
    assert {"walk_dist_to_school_m"} <= set(df.columns)


def test_no_layers_returns_unchanged_frame() -> None:
    cell = h3.latlng_to_cell(_KAZAN_LAT, _KAZAN_LON, 10)
    frame = _cell_frame(cell)

    df = compute_cell_graph_distance_features(frame, point_layers={}, road_graph=_FakeRoadGraph())

    assert df.columns == frame.columns
