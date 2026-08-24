"""Tests for the GetCellExplanation use case (ADR-0029).

The stored cell_valuation rows keep only the top-15 locational terms;
this use case recomputes the FULL decomposition for one cell on demand.
"""

from __future__ import annotations

from typing import Any

import h3
import numpy as np
import polars as pl

from kadastra.domain.asset_class import AssetClass
from kadastra.usecases.get_cell_explanation import GetCellExplanation

_LAT, _LON = 55.79, 49.11
_CELL_A = h3.latlng_to_cell(_LAT, _LON, 10)
_CELL_B = h3.latlng_to_cell(55.2, 49.8, 10)  # far away, no objects


class FakeCellFeatureReader:
    def __init__(self, sets: dict[str, pl.DataFrame]) -> None:
        self._sets = sets

    def load(self, region_code: str, resolution: int, feature_set: str) -> pl.DataFrame:
        return self._sets[feature_set]


class FakeObjectReader:
    def __init__(self, frames: dict[AssetClass, pl.DataFrame]) -> None:
        self._frames = frames

    def load(self, region_code: str, asset_class: AssetClass) -> pl.DataFrame:
        return self._frames[asset_class]


class FakeEbm:
    """Hand-made additive model: intercept − dist + 10·area + (5_000 if brick).

    One locational term (dist_to_cbd_m) and two object terms (area_m2,
    materials) — the decomposition must keep only the first.
    """

    def __init__(self, feature_names: list[str]) -> None:
        self._idx = {name: feature_names.index(name) for name in ("dist_to_cbd_m", "area_m2", "materials")}

    def predict(self, X: Any) -> np.ndarray:
        return self.intercept() + self.eval_terms(X).sum(axis=1)

    def eval_terms(self, X: Any) -> np.ndarray:
        arr = np.asarray(X, dtype=object)
        dist = arr[:, self._idx["dist_to_cbd_m"]].astype(np.float64)
        area = arr[:, self._idx["area_m2"]].astype(np.float64)
        mat = arr[:, self._idx["materials"]]
        mat_term = np.where(mat == "brick", 5_000.0, 0.0)
        return np.column_stack([-dist, 10.0 * area, mat_term])

    def intercept(self) -> float:
        return 100_000.0

    def term_feature_indices(self) -> list[tuple[int, ...]]:
        return [(self._idx["dist_to_cbd_m"],), (self._idx["area_m2"],), (self._idx["materials"],)]


class FakeEbmLoader:
    def __init__(self, model: FakeEbm) -> None:
        self._model = model

    def load_latest(self, asset_class: AssetClass) -> FakeEbm:
        return self._model


def _gold() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "object_id": ["o1", "o2", "o3"],
            "asset_class": ["house"] * 3,
            "lat": [_LAT, _LAT + 0.0002, _LAT + 0.0004],
            "lon": [_LON, _LON, _LON],
            "dist_to_cbd_m": [1_000.0, 1_100.0, 900.0],
            "area_m2": [60.0, 80.0, 100.0],
            "materials": ["brick", "brick", "panel"],
            "synthetic_target_rub_per_m2": [90_000.0, 95_000.0, 85_000.0],
        }
    )


def _cell_sets() -> dict[str, pl.DataFrame]:
    return {
        "enrichment": pl.DataFrame(
            {
                "h3_index": [_CELL_A, _CELL_B],
                "dist_to_cbd_m": [1_000.0, 25_000.0],
            }
        ),
    }


_FEATURE_NAMES = ["dist_to_cbd_m", "area_m2", "materials"]


def _usecase(gold: pl.DataFrame | None = None) -> GetCellExplanation:
    return GetCellExplanation(
        cell_feature_reader=FakeCellFeatureReader(_cell_sets()),
        object_reader=FakeObjectReader({AssetClass.HOUSE: gold if gold is not None else _gold()}),
        ebm_loader=FakeEbmLoader(FakeEbm(_FEATURE_NAMES)),
        asset_classes=[AssetClass.HOUSE],
        cell_feature_sets=("enrichment",),
        resolution=10,
        relative_parent_resolutions=[7],
        relative_feature_columns=[],
        current_year=2026,
    )


def test_explain_returns_all_locational_terms_only() -> None:
    out = _usecase().explain("RU-KAZAN-AGG", AssetClass.HOUSE, _CELL_A)
    assert out is not None
    # Object terms (area_m2, materials) are excluded; the single
    # locational term is the cell's dist_to_cbd_m contribution.
    assert out["terms"] == [{"feature": "dist_to_cbd_m", "contribution": -1_000.0}]
    assert out["intercept"] == 100_000.0
    assert out["location_score"] == 99_000.0


def test_explain_sum_reconciles_exactly() -> None:
    out = _usecase().explain("RU-KAZAN-AGG", AssetClass.HOUSE, _CELL_B)
    assert out is not None
    total = out["intercept"] + sum(t["contribution"] for t in out["terms"])
    assert abs(total - out["location_score"]) < 0.5  # contributions rounded to 0.1


def test_explain_unknown_cell_returns_none() -> None:
    unknown = h3.latlng_to_cell(60.0, 50.0, 10)
    assert _usecase().explain("RU-KAZAN-AGG", AssetClass.HOUSE, unknown) is None


def test_explain_empty_class_gold_returns_none() -> None:
    empty = _gold().head(0)
    assert _usecase(gold=empty).explain("RU-KAZAN-AGG", AssetClass.HOUSE, _CELL_A) is None
