"""Tests for the BuildCellValuation use case (ADR-0029)."""

from __future__ import annotations

import json
from typing import Any

import h3
import numpy as np
import polars as pl
import pytest

from kadastra.domain.asset_class import AssetClass
from kadastra.etl.relative_features import compute_relative_features
from kadastra.usecases.build_cell_valuation import BuildCellValuation

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

    Columns are located by name so the matrix layout (numeric-then-
    categorical, gold column order) is handled explicitly.
    """

    def __init__(self, feature_names: list[str]) -> None:
        self._idx = {name: feature_names.index(name) for name in ("dist_to_cbd_m", "area_m2", "materials")}

    def _term_columns(self, X: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        arr = np.asarray(X, dtype=object)
        dist = arr[:, self._idx["dist_to_cbd_m"]].astype(np.float64)
        area = arr[:, self._idx["area_m2"]].astype(np.float64)
        mat = arr[:, self._idx["materials"]]
        return dist, area, mat

    def predict(self, X: Any) -> np.ndarray:
        return self.intercept() + self.eval_terms(X).sum(axis=1)

    def eval_terms(self, X: Any) -> np.ndarray:
        dist, area, mat = self._term_columns(X)
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


class FakeStore:
    def __init__(self) -> None:
        self.saved: dict[AssetClass, pl.DataFrame] = {}

    def save(self, region_code: str, asset_class: AssetClass, df: pl.DataFrame) -> None:
        self.saved[asset_class] = df


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


def _usecase(store: FakeStore, *, model: FakeEbm | None = None, gold: pl.DataFrame | None = None) -> BuildCellValuation:
    return BuildCellValuation(
        cell_feature_reader=FakeCellFeatureReader(_cell_sets()),
        object_reader=FakeObjectReader({AssetClass.HOUSE: gold if gold is not None else _gold()}),
        ebm_loader=FakeEbmLoader(model if model is not None else FakeEbm(_FEATURE_NAMES)),
        output_store=store,
        cell_feature_sets=("enrichment",),
        resolution=10,
        relative_parent_resolutions=[7],
        relative_feature_columns=[],
        current_year=2026,
    )


def test_output_schema_and_rows() -> None:
    store = FakeStore()
    results = _usecase(store).execute("RU-KAZAN-AGG", [AssetClass.HOUSE])
    assert AssetClass.HOUSE in results

    out = store.saved[AssetClass.HOUSE]
    assert out.columns == [
        "h3_index",
        "lat",
        "lon",
        "cell_water_share",
        "on_water",
        "reference_variant",
        "reference_rub_per_m2",
        "location_score_rub_per_m2",
        "top_terms_json",
        "n_sample_objects",
        "sample_covered",
    ]
    # Water mask defaults to dry: no water layer wired in tests → all
    # zeros / False (the revert path for the masking decision).
    assert out["cell_water_share"].to_list() == [0.0, 0.0]
    assert out["on_water"].to_list() == [False, False]
    assert out.height == 2
    assert out["reference_variant"].to_list() == ["default", "default"]


def test_reference_price_uses_template_and_cell_location() -> None:
    store = FakeStore()
    _usecase(store).execute("RU-KAZAN-AGG", [AssetClass.HOUSE])
    out = store.saved[AssetClass.HOUSE]
    row_a = out.filter(pl.col("h3_index") == _CELL_A)
    # Template: median area 80, mode materials brick →
    # 100_000 − 1_000 + 10·80 + 5_000 = 104_800.
    assert row_a["reference_rub_per_m2"][0] == pytest.approx(104_800.0)
    row_b = out.filter(pl.col("h3_index") == _CELL_B)
    assert row_b["reference_rub_per_m2"][0] == pytest.approx(100_000 - 25_000 + 800 + 5_000)


def test_location_score_excludes_object_terms() -> None:
    store = FakeStore()
    _usecase(store).execute("RU-KAZAN-AGG", [AssetClass.HOUSE])
    out = store.saved[AssetClass.HOUSE]
    row_a = out.filter(pl.col("h3_index") == _CELL_A)
    # intercept + dist term only: 100_000 − 1_000.
    assert row_a["location_score_rub_per_m2"][0] == pytest.approx(99_000.0)
    row_b = out.filter(pl.col("h3_index") == _CELL_B)
    assert row_b["location_score_rub_per_m2"][0] == pytest.approx(75_000.0)


def test_top_terms_json_keeps_only_locational_terms() -> None:
    store = FakeStore()
    _usecase(store).execute("RU-KAZAN-AGG", [AssetClass.HOUSE])
    out = store.saved[AssetClass.HOUSE]
    row_a = out.filter(pl.col("h3_index") == _CELL_A)
    # Only dist_to_cbd_m is locational; area/materials are object terms.
    assert json.loads(row_a["top_terms_json"][0]) == [{"feature": "dist_to_cbd_m", "contribution": -1_000.0}]
    row_b = out.filter(pl.col("h3_index") == _CELL_B)
    assert json.loads(row_b["top_terms_json"][0]) == [{"feature": "dist_to_cbd_m", "contribution": -25_000.0}]


def test_sample_coverage_flags() -> None:
    store = FakeStore()
    _usecase(store).execute("RU-KAZAN-AGG", [AssetClass.HOUSE])
    out = store.saved[AssetClass.HOUSE]
    row_a = out.filter(pl.col("h3_index") == _CELL_A)
    assert row_a["n_sample_objects"][0] == 3
    assert row_a["sample_covered"][0] is True
    row_b = out.filter(pl.col("h3_index") == _CELL_B)
    assert row_b["n_sample_objects"][0] == 0
    assert row_b["sample_covered"][0] is False


def test_result_reports_cbd_correlation() -> None:
    store = FakeStore()
    results = _usecase(store).execute("RU-KAZAN-AGG", [AssetClass.HOUSE])
    res = results[AssetClass.HOUSE]
    # location_score = 100_000 − dist → perfect negative correlation.
    assert res.cbd_correlation == pytest.approx(-1.0)
    assert res.covered_share == pytest.approx(0.5)
    assert len(res.reference_objects) == 1
    assert res.reference_objects[0].attributes["area_m2"] == 80.0


def test_relative_features_computed_from_gold_aggregates() -> None:
    """Cell rel columns come from training-time gold aggregates.

    Gold carries rel columns (as the real gold does); the far cell B
    sits in a parent without objects → null rel → model receives NaN.
    """
    gold = compute_relative_features(_gold(), parent_resolutions=[7], feature_columns=["area_m2"])

    class RelAwareEbm(FakeEbm):
        def __init__(self, feature_names: list[str]) -> None:
            super().__init__(feature_names)
            self._rel_idx = feature_names.index("area_m2__rel_p7_diff_med")

        def eval_terms(self, X: Any) -> np.ndarray:
            arr = np.asarray(X, dtype=object)
            rel_diff = np.nan_to_num(arr[:, self._rel_idx].astype(np.float64))
            return np.column_stack([super().eval_terms(X), rel_diff])

        def term_feature_indices(self) -> list[tuple[int, ...]]:
            return [*super().term_feature_indices(), (self._rel_idx,)]

    numeric = [
        "dist_to_cbd_m",
        "area_m2",
        "count_p7",
        "area_m2__rel_p7_diff_med",
        "area_m2__rel_p7_ratio_med",
        "area_m2__rel_p7_z_iqr",
    ]
    model = RelAwareEbm([*numeric, "materials"])
    store = FakeStore()
    usecase = BuildCellValuation(
        cell_feature_reader=FakeCellFeatureReader(_cell_sets()),
        object_reader=FakeObjectReader({AssetClass.HOUSE: gold}),
        ebm_loader=FakeEbmLoader(model),
        output_store=store,
        cell_feature_sets=("enrichment",),
        resolution=10,
        relative_parent_resolutions=[7],
        relative_feature_columns=["area_m2"],
        current_year=2026,
    )
    usecase.execute("RU-KAZAN-AGG", [AssetClass.HOUSE])
    out = store.saved[AssetClass.HOUSE]
    row_a = out.filter(pl.col("h3_index") == _CELL_A)
    # Median gold area = 80; template area = 80 → rel diff = 0 → base price.
    assert row_a["reference_rub_per_m2"][0] == pytest.approx(104_800.0)
    row_b = out.filter(pl.col("h3_index") == _CELL_B)
    # No parent aggregate → rel term null → NaN → 0 contribution.
    assert row_b["reference_rub_per_m2"][0] == pytest.approx(100_000 - 25_000 + 800 + 5_000)
