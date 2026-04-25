import io
from collections.abc import Mapping
from typing import Any

import polars as pl
from catboost import CatBoostRegressor

from kadastra.domain.asset_class import AssetClass
from kadastra.ml.train import CatBoostParams
from kadastra.usecases.train_object_valuation_model import TrainObjectValuationModel

KAZAN_LAT, KAZAN_LON = 55.7887, 49.1221


class _FakeReader:
    def __init__(self, by_class: dict[AssetClass, pl.DataFrame]) -> None:
        self._by_class = by_class

    def load(self, region_code: str, asset_class: AssetClass) -> pl.DataFrame:
        return self._by_class[asset_class]


class _FakeRegistry:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def log_run(
        self,
        *,
        run_name: str,
        params: Mapping[str, Any],
        metrics: Mapping[str, float],
        model: CatBoostRegressor,
        artifacts: Mapping[str, bytes] | None = None,
    ) -> str:
        self.calls.append(
            {
                "run_name": run_name,
                "params": dict(params),
                "metrics": dict(metrics),
                "model": model,
                "artifacts": dict(artifacts) if artifacts else {},
            }
        )
        return f"run_{len(self.calls)}"


def _featured(ac: AssetClass, n: int) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "object_id": f"way/{ac.value}-{i}",
                "asset_class": ac.value,
                "lat": KAZAN_LAT + 0.005 * (i // 5),
                "lon": KAZAN_LON + 0.005 * (i % 5),
                "levels": (i % 9) + 1,
                "flats": ((i % 7) + 1) * 10,
                "dist_metro_m": 200.0 + 30.0 * i,
                "dist_entrance_m": 150.0 + 25.0 * i,
                "count_stations_1km": (i % 3),
                "count_entrances_500m": (i % 2),
                "count_apartments_500m": (i % 5),
                "count_houses_500m": (i % 4),
                "count_commercial_500m": (i % 3),
                "road_length_500m": 500.0 + 50.0 * (i % 8),
                "synthetic_target_rub_per_m2": 50_000.0 + 1_000.0 * i,
            }
            for i in range(n)
        ],
        schema={
            "object_id": pl.Utf8,
            "asset_class": pl.Utf8,
            "lat": pl.Float64,
            "lon": pl.Float64,
            "levels": pl.Int64,
            "flats": pl.Int64,
            "dist_metro_m": pl.Float64,
            "dist_entrance_m": pl.Float64,
            "count_stations_1km": pl.Int64,
            "count_entrances_500m": pl.Int64,
            "count_apartments_500m": pl.Int64,
            "count_houses_500m": pl.Int64,
            "count_commercial_500m": pl.Int64,
            "road_length_500m": pl.Float64,
            "synthetic_target_rub_per_m2": pl.Float64,
        },
    )


_FAST_PARAMS = CatBoostParams(iterations=20, learning_rate=0.3, depth=4, seed=42)


def _usecase(reader: _FakeReader, registry: _FakeRegistry) -> TrainObjectValuationModel:
    return TrainObjectValuationModel(
        reader=reader,
        model_registry=registry,
        params=_FAST_PARAMS,
        n_splits=3,
        parent_resolution=7,
    )


def test_execute_returns_registry_run_id() -> None:
    reader = _FakeReader({AssetClass.APARTMENT: _featured(AssetClass.APARTMENT, 60)})
    registry = _FakeRegistry()

    run_id = _usecase(reader, registry).execute("RU-KAZAN-AGG", AssetClass.APARTMENT)

    assert run_id == "run_1"
    assert len(registry.calls) == 1


def test_run_name_includes_asset_class() -> None:
    reader = _FakeReader({AssetClass.HOUSE: _featured(AssetClass.HOUSE, 60)})
    registry = _FakeRegistry()

    _usecase(reader, registry).execute("RU-KAZAN-AGG", AssetClass.HOUSE)

    assert "house" in registry.calls[0]["run_name"]


def test_logs_metrics_and_feature_columns_in_params() -> None:
    reader = _FakeReader(
        {AssetClass.COMMERCIAL: _featured(AssetClass.COMMERCIAL, 60)}
    )
    registry = _FakeRegistry()

    _usecase(reader, registry).execute("RU-KAZAN-AGG", AssetClass.COMMERCIAL)

    metrics = registry.calls[0]["metrics"]
    assert {"mean_mae", "mean_rmse", "mean_mape"}.issubset(metrics.keys())

    params = registry.calls[0]["params"]
    feature_cols = set(params["feature_columns"])
    # Identifiers and target must not leak in
    for forbidden in (
        "object_id",
        "asset_class",
        "lat",
        "lon",
        "synthetic_target_rub_per_m2",
    ):
        assert forbidden not in feature_cols
    # Real features should be there
    for required in (
        "levels",
        "flats",
        "dist_metro_m",
        "count_apartments_500m",
        "road_length_500m",
    ):
        assert required in feature_cols


def test_drops_rows_with_null_target() -> None:
    df = _featured(AssetClass.APARTMENT, 60)
    df = df.with_columns(
        pl.when(pl.col("object_id") == "way/apartment-0")
        .then(None)
        .otherwise(pl.col("synthetic_target_rub_per_m2"))
        .alias("synthetic_target_rub_per_m2")
    )
    reader = _FakeReader({AssetClass.APARTMENT: df})
    registry = _FakeRegistry()

    _usecase(reader, registry).execute("RU-KAZAN-AGG", AssetClass.APARTMENT)

    n_samples = registry.calls[0]["params"]["n_samples"]
    assert n_samples == 59


def test_logs_asset_class_in_params() -> None:
    reader = _FakeReader({AssetClass.HOUSE: _featured(AssetClass.HOUSE, 60)})
    registry = _FakeRegistry()

    _usecase(reader, registry).execute("RU-KAZAN-AGG", AssetClass.HOUSE)

    assert registry.calls[0]["params"]["asset_class"] == "house"


def test_excludes_cost_value_rub_as_target_leak() -> None:
    """cost_value_rub is the EГРН total from which cost_index = cost_value / area
    is derived; passing it as a feature would let the model trivially recover the
    target. Must be excluded even though it's numeric.
    """
    df = _featured(AssetClass.APARTMENT, 60).with_columns(
        pl.lit(5_000_000.0).alias("cost_value_rub")
    )
    reader = _FakeReader({AssetClass.APARTMENT: df})
    registry = _FakeRegistry()

    _usecase(reader, registry).execute("RU-KAZAN-AGG", AssetClass.APARTMENT)

    feature_cols = set(registry.calls[0]["params"]["feature_columns"])
    assert "cost_value_rub" not in feature_cols


def test_logs_oof_predictions_artifact() -> None:
    """``cross_validate`` collects out-of-fold predictions; the use case
    must serialize them as parquet bytes under the
    ``oof_predictions.parquet`` artifact name. Inspector / map read
    this to compare y_true vs y_pred_oof per object."""
    reader = _FakeReader({AssetClass.APARTMENT: _featured(AssetClass.APARTMENT, 60)})
    registry = _FakeRegistry()

    _usecase(reader, registry).execute("RU-KAZAN-AGG", AssetClass.APARTMENT)

    artifacts = registry.calls[0]["artifacts"]
    assert "oof_predictions.parquet" in artifacts
    blob = artifacts["oof_predictions.parquet"]
    assert isinstance(blob, bytes) and len(blob) > 0

    df = pl.read_parquet(io.BytesIO(blob))
    assert set(df.columns) == {
        "object_id",
        "lat",
        "lon",
        "fold_id",
        "y_true",
        "y_pred_oof",
    }
    # Every input row appears once in OOF (each row is in exactly one
    # validation fold under spatial_kfold_split).
    assert df.height == 60
    # Sorted by object_id for stable lookups.
    obj_ids = df["object_id"].to_list()
    assert obj_ids == sorted(obj_ids)
    # All y_pred_oof are finite floats; fold_ids are 0..n_splits-1.
    assert df["y_pred_oof"].is_not_null().all()
    fold_ids = set(df["fold_id"].to_list())
    assert fold_ids == {0, 1, 2}  # n_splits=3 in _usecase fixture


def test_passes_string_columns_as_categorical_features() -> None:
    """NSPD valuation objects carry string fields like materials =
    "Кирпичные"/"Панельные"/"Монолитные" that strongly correlate with
    price per m². CatBoost handles strings natively when their column
    indices are passed via cat_features. The training pipeline must
    wire them in instead of dropping them.
    """
    df = _featured(AssetClass.HOUSE, 60).with_columns(
        pl.Series(
            "materials",
            ["Кирпичные", "Панельные", "Монолитные"] * 20,
            dtype=pl.Utf8,
        )
    )
    reader = _FakeReader({AssetClass.HOUSE: df})
    registry = _FakeRegistry()

    _usecase(reader, registry).execute("RU-KAZAN-AGG", AssetClass.HOUSE)

    params = registry.calls[0]["params"]
    feature_cols: list[str] = params["feature_columns"]
    assert "materials" in feature_cols
    cat_indices = params["cat_feature_indices"]
    assert cat_indices == [feature_cols.index("materials")]
