from collections.abc import Mapping
from typing import Any

import h3
import polars as pl
from catboost import CatBoostRegressor

from kadastra.ml.train import CatBoostParams
from kadastra.usecases.train_valuation_model import TrainValuationModel

KAZAN_LAT, KAZAN_LON = 55.7887, 49.1221


class FakeGoldReader:
    def __init__(self, by_resolution: dict[int, pl.DataFrame]) -> None:
        self._by_resolution = by_resolution

    def load(self, region_code: str, resolution: int) -> pl.DataFrame:
        return self._by_resolution[resolution]


class FakeRegistry:
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
                "artifacts": dict(artifacts or {}),
            }
        )
        return f"run_{len(self.calls)}"


def _features_target_pair(n: int, resolution: int) -> tuple[pl.DataFrame, pl.DataFrame]:
    cells = [h3.latlng_to_cell(KAZAN_LAT + 0.01 * i, KAZAN_LON, resolution) for i in range(n)]
    features = pl.DataFrame(
        {
            "h3_index": cells,
            "resolution": [resolution] * n,
            "building_count": list(range(n)),
            "count_stations_1km": [i % 2 for i in range(n)],
        }
    )
    target = pl.DataFrame(
        {
            "h3_index": cells,
            "resolution": [resolution] * n,
            "synthetic_target_rub_per_m2": [1000.0 * i for i in range(n)],
        }
    )
    return features, target


_FAST_PARAMS = CatBoostParams(iterations=20, learning_rate=0.3, depth=4, seed=42)


def test_execute_returns_registry_run_id() -> None:
    features, target = _features_target_pair(60, 8)
    registry = FakeRegistry()
    usecase = TrainValuationModel(
        gold_reader=FakeGoldReader({8: features}),
        target_reader=FakeGoldReader({8: target}),
        model_registry=registry,
        params=_FAST_PARAMS,
        n_splits=3,
        parent_resolution=6,
    )

    run_id = usecase.execute("RU-TA", 8)

    assert run_id == "run_1"


def test_execute_logs_one_run_with_metrics_and_params() -> None:
    features, target = _features_target_pair(60, 8)
    registry = FakeRegistry()
    usecase = TrainValuationModel(
        gold_reader=FakeGoldReader({8: features}),
        target_reader=FakeGoldReader({8: target}),
        model_registry=registry,
        params=_FAST_PARAMS,
        n_splits=3,
        parent_resolution=6,
    )

    usecase.execute("RU-TA", 8)

    assert len(registry.calls) == 1
    call = registry.calls[0]
    assert "mean_mae" in call["metrics"]
    assert "mean_rmse" in call["metrics"]
    assert "mean_mape" in call["metrics"]
    assert call["params"]["iterations"] == 20
    assert call["params"]["depth"] == 4
    assert call["params"]["n_splits"] == 3
    assert call["params"]["parent_resolution"] == 6
    assert call["params"]["n_samples"] == 60


def test_execute_run_name_includes_resolution() -> None:
    features, target = _features_target_pair(60, 8)
    registry = FakeRegistry()
    usecase = TrainValuationModel(
        gold_reader=FakeGoldReader({8: features}),
        target_reader=FakeGoldReader({8: target}),
        model_registry=registry,
        params=_FAST_PARAMS,
        n_splits=3,
        parent_resolution=6,
    )

    usecase.execute("RU-TA", 8)

    assert "res8" in registry.calls[0]["run_name"]


def test_execute_passes_fitted_catboost_model_to_registry() -> None:
    features, target = _features_target_pair(60, 8)
    registry = FakeRegistry()
    usecase = TrainValuationModel(
        gold_reader=FakeGoldReader({8: features}),
        target_reader=FakeGoldReader({8: target}),
        model_registry=registry,
        params=_FAST_PARAMS,
        n_splits=3,
        parent_resolution=6,
    )

    usecase.execute("RU-TA", 8)

    model = registry.calls[0]["model"]
    assert isinstance(model, CatBoostRegressor)
    assert model.is_fitted()


def test_execute_auto_clips_parent_res_above_cell_res() -> None:
    """parent_resolution=10 with res=8 cells must not crash; should clip to res-1."""
    features, target = _features_target_pair(60, 8)
    registry = FakeRegistry()
    usecase = TrainValuationModel(
        gold_reader=FakeGoldReader({8: features}),
        target_reader=FakeGoldReader({8: target}),
        model_registry=registry,
        params=_FAST_PARAMS,
        n_splits=3,
        parent_resolution=10,
    )

    usecase.execute("RU-TA", 8)

    # 1 run successfully logged; clip prevented the crash spatial_kfold would raise
    assert len(registry.calls) == 1


def test_execute_drops_rows_with_null_target() -> None:
    cells = [h3.latlng_to_cell(KAZAN_LAT + 0.01 * i, KAZAN_LON, 8) for i in range(60)]
    features = pl.DataFrame(
        {
            "h3_index": cells,
            "resolution": [8] * 60,
            "building_count": list(range(60)),
            "count_stations_1km": [i % 2 for i in range(60)],
        }
    )
    # Target only covers first 50 rows
    target = pl.DataFrame(
        {
            "h3_index": cells[:50],
            "resolution": [8] * 50,
            "synthetic_target_rub_per_m2": [1000.0 * i for i in range(50)],
        }
    )
    registry = FakeRegistry()
    usecase = TrainValuationModel(
        gold_reader=FakeGoldReader({8: features}),
        target_reader=FakeGoldReader({8: target}),
        model_registry=registry,
        params=_FAST_PARAMS,
        n_splits=3,
        parent_resolution=6,
    )

    usecase.execute("RU-TA", 8)

    assert registry.calls[0]["params"]["n_samples"] == 50
