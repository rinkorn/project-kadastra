from pathlib import Path

import mlflow
import numpy as np
import pytest
from catboost import CatBoostRegressor

from kadastra.adapters.mlflow_model_registry import MLflowModelRegistry
from kadastra.ml.train import CatBoostParams, train_catboost


@pytest.fixture
def trained_model() -> CatBoostRegressor:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 2))
    y = X[:, 0] + 2 * X[:, 1]
    return train_catboost(X, y, CatBoostParams(iterations=10, learning_rate=0.3, depth=3, seed=0))


def test_log_run_returns_mlflow_run_id(tmp_path: Path, trained_model: CatBoostRegressor) -> None:
    registry = MLflowModelRegistry(
        tracking_uri=f"file:{tmp_path / 'mlruns'}", experiment_name="test-exp"
    )

    run_id = registry.log_run(
        run_name="baseline",
        params={"iterations": 10},
        metrics={"mae": 1.2, "rmse": 1.5},
        model=trained_model,
    )

    # MLflow run IDs are 32-char hex strings
    assert len(run_id) == 32
    assert all(c in "0123456789abcdef" for c in run_id)


def test_log_run_records_params_and_metrics_in_mlflow(tmp_path: Path, trained_model: CatBoostRegressor) -> None:
    registry = MLflowModelRegistry(
        tracking_uri=f"file:{tmp_path / 'mlruns'}", experiment_name="test-exp"
    )

    run_id = registry.log_run(
        run_name="baseline",
        params={"iterations": 10, "depth": 3},
        metrics={"mae": 1.2, "rmse": 1.5, "mape": 0.05},
        model=trained_model,
    )

    client = mlflow.MlflowClient(tracking_uri=f"file:{tmp_path / 'mlruns'}")
    run = client.get_run(run_id)
    assert run.data.params["iterations"] == "10"
    assert run.data.params["depth"] == "3"
    assert run.data.metrics["mae"] == 1.2
    assert run.data.metrics["rmse"] == 1.5
    assert run.data.metrics["mape"] == 0.05


def test_log_run_creates_experiment_if_missing(tmp_path: Path, trained_model: CatBoostRegressor) -> None:
    registry = MLflowModelRegistry(
        tracking_uri=f"file:{tmp_path / 'mlruns'}", experiment_name="brand-new-exp"
    )

    registry.log_run(
        run_name="baseline", params={}, metrics={"mae": 1.0}, model=trained_model
    )

    client = mlflow.MlflowClient(tracking_uri=f"file:{tmp_path / 'mlruns'}")
    exp = client.get_experiment_by_name("brand-new-exp")
    assert exp is not None
