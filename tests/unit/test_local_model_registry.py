import json
from pathlib import Path

import numpy as np
import pytest
from catboost import CatBoostRegressor

from kadastra.adapters.local_model_registry import LocalModelRegistry
from kadastra.ml.train import CatBoostParams, train_catboost


@pytest.fixture
def trained_model() -> CatBoostRegressor:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 2))
    y = X[:, 0] + 2 * X[:, 1]
    return train_catboost(X, y, CatBoostParams(iterations=10, learning_rate=0.3, depth=3, seed=0))


def test_log_run_creates_run_directory(tmp_path: Path, trained_model: CatBoostRegressor) -> None:
    registry = LocalModelRegistry(tmp_path / "models")

    run_id = registry.log_run(
        run_name="baseline",
        params={"iterations": 10, "depth": 3},
        metrics={"mae": 1.2, "rmse": 1.5},
        model=trained_model,
    )

    run_dir = tmp_path / "models" / run_id
    assert run_dir.is_dir()


def test_log_run_writes_params_metrics_and_model_files(tmp_path: Path, trained_model: CatBoostRegressor) -> None:
    registry = LocalModelRegistry(tmp_path / "models")

    run_id = registry.log_run(
        run_name="baseline",
        params={"iterations": 10, "depth": 3},
        metrics={"mae": 1.2, "rmse": 1.5},
        model=trained_model,
    )

    run_dir = tmp_path / "models" / run_id
    assert (run_dir / "params.json").is_file()
    assert (run_dir / "metrics.json").is_file()
    assert (run_dir / "model.cbm").is_file()

    assert json.loads((run_dir / "params.json").read_text()) == {"iterations": 10, "depth": 3}
    assert json.loads((run_dir / "metrics.json").read_text()) == {"mae": 1.2, "rmse": 1.5}


def test_log_run_returns_unique_id_for_consecutive_calls(tmp_path: Path, trained_model: CatBoostRegressor) -> None:
    registry = LocalModelRegistry(tmp_path / "models")

    run_id_1 = registry.log_run(
        run_name="baseline", params={}, metrics={}, model=trained_model
    )
    run_id_2 = registry.log_run(
        run_name="baseline", params={}, metrics={}, model=trained_model
    )

    assert run_id_1 != run_id_2


def test_log_run_id_starts_with_run_name(tmp_path: Path, trained_model: CatBoostRegressor) -> None:
    registry = LocalModelRegistry(tmp_path / "models")

    run_id = registry.log_run(
        run_name="catboost-v1", params={}, metrics={}, model=trained_model
    )

    assert run_id.startswith("catboost-v1")
