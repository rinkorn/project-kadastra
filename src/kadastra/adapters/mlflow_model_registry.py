from collections.abc import Mapping
from typing import Any

import mlflow
from catboost import CatBoostRegressor
from mlflow import catboost as mlflow_catboost


class MLflowModelRegistry:
    def __init__(self, *, tracking_uri: str, experiment_name: str) -> None:
        self._client = mlflow.MlflowClient(tracking_uri=tracking_uri)
        self._tracking_uri = tracking_uri
        self._experiment_name = experiment_name

    def log_run(
        self,
        *,
        run_name: str,
        params: Mapping[str, Any],
        metrics: Mapping[str, float],
        model: CatBoostRegressor,
    ) -> str:
        mlflow.set_tracking_uri(self._tracking_uri)
        mlflow.set_experiment(self._experiment_name)
        with mlflow.start_run(run_name=run_name) as run:
            mlflow.log_params(dict(params))
            mlflow.log_metrics(dict(metrics))
            mlflow_catboost.log_model(model, name="model")
            return run.info.run_id
