import mlflow
from catboost import CatBoostRegressor
from mlflow import catboost as mlflow_catboost


class MLflowModelLoader:
    def __init__(self, *, tracking_uri: str, experiment_name: str) -> None:
        self._tracking_uri = tracking_uri
        self._experiment_name = experiment_name
        self._client = mlflow.MlflowClient(tracking_uri=tracking_uri)

    def load(self, run_id: str) -> CatBoostRegressor:
        mlflow.set_tracking_uri(self._tracking_uri)
        model = mlflow_catboost.load_model(model_uri=f"runs:/{run_id}/model")
        if not isinstance(model, CatBoostRegressor):
            raise TypeError(
                f"expected CatBoostRegressor for run_id={run_id}, got {type(model).__name__}"
            )
        return model

    def find_latest_run_id(self, run_name_prefix: str) -> str:
        experiment = self._client.get_experiment_by_name(self._experiment_name)
        if experiment is None:
            raise FileNotFoundError(
                f"experiment {self._experiment_name!r} not found in {self._tracking_uri}"
            )
        runs = self._client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"tags.mlflow.runName LIKE '{run_name_prefix}%'",
            order_by=["start_time DESC"],
            max_results=1,
        )
        if not runs:
            raise FileNotFoundError(
                f"no runs matching prefix={run_name_prefix!r} in experiment {self._experiment_name!r}"
            )
        return runs[0].info.run_id
