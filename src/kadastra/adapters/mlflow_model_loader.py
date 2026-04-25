from catboost import CatBoostRegressor


class MLflowModelLoader:
    def __init__(self, *, tracking_uri: str, experiment_name: str) -> None:
        raise NotImplementedError

    def load(self, run_id: str) -> CatBoostRegressor:
        raise NotImplementedError

    def find_latest_run_id(self, run_name_prefix: str) -> str:
        raise NotImplementedError
