from pathlib import Path

from catboost import CatBoostRegressor


class LocalModelLoader:
    def __init__(self, base_path: Path) -> None:
        raise NotImplementedError

    def load(self, run_id: str) -> CatBoostRegressor:
        raise NotImplementedError

    def find_latest_run_id(self, run_name_prefix: str) -> str:
        raise NotImplementedError
