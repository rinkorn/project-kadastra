from typing import Protocol

from catboost import CatBoostRegressor


class ModelLoaderPort(Protocol):
    def load(self, run_id: str) -> CatBoostRegressor: ...

    def find_latest_run_id(self, run_name_prefix: str) -> str: ...
