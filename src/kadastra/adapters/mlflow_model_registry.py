from collections.abc import Mapping
from typing import Any

from catboost import CatBoostRegressor


class MLflowModelRegistry:
    def __init__(self, *, tracking_uri: str, experiment_name: str) -> None:
        raise NotImplementedError

    def log_run(
        self,
        *,
        run_name: str,
        params: Mapping[str, Any],
        metrics: Mapping[str, float],
        model: CatBoostRegressor,
    ) -> str:
        raise NotImplementedError
