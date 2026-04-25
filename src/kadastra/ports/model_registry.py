from collections.abc import Mapping
from typing import Any, Protocol

from catboost import CatBoostRegressor


class ModelRegistryPort(Protocol):
    def log_run(
        self,
        *,
        run_name: str,
        params: Mapping[str, Any],
        metrics: Mapping[str, float],
        model: CatBoostRegressor,
    ) -> str: ...
