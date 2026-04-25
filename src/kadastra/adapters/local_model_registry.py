from collections.abc import Mapping
from pathlib import Path
from typing import Any

from catboost import CatBoostRegressor


class LocalModelRegistry:
    def __init__(self, base_path: Path) -> None:
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
