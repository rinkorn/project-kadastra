import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from catboost import CatBoostRegressor


class LocalModelRegistry:
    def __init__(self, base_path: Path) -> None:
        self._base_path = base_path

    def log_run(
        self,
        *,
        run_name: str,
        params: Mapping[str, Any],
        metrics: Mapping[str, float],
        model: CatBoostRegressor,
    ) -> str:
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        run_id = f"{run_name}_{timestamp}"
        run_dir = self._base_path / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        (run_dir / "params.json").write_text(json.dumps(dict(params)))
        (run_dir / "metrics.json").write_text(json.dumps(dict(metrics)))
        model.save_model(str(run_dir / "model.cbm"))

        return run_id
