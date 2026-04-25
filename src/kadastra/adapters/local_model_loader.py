from pathlib import Path

from catboost import CatBoostRegressor


class LocalModelLoader:
    def __init__(self, base_path: Path) -> None:
        self._base_path = base_path

    def load(self, run_id: str) -> CatBoostRegressor:
        model_path = self._base_path / run_id / "model.cbm"
        if not model_path.is_file():
            raise FileNotFoundError(f"model artifact not found: {model_path}")
        model = CatBoostRegressor()
        model.load_model(str(model_path))
        return model

    def find_latest_run_id(self, run_name_prefix: str) -> str:
        if not self._base_path.is_dir():
            raise FileNotFoundError(f"registry path does not exist: {self._base_path}")
        matches = sorted(
            d.name for d in self._base_path.iterdir() if d.is_dir() and d.name.startswith(run_name_prefix)
        )
        if not matches:
            raise FileNotFoundError(
                f"no runs matching prefix={run_name_prefix!r} under {self._base_path}"
            )
        return matches[-1]
