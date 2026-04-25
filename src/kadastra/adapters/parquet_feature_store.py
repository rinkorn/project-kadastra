from pathlib import Path

import polars as pl


class ParquetFeatureStore:
    def __init__(self, base_path: Path) -> None:
        self._base_path = base_path

    def save(self, region_code: str, resolution: int, feature_set: str, df: pl.DataFrame) -> None:
        partition = (
            self._base_path
            / f"region={region_code}"
            / f"feature_set={feature_set}"
            / f"resolution={resolution}"
        )
        partition.mkdir(parents=True, exist_ok=True)
        df.write_parquet(partition / "data.parquet")

    def load(self, region_code: str, resolution: int, feature_set: str) -> pl.DataFrame:
        path = (
            self._base_path
            / f"region={region_code}"
            / f"feature_set={feature_set}"
            / f"resolution={resolution}"
            / "data.parquet"
        )
        return pl.read_parquet(path)
