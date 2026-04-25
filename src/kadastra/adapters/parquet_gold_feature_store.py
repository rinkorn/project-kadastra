from pathlib import Path

import polars as pl


class ParquetGoldFeatureStore:
    def __init__(self, base_path: Path) -> None:
        self._base_path = base_path

    def save(self, region_code: str, resolution: int, df: pl.DataFrame) -> None:
        partition = self._base_path / f"region={region_code}" / f"resolution={resolution}"
        partition.mkdir(parents=True, exist_ok=True)
        df.write_parquet(partition / "data.parquet")

    def load(self, region_code: str, resolution: int) -> pl.DataFrame:
        raise NotImplementedError
