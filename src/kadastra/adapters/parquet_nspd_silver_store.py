from pathlib import Path

import polars as pl


class ParquetNspdSilverStore:
    def __init__(self, base_path: Path) -> None:
        self._base_path = base_path

    def _partition_dir(self, region_code: str, source: str) -> Path:
        return self._base_path / f"region={region_code}" / f"source={source}"

    def save(self, region_code: str, source: str, df: pl.DataFrame) -> None:
        partition = self._partition_dir(region_code, source)
        partition.mkdir(parents=True, exist_ok=True)
        df.write_parquet(partition / "data.parquet")

    def load(self, region_code: str, source: str) -> pl.DataFrame:
        path = self._partition_dir(region_code, source) / "data.parquet"
        if not path.is_file():
            raise FileNotFoundError(path)
        return pl.read_parquet(path)
