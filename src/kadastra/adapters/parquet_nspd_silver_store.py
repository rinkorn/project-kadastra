from pathlib import Path

import polars as pl


class ParquetNspdSilverStore:
    def __init__(self, base_path: Path) -> None:
        self._base_path = base_path

    def _partition_dir(self, region_code: str, source: str) -> Path:
        return self._base_path / f"region={region_code}" / f"source={source}"

    def save(self, region_code: str, source: str, df: pl.DataFrame) -> None:
        raise NotImplementedError

    def load(self, region_code: str, source: str) -> pl.DataFrame:
        raise NotImplementedError
