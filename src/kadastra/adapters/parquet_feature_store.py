from pathlib import Path

import polars as pl


class ParquetFeatureStore:
    def __init__(self, base_path: Path) -> None:
        self._base_path = base_path

    def save(self, region_code: str, resolution: int, feature_set: str, df: pl.DataFrame) -> None:
        raise NotImplementedError
