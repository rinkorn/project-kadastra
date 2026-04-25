from pathlib import Path

import polars as pl

from kadastra.domain.asset_class import AssetClass


class ParquetValuationObjectStore:
    def __init__(self, base_path: Path) -> None:
        self._base_path = base_path

    def _partition_dir(self, region_code: str, asset_class: AssetClass) -> Path:
        return (
            self._base_path
            / f"region={region_code}"
            / f"asset_class={asset_class.value}"
        )

    def save(
        self, region_code: str, asset_class: AssetClass, df: pl.DataFrame
    ) -> None:
        partition = self._partition_dir(region_code, asset_class)
        partition.mkdir(parents=True, exist_ok=True)
        df.write_parquet(partition / "data.parquet")

    def load(
        self, region_code: str, asset_class: AssetClass
    ) -> pl.DataFrame:
        path = self._partition_dir(region_code, asset_class) / "data.parquet"
        if not path.is_file():
            raise FileNotFoundError(path)
        return pl.read_parquet(path)
