from typing import Protocol

import polars as pl


class GoldFeatureStorePort(Protocol):
    def save(self, region_code: str, resolution: int, df: pl.DataFrame) -> None: ...
