from typing import Protocol

import polars as pl


class FeatureStorePort(Protocol):
    def save(self, region_code: str, resolution: int, feature_set: str, df: pl.DataFrame) -> None: ...
