from typing import Protocol

import polars as pl


class FeatureReaderPort(Protocol):
    def load(self, region_code: str, resolution: int, feature_set: str) -> pl.DataFrame: ...
