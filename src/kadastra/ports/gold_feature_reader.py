from typing import Protocol

import polars as pl


class GoldFeatureReaderPort(Protocol):
    def load(self, region_code: str, resolution: int) -> pl.DataFrame: ...
