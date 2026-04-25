from typing import Protocol

import polars as pl


class NspdSilverStorePort(Protocol):
    def save(self, region_code: str, source: str, df: pl.DataFrame) -> None: ...
    def load(self, region_code: str, source: str) -> pl.DataFrame: ...
