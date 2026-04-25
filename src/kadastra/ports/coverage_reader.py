from typing import Protocol

import polars as pl


class CoverageReaderPort(Protocol):
    def load(self, region_code: str, resolution: int) -> pl.DataFrame: ...
