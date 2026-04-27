from typing import Protocol

import polars as pl

from kadastra.domain.asset_class import AssetClass


class ValuationObjectReaderPort(Protocol):
    def load(self, region_code: str, asset_class: AssetClass) -> pl.DataFrame: ...
