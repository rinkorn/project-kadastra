from typing import Protocol

import polars as pl

from kadastra.domain.asset_class import AssetClass


class ValuationObjectStorePort(Protocol):
    def save(self, region_code: str, asset_class: AssetClass, df: pl.DataFrame) -> None: ...
