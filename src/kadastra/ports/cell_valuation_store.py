"""Port for the cell valuation store (ADR-0029 gold layer)."""

from typing import Protocol

import polars as pl

from kadastra.domain.asset_class import AssetClass


class CellValuationStorePort(Protocol):
    def save(self, region_code: str, asset_class: AssetClass, df: pl.DataFrame) -> None: ...
