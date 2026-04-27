import polars as pl

from kadastra.domain.asset_class import AssetClass
from kadastra.etl.object_synthetic_target import compute_object_synthetic_target
from kadastra.ports.valuation_object_reader import ValuationObjectReaderPort
from kadastra.ports.valuation_object_store import ValuationObjectStorePort


class BuildObjectSyntheticTarget:
    def __init__(
        self,
        reader: ValuationObjectReaderPort,
        store: ValuationObjectStorePort,
        seed: int,
    ) -> None:
        self._reader = reader
        self._store = store
        self._seed = seed

    def execute(self, region_code: str, asset_classes: list[AssetClass]) -> None:
        for asset_class in asset_classes:
            df = self._reader.load(region_code, asset_class)
            if df.is_empty():
                with_target = df.with_columns(pl.lit(None, dtype=pl.Float64).alias("synthetic_target_rub_per_m2"))
            else:
                with_target = compute_object_synthetic_target(df, seed=self._seed)
            self._store.save(region_code, asset_class, with_target)
