import io

import polars as pl

from kadastra.domain.asset_class import AssetClass
from kadastra.etl.valuation_objects import assemble_valuation_objects
from kadastra.ports.raw_data import RawDataPort
from kadastra.ports.valuation_object_store import ValuationObjectStorePort

_BUILDINGS_SCHEMA_OVERRIDES = {
    "osm_id": pl.Utf8,
    "osm_type": pl.Utf8,
    "levels": pl.Utf8,
    "flats": pl.Utf8,
    "housenumber": pl.Utf8,
    "postcode": pl.Utf8,
    "start_date": pl.Utf8,
}

_OUTPUT_SCHEMA = {
    "object_id": pl.Utf8,
    "asset_class": pl.Utf8,
    "lat": pl.Float64,
    "lon": pl.Float64,
    "levels": pl.Int64,
    "flats": pl.Int64,
}


class BuildValuationObjects:
    def __init__(
        self,
        raw_data: RawDataPort,
        store: ValuationObjectStorePort,
        buildings_key: str,
    ) -> None:
        self._raw_data = raw_data
        self._store = store
        self._buildings_key = buildings_key

    def execute(self, region_code: str, asset_classes: list[AssetClass]) -> None:
        buildings = pl.read_csv(
            io.BytesIO(self._raw_data.read_bytes(self._buildings_key)),
            schema_overrides=_BUILDINGS_SCHEMA_OVERRIDES,
        )
        objects = assemble_valuation_objects(buildings)

        for asset_class in asset_classes:
            slice_df = objects.filter(pl.col("asset_class") == asset_class.value)
            if slice_df.is_empty():
                slice_df = pl.DataFrame(schema=_OUTPUT_SCHEMA)
            self._store.save(region_code, asset_class, slice_df)
