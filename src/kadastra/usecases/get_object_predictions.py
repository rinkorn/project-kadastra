import polars as pl

from kadastra.domain.asset_class import AssetClass
from kadastra.ports.valuation_object_reader import ValuationObjectReaderPort

_VALUE_COLUMN = "predicted_value"


class GetObjectPredictions:
    def __init__(self, reader: ValuationObjectReaderPort) -> None:
        self._reader = reader

    def execute(
        self, region_code: str, asset_class: AssetClass
    ) -> list[dict[str, object]]:
        df = self._reader.load(region_code, asset_class)
        if df.is_empty():
            return []
        slim = df.select(
            ["object_id", "lat", "lon", pl.col(_VALUE_COLUMN).alias("value")]
        ).drop_nulls("value")
        return [
            {
                "object_id": row["object_id"],
                "lat": row["lat"],
                "lon": row["lon"],
                "value": row["value"],
            }
            for row in slim.iter_rows(named=True)
        ]
