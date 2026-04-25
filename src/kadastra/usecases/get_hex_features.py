import polars as pl

from kadastra.ports.gold_feature_reader import GoldFeatureReaderPort

PREDICTED_VALUE_COLUMN = "predicted_value"


class GetHexFeatures:
    def __init__(
        self,
        gold_reader: GoldFeatureReaderPort,
        prediction_reader: GoldFeatureReaderPort | None = None,
    ) -> None:
        self._gold_reader = gold_reader
        self._prediction_reader = prediction_reader

    def execute(self, region_code: str, resolution: int, feature: str) -> list[dict[str, object]]:
        if feature == PREDICTED_VALUE_COLUMN:
            if self._prediction_reader is None:
                raise KeyError(
                    "predicted_value requested but no prediction_reader is configured"
                )
            df = self._prediction_reader.load(region_code, resolution)
        else:
            df = self._gold_reader.load(region_code, resolution)

        if feature not in df.columns:
            available = [c for c in df.columns if c not in {"h3_index", "resolution"}]
            raise KeyError(f"feature {feature!r} not in gold table; available: {available}")

        slim = df.select(["h3_index", pl.col(feature).alias("value")]).drop_nulls("value")
        return [{"hex": row["h3_index"], "value": row["value"]} for row in slim.iter_rows(named=True)]
