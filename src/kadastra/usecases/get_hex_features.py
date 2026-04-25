import polars as pl

from kadastra.ports.gold_feature_reader import GoldFeatureReaderPort


class GetHexFeatures:
    def __init__(self, gold_reader: GoldFeatureReaderPort) -> None:
        self._gold_reader = gold_reader

    def execute(self, region_code: str, resolution: int, feature: str) -> list[dict[str, object]]:
        df = self._gold_reader.load(region_code, resolution)
        if feature not in df.columns:
            available = [c for c in df.columns if c not in {"h3_index", "resolution"}]
            raise KeyError(f"feature {feature!r} not in gold table; available: {available}")

        slim = df.select(["h3_index", pl.col(feature).alias("value")]).drop_nulls("value")
        return [{"hex": row["h3_index"], "value": row["value"]} for row in slim.iter_rows(named=True)]
