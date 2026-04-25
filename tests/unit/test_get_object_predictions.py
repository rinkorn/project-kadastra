import polars as pl
import pytest

from kadastra.domain.asset_class import AssetClass
from kadastra.usecases.get_object_predictions import GetObjectPredictions


class _FakeReader:
    def __init__(self, by_class: dict[AssetClass, pl.DataFrame]) -> None:
        self._by_class = by_class

    def load(self, region_code: str, asset_class: AssetClass) -> pl.DataFrame:
        if asset_class not in self._by_class:
            raise FileNotFoundError(f"missing {asset_class}")
        return self._by_class[asset_class]


def _predictions(rows: list[tuple[str, float, float, float]]) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "object_id": oid,
                "asset_class": "apartment",
                "lat": lat,
                "lon": lon,
                "predicted_value": pv,
            }
            for oid, lat, lon, pv in rows
        ],
        schema={
            "object_id": pl.Utf8,
            "asset_class": pl.Utf8,
            "lat": pl.Float64,
            "lon": pl.Float64,
            "predicted_value": pl.Float64,
        },
    )


def test_returns_one_record_per_object_with_lat_lon_value() -> None:
    reader = _FakeReader(
        {AssetClass.APARTMENT: _predictions([("way/1", 55.78, 49.12, 75_000.0)])}
    )
    usecase = GetObjectPredictions(reader)

    out = usecase.execute("RU-KAZAN-AGG", AssetClass.APARTMENT)

    assert out == [
        {
            "object_id": "way/1",
            "lat": 55.78,
            "lon": 49.12,
            "value": 75_000.0,
        }
    ]


def test_drops_rows_with_null_predicted_value() -> None:
    df = _predictions([("way/1", 55.78, 49.12, 75_000.0), ("way/2", 55.79, 49.13, 0.0)])
    df = df.with_columns(
        pl.when(pl.col("object_id") == "way/2")
        .then(None)
        .otherwise(pl.col("predicted_value"))
        .alias("predicted_value")
    )
    reader = _FakeReader({AssetClass.APARTMENT: df})
    usecase = GetObjectPredictions(reader)

    out = usecase.execute("RU-KAZAN-AGG", AssetClass.APARTMENT)

    assert len(out) == 1
    assert out[0]["object_id"] == "way/1"


def test_propagates_filenotfound_when_partition_missing() -> None:
    reader = _FakeReader({})
    usecase = GetObjectPredictions(reader)

    with pytest.raises(FileNotFoundError):
        usecase.execute("RU-KAZAN-AGG", AssetClass.HOUSE)


def test_returns_empty_list_for_empty_partition() -> None:
    empty = pl.DataFrame(
        schema={
            "object_id": pl.Utf8,
            "asset_class": pl.Utf8,
            "lat": pl.Float64,
            "lon": pl.Float64,
            "predicted_value": pl.Float64,
        }
    )
    reader = _FakeReader({AssetClass.COMMERCIAL: empty})
    usecase = GetObjectPredictions(reader)

    assert usecase.execute("RU-KAZAN-AGG", AssetClass.COMMERCIAL) == []
