import polars as pl
import pytest

from kadastra.usecases.get_hex_features import GetHexFeatures


class FakeGoldReader:
    def __init__(self, by_resolution: dict[int, pl.DataFrame]) -> None:
        self._by_resolution = by_resolution

    def load(self, region_code: str, resolution: int) -> pl.DataFrame:
        return self._by_resolution[resolution]


def _gold_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "h3_index": ["h_a", "h_b", "h_c"],
            "resolution": [8, 8, 8],
            "building_count": [3, 0, 1],
            "dist_metro_m": [10.0, None, 200.0],
        }
    )


def test_returns_hex_value_pairs_for_requested_feature() -> None:
    usecase = GetHexFeatures(FakeGoldReader({8: _gold_df()}))

    result = usecase.execute("RU-TA", 8, "building_count")

    assert {(r["hex"], r["value"]) for r in result} == {("h_a", 3), ("h_b", 0), ("h_c", 1)}


def test_drops_nulls_for_requested_feature() -> None:
    usecase = GetHexFeatures(FakeGoldReader({8: _gold_df()}))

    result = usecase.execute("RU-TA", 8, "dist_metro_m")

    hexes = {r["hex"] for r in result}
    assert hexes == {"h_a", "h_c"}  # h_b had null


def test_raises_keyerror_for_unknown_feature() -> None:
    usecase = GetHexFeatures(FakeGoldReader({8: _gold_df()}))

    with pytest.raises(KeyError, match="not in gold table"):
        usecase.execute("RU-TA", 8, "nope")
