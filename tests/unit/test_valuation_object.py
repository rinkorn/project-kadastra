import pytest

from kadastra.domain.asset_class import AssetClass
from kadastra.domain.valuation_object import ValuationObject


def test_valuation_object_holds_required_fields() -> None:
    obj = ValuationObject(
        object_id="way/103502712",
        asset_class=AssetClass.APARTMENT,
        lat=55.7887,
        lon=49.1221,
        levels=9,
        flats=72,
    )

    assert obj.object_id == "way/103502712"
    assert obj.asset_class is AssetClass.APARTMENT
    assert obj.lat == 55.7887
    assert obj.lon == 49.1221
    assert obj.levels == 9
    assert obj.flats == 72


def test_valuation_object_levels_and_flats_are_optional() -> None:
    obj = ValuationObject(
        object_id="way/1",
        asset_class=AssetClass.HOUSE,
        lat=55.78,
        lon=49.12,
    )

    assert obj.levels is None
    assert obj.flats is None


def test_valuation_object_is_frozen() -> None:
    obj = ValuationObject(
        object_id="way/1",
        asset_class=AssetClass.HOUSE,
        lat=55.78,
        lon=49.12,
    )

    with pytest.raises(AttributeError):
        obj.lat = 99.9  # type: ignore[misc]


def test_valuation_object_rejects_invalid_lat_lon() -> None:
    with pytest.raises(ValueError):
        ValuationObject(
            object_id="way/1",
            asset_class=AssetClass.HOUSE,
            lat=95.0,
            lon=49.12,
        )
    with pytest.raises(ValueError):
        ValuationObject(
            object_id="way/1",
            asset_class=AssetClass.HOUSE,
            lat=55.78,
            lon=-181.0,
        )


def test_valuation_object_rejects_blank_id() -> None:
    with pytest.raises(ValueError):
        ValuationObject(
            object_id="",
            asset_class=AssetClass.HOUSE,
            lat=55.78,
            lon=49.12,
        )
