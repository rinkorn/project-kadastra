import pytest

from kadastra.domain.asset_class import AssetClass


def test_asset_class_has_apartment_house_commercial_landplot() -> None:
    assert AssetClass.APARTMENT.value == "apartment"
    assert AssetClass.HOUSE.value == "house"
    assert AssetClass.COMMERCIAL.value == "commercial"
    assert AssetClass.LANDPLOT.value == "landplot"


def test_asset_class_is_hashable_and_iterable() -> None:
    classes = list(AssetClass)
    assert len(classes) == 4
    assert {c for c in classes} == {
        AssetClass.APARTMENT,
        AssetClass.HOUSE,
        AssetClass.COMMERCIAL,
        AssetClass.LANDPLOT,
    }


def test_asset_class_from_value_round_trips() -> None:
    for c in AssetClass:
        assert AssetClass(c.value) is c


def test_asset_class_unknown_value_raises() -> None:
    with pytest.raises(ValueError):
        AssetClass("not-a-class")
