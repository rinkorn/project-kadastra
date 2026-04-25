import pytest

from kadastra.domain.asset_class import AssetClass
from kadastra.domain.classify_asset_class import classify_asset_class


@pytest.mark.parametrize(
    "tag",
    ["apartments"],
)
def test_apartment_tags_classify_as_apartment(tag: str) -> None:
    assert classify_asset_class(tag) is AssetClass.APARTMENT


@pytest.mark.parametrize(
    "tag",
    ["house", "detached", "semidetached_house", "terrace", "bungalow"],
)
def test_house_tags_classify_as_house(tag: str) -> None:
    assert classify_asset_class(tag) is AssetClass.HOUSE


@pytest.mark.parametrize(
    "tag",
    [
        "retail",
        "commercial",
        "supermarket",
        "kiosk",
        "office",
        "industrial",
        "warehouse",
    ],
)
def test_commercial_tags_classify_as_commercial(tag: str) -> None:
    assert classify_asset_class(tag) is AssetClass.COMMERCIAL


@pytest.mark.parametrize(
    "tag",
    ["yes", "garage", "shed", "barn", "roof", "hut", ""],
)
def test_unrelated_tags_classify_as_none(tag: str) -> None:
    assert classify_asset_class(tag) is None


def test_unknown_string_classifies_as_none() -> None:
    assert classify_asset_class("not-a-real-osm-tag") is None


def test_classification_is_case_insensitive() -> None:
    assert classify_asset_class("APARTMENTS") is AssetClass.APARTMENT
    assert classify_asset_class("Detached") is AssetClass.HOUSE


def test_none_input_returns_none() -> None:
    assert classify_asset_class(None) is None
