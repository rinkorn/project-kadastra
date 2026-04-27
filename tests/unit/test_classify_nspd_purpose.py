import pytest

from kadastra.domain.asset_class import AssetClass
from kadastra.domain.classify_nspd_purpose import classify_nspd_building_purpose


@pytest.mark.parametrize("purpose", ["Многоквартирный дом", "многоквартирный дом"])
def test_apartment_purpose(purpose: str) -> None:
    assert classify_nspd_building_purpose(purpose) is AssetClass.APARTMENT


@pytest.mark.parametrize("purpose", ["Жилой дом", "Жилое", "Садовый дом", "ЖИЛОЙ ДОМ"])
def test_house_purpose(purpose: str) -> None:
    assert classify_nspd_building_purpose(purpose) is AssetClass.HOUSE


def test_non_residential_classifies_as_commercial() -> None:
    assert classify_nspd_building_purpose("Нежилое") is AssetClass.COMMERCIAL


@pytest.mark.parametrize("purpose", ["Гараж", ""])
def test_skipped_purposes_classify_as_none(purpose: str) -> None:
    assert classify_nspd_building_purpose(purpose) is None


def test_none_input_returns_none() -> None:
    assert classify_nspd_building_purpose(None) is None


def test_unknown_value_returns_none() -> None:
    assert classify_nspd_building_purpose("Что-то невразумительное") is None
