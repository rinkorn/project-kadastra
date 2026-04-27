"""Verify every feature name the API actually surfaces gets a description.

Without this guard a newly added feature would arrive in the UI with
no tooltip and no failing test — silent UX regression.
"""

from __future__ import annotations

from kadastra.domain.feature_descriptions import describe_feature
from kadastra.usecases.get_hex_aggregates import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
)
from kadastra.usecases.load_object_inspection import OBJECT_FEATURE_COLUMNS


def test_every_hex_numeric_feature_has_description() -> None:
    missing = [f for f in NUMERIC_FEATURES if not describe_feature(f)]
    assert not missing, f"hex numeric features without description: {missing}"


def test_every_hex_categorical_feature_has_description() -> None:
    missing = [f for f in CATEGORICAL_FEATURES if not describe_feature(f)]
    assert not missing, f"hex categorical features without description: {missing}"


def test_every_object_feature_has_description() -> None:
    missing = [f for f in OBJECT_FEATURE_COLUMNS if not describe_feature(f)]
    assert not missing, f"object features without description: {missing}"


def test_unknown_feature_returns_none() -> None:
    assert describe_feature("totally_made_up_feature") is None


def test_dist_to_pattern_inserts_positive_interpretation() -> None:
    """Distance to a desirable POI should hint that closer = better."""
    text = describe_feature("mean_dist_to_park_m")
    assert text is not None
    assert "положительный фактор" in text
    assert "Усреднено по объектам" in text


def test_dist_to_pattern_inserts_negative_interpretation() -> None:
    text = describe_feature("dist_to_industrial_m")
    assert text is not None
    assert "негативный фактор" in text


def test_share_pattern_explains_zero_one_scale() -> None:
    text = describe_feature("park_share_500m")
    assert text is not None
    assert "0 — нет" in text
    assert "1 — круг полностью покрыт" in text
    assert "500" in text


def test_within_pattern_uses_radius() -> None:
    text = describe_feature("school_within_500m")
    assert text is not None
    assert "500" in text
    assert "школа" in text


def test_dominant_pattern_marks_categorical() -> None:
    text = describe_feature("dominant_intra_city_raion")
    assert text is not None
    assert "Категориальный" in text
