"""Verify every feature name the API actually surfaces gets a description.

Without this guard a newly added feature would arrive in the UI with
no tooltip and no failing test — silent UX regression.
"""

from __future__ import annotations

from kadastra.domain.feature_descriptions import describe_feature, feature_label
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


def test_relative_pattern_diff_med() -> None:
    text = describe_feature("road_length_500m__rel_p7_diff_med")
    assert text is not None
    assert "разрешения 7" in text
    assert "отклонение от медианы" in text


def test_relative_pattern_ratio_and_z() -> None:
    assert describe_feature("road_length_500m__rel_p8_ratio_med") is not None
    assert describe_feature("road_length_500m__rel_p7_z_iqr") is not None


def test_count_parent_pattern() -> None:
    text = describe_feature("count_p7")
    assert text is not None
    assert "разрешения 7" in text


def test_geometry_features_have_descriptions() -> None:
    for f in (
        "polygon_area_m2",
        "polygon_perimeter_m",
        "polygon_compactness",
        "polygon_convexity",
        "bbox_aspect_ratio",
        "polygon_orientation_deg",
        "polygon_n_vertices",
    ):
        assert describe_feature(f) is not None, f


def test_admin_categorical_features_have_descriptions() -> None:
    for f in (
        "materials",
        "era_category",
        "mun_okrug_name",
        "mun_okrug_oktmo",
        "settlement_name",
        "intra_city_raion",
        "oktmo_full",
        "okato",
        "postal_index",
        "age_years_sq",
    ):
        assert describe_feature(f) is not None, f


# --- feature_label: короткие русские подписи для UI ---


def test_label_explicit_dict() -> None:
    assert feature_label("area_m2") == "Площадь, м²"
    assert feature_label("median_target_rub_per_m2") == "Медианная цена, ₽/м²"
    assert feature_label("location_score_rub_per_m2") == "Локационный скор, ₽/м²"


def test_label_dist_to_uses_genitive() -> None:
    assert feature_label("dist_to_secondary_m") == "До районной улицы, м"
    assert feature_label("dist_to_cbd_m") == "До центра (CBD), м"
    assert feature_label("walk_dist_to_school_m") == "Пешком до школы, м"


def test_label_mean_prefix() -> None:
    assert feature_label("mean_dist_to_park_m") == "До парка, м (среднее)"


def test_label_share_within_count() -> None:
    assert feature_label("park_share_500m") == "Доля «парк» в 500 м"
    assert feature_label("school_within_500m") == "Школа в 500 м"
    assert feature_label("count_school_2km") == "Школа в 2 км"


def test_label_relative_and_count_p() -> None:
    assert feature_label("road_length_500m__rel_p7_diff_med") == ("Дороги в 500 м, м — к медиане p7 (отклонение)")
    assert feature_label("count_p7") == "Объектов в гексе p7"


def test_label_pair_terms_split_per_side() -> None:
    assert feature_label("dist_to_cbd_m × polygon_area_m2") == ("До центра (CBD), м × Площадь контура, м²")
    assert feature_label("area_m2 & levels") == "Площадь, м² × Этажность"


def test_label_unknown_falls_back_to_humanized() -> None:
    assert feature_label("weird_future_feature") == "Weird future feature"


def test_every_known_feature_has_label() -> None:
    """Label never returns None and never the raw snake_case name."""
    known = list(NUMERIC_FEATURES) + list(CATEGORICAL_FEATURES) + list(OBJECT_FEATURE_COLUMNS)
    for f in known:
        label = feature_label(f)
        assert label, f
        assert "_" not in label, f"{f} -> {label!r} still looks raw"
