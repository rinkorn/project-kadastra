"""Tests for LoadObjectInspection.

Joins gold per-object frame with OOF predictions; exposes a slim list
for map rendering and a full dict for the inspector side panel.
"""

from __future__ import annotations

import polars as pl

from kadastra.domain.asset_class import AssetClass
from kadastra.usecases.load_object_inspection import LoadObjectInspection


class _FakeReader:
    def __init__(self, by_class: dict[AssetClass, pl.DataFrame]) -> None:
        self._by_class = by_class

    def load(self, region_code: str, asset_class: AssetClass) -> pl.DataFrame:
        return self._by_class.get(asset_class, pl.DataFrame())


class _FakeOofReader:
    def __init__(self, by_class: dict[AssetClass, pl.DataFrame]) -> None:
        self._by_class = by_class

    def load_latest(self, asset_class: AssetClass) -> pl.DataFrame:
        return self._by_class.get(
            asset_class,
            pl.DataFrame(
                schema={
                    "object_id": pl.Utf8,
                    "lat": pl.Float64,
                    "lon": pl.Float64,
                    "fold_id": pl.Int64,
                    "y_true": pl.Float64,
                    "y_pred_oof": pl.Float64,
                }
            ),
        )


def _objects(rows: list[dict[str, object]]) -> pl.DataFrame:
    return pl.DataFrame(
        rows,
        schema={
            "object_id": pl.Utf8,
            "asset_class": pl.Utf8,
            "lat": pl.Float64,
            "lon": pl.Float64,
            "synthetic_target_rub_per_m2": pl.Float64,
            "intra_city_raion": pl.Utf8,
            "levels": pl.Int64,
        },
    )


def _oof(rows: list[dict[str, object]]) -> pl.DataFrame:
    return pl.DataFrame(
        rows,
        schema={
            "object_id": pl.Utf8,
            "lat": pl.Float64,
            "lon": pl.Float64,
            "fold_id": pl.Int64,
            "y_true": pl.Float64,
            "y_pred_oof": pl.Float64,
        },
    )


def test_list_for_map_joins_predictions() -> None:
    objects = _objects(
        [
            {
                "object_id": "a1", "asset_class": "apartment",
                "lat": 55.78, "lon": 49.12,
                "synthetic_target_rub_per_m2": 100_000.0,
                "intra_city_raion": "Советский", "levels": 5,
            }
        ]
    )
    oof = _oof(
        [
            {
                "object_id": "a1", "lat": 55.78, "lon": 49.12,
                "fold_id": 2, "y_true": 100_000.0, "y_pred_oof": 95_000.0,
            }
        ]
    )

    usecase = LoadObjectInspection(
        reader=_FakeReader({AssetClass.APARTMENT: objects}),
        oof_reader=_FakeOofReader({AssetClass.APARTMENT: oof}),
    )
    result = usecase.list_for_map("RU-KAZAN-AGG", AssetClass.APARTMENT)

    assert len(result) == 1
    item = result[0]
    assert item["object_id"] == "a1"
    assert item["y_true"] == 100_000.0
    assert item["y_pred_oof"] == 95_000.0
    assert item["residual"] == -5_000.0
    assert item["fold_id"] == 2
    assert set(item.keys()) == {
        "object_id", "lat", "lon", "y_true", "y_pred_oof", "residual", "fold_id"
    }


def test_list_for_map_null_predictions_when_oof_missing() -> None:
    """When the OOF artifact is missing for a class, the map list still
    contains the objects with null predictions / null residuals."""
    objects = _objects(
        [
            {
                "object_id": "a1", "asset_class": "apartment",
                "lat": 55.78, "lon": 49.12,
                "synthetic_target_rub_per_m2": 100_000.0,
                "intra_city_raion": "Советский", "levels": 5,
            }
        ]
    )
    usecase = LoadObjectInspection(
        reader=_FakeReader({AssetClass.APARTMENT: objects}),
        oof_reader=_FakeOofReader({}),
    )
    result = usecase.list_for_map("RU-KAZAN-AGG", AssetClass.APARTMENT)
    item = result[0]
    assert item["y_true"] == 100_000.0
    assert item["y_pred_oof"] is None
    assert item["residual"] is None
    assert item["fold_id"] is None


def test_list_for_map_empty_when_no_objects() -> None:
    usecase = LoadObjectInspection(
        reader=_FakeReader({}),
        oof_reader=_FakeOofReader({}),
    )
    assert usecase.list_for_map("RU-KAZAN-AGG", AssetClass.APARTMENT) == []


def test_get_detail_returns_full_feature_dict() -> None:
    """The detail view returns every gold column for the matching
    object (renamed target → y_true, plus joined OOF columns)."""
    objects = _objects(
        [
            {
                "object_id": "a1", "asset_class": "apartment",
                "lat": 55.78, "lon": 49.12,
                "synthetic_target_rub_per_m2": 100_000.0,
                "intra_city_raion": "Советский", "levels": 5,
            },
            {
                "object_id": "a2", "asset_class": "apartment",
                "lat": 55.79, "lon": 49.13,
                "synthetic_target_rub_per_m2": 110_000.0,
                "intra_city_raion": "Вахитовский", "levels": 9,
            },
        ]
    )
    oof = _oof(
        [
            {
                "object_id": "a1", "lat": 55.78, "lon": 49.12,
                "fold_id": 2, "y_true": 100_000.0, "y_pred_oof": 95_000.0,
            }
        ]
    )

    usecase = LoadObjectInspection(
        reader=_FakeReader({AssetClass.APARTMENT: objects}),
        oof_reader=_FakeOofReader({AssetClass.APARTMENT: oof}),
    )

    detail = usecase.get_detail("RU-KAZAN-AGG", AssetClass.APARTMENT, "a1")
    assert detail is not None
    assert detail["object_id"] == "a1"
    assert detail["intra_city_raion"] == "Советский"
    assert detail["levels"] == 5
    assert detail["y_true"] == 100_000.0
    assert detail["y_pred_oof"] == 95_000.0
    assert detail["residual"] == -5_000.0


def test_get_detail_returns_none_for_unknown_object_id() -> None:
    objects = _objects(
        [
            {
                "object_id": "a1", "asset_class": "apartment",
                "lat": 55.78, "lon": 49.12,
                "synthetic_target_rub_per_m2": 100_000.0,
                "intra_city_raion": "Советский", "levels": 5,
            }
        ]
    )
    usecase = LoadObjectInspection(
        reader=_FakeReader({AssetClass.APARTMENT: objects}),
        oof_reader=_FakeOofReader({}),
    )
    assert usecase.get_detail("RU-KAZAN-AGG", AssetClass.APARTMENT, "missing") is None


def test_get_detail_returns_none_when_class_empty() -> None:
    usecase = LoadObjectInspection(
        reader=_FakeReader({}),
        oof_reader=_FakeOofReader({}),
    )
    assert usecase.get_detail("RU-KAZAN-AGG", AssetClass.APARTMENT, "a1") is None
