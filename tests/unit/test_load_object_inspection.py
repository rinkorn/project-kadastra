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
    def __init__(
        self,
        by_class: dict[AssetClass, pl.DataFrame] | None = None,
        *,
        by_model: dict[tuple[AssetClass, str], pl.DataFrame] | None = None,
    ) -> None:
        # Two ctor styles: ``by_class`` keeps the legacy single-model
        # tests intact; ``by_model`` lets us pin a per-(class, model)
        # frame for the new ADR-0016 quartet tests.
        self._by_class = by_class or {}
        self._by_model = by_model or {}
        self.model_calls: list[tuple[AssetClass, str]] = []

    def load_latest(
        self, asset_class: AssetClass, *, model: str = "catboost"
    ) -> pl.DataFrame:
        self.model_calls.append((asset_class, model))
        if (asset_class, model) in self._by_model:
            return self._by_model[(asset_class, model)]
        if model == "catboost" and asset_class in self._by_class:
            return self._by_class[asset_class]
        return pl.DataFrame(
            schema={
                "object_id": pl.Utf8,
                "lat": pl.Float64,
                "lon": pl.Float64,
                "fold_id": pl.Int64,
                "y_true": pl.Float64,
                "y_pred_oof": pl.Float64,
            }
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


# --- ADR-0016 quartet support ---------------------------------------


def test_list_for_map_threads_model_param_to_oof_reader() -> None:
    """When the API passes ``model='ebm'`` into list_for_map, the OOF
    reader must receive that same model — otherwise the map would
    silently keep showing CatBoost predictions."""
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
    ebm_oof = _oof(
        [
            {
                "object_id": "a1", "lat": 55.78, "lon": 49.12,
                "fold_id": 2, "y_true": 100_000.0, "y_pred_oof": 92_000.0,
            }
        ]
    )
    cb_oof = _oof(
        [
            {
                "object_id": "a1", "lat": 55.78, "lon": 49.12,
                "fold_id": 2, "y_true": 100_000.0, "y_pred_oof": 95_000.0,
            }
        ]
    )
    fake_oof = _FakeOofReader(
        by_model={
            (AssetClass.APARTMENT, "catboost"): cb_oof,
            (AssetClass.APARTMENT, "ebm"): ebm_oof,
        }
    )
    usecase = LoadObjectInspection(
        reader=_FakeReader({AssetClass.APARTMENT: objects}),
        oof_reader=fake_oof,
    )

    result = usecase.list_for_map("RU-KAZAN-AGG", AssetClass.APARTMENT, model="ebm")
    assert result[0]["y_pred_oof"] == 92_000.0
    assert (AssetClass.APARTMENT, "ebm") in fake_oof.model_calls


def test_get_detail_quartet_returns_per_model_predictions() -> None:
    """Side-panel comparison view: one shared gold feature dict +
    a ``models`` dict with per-model ``y_pred_oof`` / ``residual`` /
    ``fold_id``. Folds are same across models in ADR-0016 (single
    spatial split shared by all 4 trainers), so each per-model
    entry can be compared like-for-like."""
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
    cb_oof = _oof(
        [{"object_id": "a1", "lat": 55.78, "lon": 49.12,
          "fold_id": 2, "y_true": 100_000.0, "y_pred_oof": 95_000.0}]
    )
    ebm_oof = _oof(
        [{"object_id": "a1", "lat": 55.78, "lon": 49.12,
          "fold_id": 2, "y_true": 100_000.0, "y_pred_oof": 92_000.0}]
    )
    grey_oof = _oof(
        [{"object_id": "a1", "lat": 55.78, "lon": 49.12,
          "fold_id": 2, "y_true": 100_000.0, "y_pred_oof": 96_500.0}]
    )
    naive_oof = _oof(
        [{"object_id": "a1", "lat": 55.78, "lon": 49.12,
          "fold_id": 2, "y_true": 100_000.0, "y_pred_oof": 110_000.0}]
    )
    fake_oof = _FakeOofReader(
        by_model={
            (AssetClass.APARTMENT, "catboost"): cb_oof,
            (AssetClass.APARTMENT, "ebm"): ebm_oof,
            (AssetClass.APARTMENT, "grey_tree"): grey_oof,
            (AssetClass.APARTMENT, "naive_linear"): naive_oof,
        }
    )
    usecase = LoadObjectInspection(
        reader=_FakeReader({AssetClass.APARTMENT: objects}),
        oof_reader=fake_oof,
    )

    detail = usecase.get_detail_quartet(
        "RU-KAZAN-AGG", AssetClass.APARTMENT, "a1"
    )
    assert detail is not None
    # Shared gold features are at the top level (single source of truth).
    assert detail["object_id"] == "a1"
    assert detail["intra_city_raion"] == "Советский"
    assert detail["y_true"] == 100_000.0
    # Per-model breakdown.
    models = detail["models"]
    assert set(models.keys()) == {"catboost", "ebm", "grey_tree", "naive_linear"}
    assert models["catboost"]["y_pred_oof"] == 95_000.0
    assert models["catboost"]["residual"] == -5_000.0
    assert models["catboost"]["fold_id"] == 2
    assert models["ebm"]["y_pred_oof"] == 92_000.0
    assert models["ebm"]["residual"] == -8_000.0
    assert models["grey_tree"]["y_pred_oof"] == 96_500.0
    assert models["naive_linear"]["y_pred_oof"] == 110_000.0
    assert models["naive_linear"]["residual"] == 10_000.0
    # No raw per-model y_pred_oof at top level (those live under "models").
    assert "y_pred_oof" not in detail
    assert "residual" not in detail


def test_get_detail_quartet_returns_nulls_for_missing_oof_per_model() -> None:
    """If only some quartet OOFs are present (e.g. EBM run failed),
    the missing models still appear in ``models`` with nulls — so the
    UI can render an empty cell instead of dropping the column."""
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
    cb_oof = _oof(
        [{"object_id": "a1", "lat": 55.78, "lon": 49.12,
          "fold_id": 2, "y_true": 100_000.0, "y_pred_oof": 95_000.0}]
    )
    fake_oof = _FakeOofReader(
        by_model={(AssetClass.APARTMENT, "catboost"): cb_oof},
    )
    usecase = LoadObjectInspection(
        reader=_FakeReader({AssetClass.APARTMENT: objects}),
        oof_reader=fake_oof,
    )

    detail = usecase.get_detail_quartet(
        "RU-KAZAN-AGG", AssetClass.APARTMENT, "a1"
    )
    assert detail is not None
    assert detail["models"]["catboost"]["y_pred_oof"] == 95_000.0
    assert detail["models"]["ebm"]["y_pred_oof"] is None
    assert detail["models"]["ebm"]["residual"] is None
    assert detail["models"]["ebm"]["fold_id"] is None


def test_get_detail_quartet_returns_none_for_unknown_object() -> None:
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
    assert (
        usecase.get_detail_quartet("RU-KAZAN-AGG", AssetClass.APARTMENT, "missing")
        is None
    )


def test_get_detail_threads_model_param_to_oof_reader() -> None:
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
    grey_oof = _oof(
        [
            {
                "object_id": "a1", "lat": 55.78, "lon": 49.12,
                "fold_id": 2, "y_true": 100_000.0, "y_pred_oof": 91_500.0,
            }
        ]
    )
    fake_oof = _FakeOofReader(
        by_model={(AssetClass.APARTMENT, "grey_tree"): grey_oof}
    )
    usecase = LoadObjectInspection(
        reader=_FakeReader({AssetClass.APARTMENT: objects}),
        oof_reader=fake_oof,
    )
    detail = usecase.get_detail(
        "RU-KAZAN-AGG", AssetClass.APARTMENT, "a1", model="grey_tree"
    )
    assert detail is not None
    assert detail["y_pred_oof"] == 91_500.0
    assert (AssetClass.APARTMENT, "grey_tree") in fake_oof.model_calls
