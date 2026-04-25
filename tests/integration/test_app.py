"""Integration tests for the FastAPI app.

Covers:
- ``GET /`` returning the map shell HTML.
- ``GET /api/hex_aggregates`` reading from the gold hex_aggregates parquet.
- ``GET /api/inspection`` returning per-object scatter rows.
- ``GET /api/inspection/{object_id}`` returning a single object detail card.
- ``GET /api/feature_options`` listing the UI's feature/asset_class
  controls.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest
from fastapi.testclient import TestClient

from kadastra.adapters.parquet_valuation_object_store import (
    ParquetValuationObjectStore,
)
from kadastra.composition_root import create_app
from kadastra.config import Settings
from kadastra.domain.asset_class import AssetClass


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        region_code="RU-TA",
        gold_store_path=tmp_path / "gold",
        coverage_store_path=tmp_path / "coverage",
        feature_store_path=tmp_path / "features",
        region_boundary_path=tmp_path / "boundary.geojson",
        object_predictions_store_path=tmp_path / "object_preds",
        valuation_object_store_path=tmp_path / "valuation_objects",
        hex_aggregates_base_path=tmp_path / "hex_aggregates",
        model_registry_path=tmp_path / "models",
    )


def _seed_hex_aggregates(settings: Settings) -> None:
    path = (
        settings.hex_aggregates_base_path
        / f"region={settings.region_code}"
        / "resolution=8"
        / "data.parquet"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pl.DataFrame(
        [
            {
                "h3_index": "8810a81015fffff", "resolution": 8,
                "asset_class": "apartment", "count": 3,
                "median_target_rub_per_m2": 100_000.0,
                "median_pred_oof_rub_per_m2": 95_000.0,
                "median_residual_rub_per_m2": -5_000.0,
                "dominant_intra_city_raion": "Советский",
            },
            {
                "h3_index": "8810a81017fffff", "resolution": 8,
                "asset_class": "apartment", "count": 1,
                "median_target_rub_per_m2": 80_000.0,
                "median_pred_oof_rub_per_m2": 75_000.0,
                "median_residual_rub_per_m2": -5_000.0,
                "dominant_intra_city_raion": "Вахитовский",
            },
        ],
        schema={
            "h3_index": pl.Utf8, "resolution": pl.Int32,
            "asset_class": pl.Utf8, "count": pl.UInt32,
            "median_target_rub_per_m2": pl.Float64,
            "median_pred_oof_rub_per_m2": pl.Float64,
            "median_residual_rub_per_m2": pl.Float64,
            "dominant_intra_city_raion": pl.Utf8,
        },
    )
    df.write_parquet(path)


def _seed_valuation_objects(settings: Settings) -> None:
    store = ParquetValuationObjectStore(settings.valuation_object_store_path)
    df = pl.DataFrame(
        [
            {
                "object_id": "way/1", "asset_class": "apartment",
                "lat": 55.78, "lon": 49.12,
                "synthetic_target_rub_per_m2": 100_000.0,
                "intra_city_raion": "Советский",
            },
            {
                "object_id": "way/2", "asset_class": "apartment",
                "lat": 55.79, "lon": 49.13,
                "synthetic_target_rub_per_m2": 110_000.0,
                "intra_city_raion": "Вахитовский",
            },
        ],
        schema={
            "object_id": pl.Utf8, "asset_class": pl.Utf8,
            "lat": pl.Float64, "lon": pl.Float64,
            "synthetic_target_rub_per_m2": pl.Float64,
            "intra_city_raion": pl.Utf8,
        },
    )
    store.save("RU-TA", AssetClass.APARTMENT, df)


def _seed_oof_predictions(settings: Settings) -> None:
    """Drop a single ``catboost-object-apartment_<ts>/oof_predictions.parquet``
    so the inspector / hex_aggregates can pick up OOF columns."""
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = settings.model_registry_path / f"catboost-object-apartment_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        [
            {
                "object_id": "way/1", "lat": 55.78, "lon": 49.12,
                "fold_id": 0, "y_true": 100_000.0, "y_pred_oof": 95_000.0,
            },
            {
                "object_id": "way/2", "lat": 55.79, "lon": 49.13,
                "fold_id": 1, "y_true": 110_000.0, "y_pred_oof": 105_000.0,
            },
        ],
        schema={
            "object_id": pl.Utf8, "lat": pl.Float64, "lon": pl.Float64,
            "fold_id": pl.Int64, "y_true": pl.Float64, "y_pred_oof": pl.Float64,
        },
    ).write_parquet(run_dir / "oof_predictions.parquet")


@pytest.fixture
def client(tmp_path: Path) -> Iterator[TestClient]:
    settings = _settings(tmp_path)
    _seed_hex_aggregates(settings)
    _seed_valuation_objects(settings)
    _seed_oof_predictions(settings)
    app = create_app(settings)
    with TestClient(app) as c:
        yield c


def test_hex_aggregates_returns_filtered_rows(client: TestClient) -> None:
    response = client.get(
        "/api/hex_aggregates",
        params={"resolution": 8, "asset_class": "apartment", "feature": "median_target_rub_per_m2"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["region"] == "RU-TA"
    assert payload["asset_class"] == "apartment"
    assert payload["feature"] == "median_target_rub_per_m2"
    assert payload["is_numeric"] is True
    assert payload["is_categorical"] is False
    assert {(r["hex"], r["value"]) for r in payload["data"]} == {
        ("8810a81015fffff", 100_000.0),
        ("8810a81017fffff", 80_000.0),
    }


def test_hex_aggregates_categorical_feature(client: TestClient) -> None:
    response = client.get(
        "/api/hex_aggregates",
        params={
            "resolution": 8, "asset_class": "apartment",
            "feature": "dominant_intra_city_raion",
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["is_categorical"] is True
    by_hex = {r["hex"]: r["value"] for r in payload["data"]}
    assert by_hex["8810a81015fffff"] == "Советский"


def test_hex_aggregates_returns_400_for_unknown_feature(client: TestClient) -> None:
    response = client.get(
        "/api/hex_aggregates",
        params={"resolution": 8, "asset_class": "apartment", "feature": "bogus"},
    )
    assert response.status_code == 400


def test_hex_aggregates_returns_400_for_unknown_asset_class(
    client: TestClient,
) -> None:
    response = client.get(
        "/api/hex_aggregates",
        params={"resolution": 8, "asset_class": "spaceship", "feature": "count"},
    )
    assert response.status_code == 400


def test_hex_aggregates_returns_404_for_missing_resolution(
    client: TestClient,
) -> None:
    response = client.get(
        "/api/hex_aggregates",
        params={"resolution": 9, "asset_class": "apartment", "feature": "count"},
    )
    assert response.status_code == 404


def test_inspection_list_joins_oof_predictions(client: TestClient) -> None:
    response = client.get("/api/inspection", params={"asset_class": "apartment"})
    assert response.status_code == 200
    payload = response.json()
    rows = sorted(payload["data"], key=lambda r: r["object_id"])
    assert len(rows) == 2
    way1 = rows[0]
    assert way1["object_id"] == "way/1"
    assert way1["y_true"] == 100_000.0
    assert way1["y_pred_oof"] == 95_000.0
    assert way1["residual"] == -5_000.0
    assert way1["fold_id"] == 0


def test_inspection_detail_returns_full_dict(client: TestClient) -> None:
    response = client.get(
        "/api/inspection/way/1", params={"asset_class": "apartment"}
    )
    assert response.status_code == 200
    payload = response.json()
    detail = payload["data"]
    assert detail["object_id"] == "way/1"
    assert detail["intra_city_raion"] == "Советский"
    assert detail["y_pred_oof"] == 95_000.0


def test_inspection_detail_returns_404_for_unknown_object(
    client: TestClient,
) -> None:
    response = client.get(
        "/api/inspection/way/missing", params={"asset_class": "apartment"}
    )
    assert response.status_code == 404


def test_feature_options_lists_choices(client: TestClient) -> None:
    response = client.get("/api/feature_options")
    assert response.status_code == 200
    payload = response.json()
    assert "all" in payload["asset_classes"]
    assert "apartment" in payload["asset_classes"]
    assert "median_target_rub_per_m2" in payload["numeric_features"]
    assert "dominant_intra_city_raion" in payload["categorical_features"]


def test_index_serves_html_shell(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    body = response.text
    assert "kadastra" in body
    # Map UI loads deck.gl + maplibre.
    assert "deck.gl" in body or "MapboxOverlay" in body or "H3HexagonLayer" in body
