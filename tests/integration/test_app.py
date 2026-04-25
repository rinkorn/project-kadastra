from collections.abc import Iterator
from pathlib import Path

import polars as pl
import pytest
from fastapi.testclient import TestClient

from kadastra.adapters.parquet_gold_feature_store import ParquetGoldFeatureStore
from kadastra.adapters.parquet_valuation_object_store import ParquetValuationObjectStore
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
    )


def _seed_gold(settings: Settings) -> None:
    store = ParquetGoldFeatureStore(settings.gold_store_path)
    df = pl.DataFrame(
        {
            "h3_index": ["8810a81015fffff", "8810a81017fffff"],
            "resolution": [8, 8],
            "building_count": [3, 0],
            "road_length_m": [120.5, 0.0],
        }
    )
    store.save("RU-TA", 8, df)


def _seed_object_predictions(settings: Settings) -> None:
    store = ParquetValuationObjectStore(settings.object_predictions_store_path)
    df = pl.DataFrame(
        {
            "object_id": ["way/1", "way/2"],
            "asset_class": ["apartment", "apartment"],
            "lat": [55.78, 55.79],
            "lon": [49.12, 49.13],
            "predicted_value": [75_000.0, 80_000.0],
        }
    )
    store.save("RU-TA", AssetClass.APARTMENT, df)


@pytest.fixture
def client(tmp_path: Path) -> Iterator[TestClient]:
    settings = _settings(tmp_path)
    _seed_gold(settings)
    _seed_object_predictions(settings)
    app = create_app(settings)
    with TestClient(app) as c:
        yield c


def test_hex_features_returns_data_for_known_feature(client: TestClient) -> None:
    response = client.get("/api/hex_features", params={"resolution": 8, "feature": "building_count"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["region"] == "RU-TA"
    assert payload["resolution"] == 8
    assert payload["feature"] == "building_count"
    assert {(row["hex"], row["value"]) for row in payload["data"]} == {
        ("8810a81015fffff", 3),
        ("8810a81017fffff", 0),
    }


def test_hex_features_returns_400_for_unknown_feature(client: TestClient) -> None:
    response = client.get("/api/hex_features", params={"resolution": 8, "feature": "nope"})

    assert response.status_code == 400
    assert "not in gold table" in response.json()["detail"]


def test_hex_features_returns_404_for_missing_resolution(client: TestClient) -> None:
    response = client.get("/api/hex_features", params={"resolution": 9, "feature": "building_count"})

    assert response.status_code == 404


def test_index_serves_html_with_feature_options(client: TestClient) -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    body = response.text
    assert "kadastra" in body
    assert "building_count" in body
    assert "MapboxOverlay" in body or "deck.gl" in body or "deck.MapboxOverlay" in body or "H3HexagonLayer" in body


def test_object_predictions_returns_points_for_known_class(client: TestClient) -> None:
    response = client.get(
        "/api/object_predictions", params={"asset_class": "apartment"}
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["region"] == "RU-TA"
    assert payload["asset_class"] == "apartment"
    assert {(row["object_id"], row["lat"], row["lon"], row["value"]) for row in payload["data"]} == {
        ("way/1", 55.78, 49.12, 75_000.0),
        ("way/2", 55.79, 49.13, 80_000.0),
    }


def test_object_predictions_returns_400_for_unknown_class(client: TestClient) -> None:
    response = client.get(
        "/api/object_predictions", params={"asset_class": "land_plot"}
    )

    assert response.status_code == 400


def test_object_predictions_returns_404_for_missing_class_partition(
    client: TestClient,
) -> None:
    response = client.get(
        "/api/object_predictions", params={"asset_class": "house"}
    )

    assert response.status_code == 404
