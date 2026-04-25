from collections.abc import Iterator
from pathlib import Path

import polars as pl
import pytest
from fastapi.testclient import TestClient

from kadastra.adapters.parquet_gold_feature_store import ParquetGoldFeatureStore
from kadastra.composition_root import create_app
from kadastra.config import Settings


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        region_code="RU-TA",
        gold_store_path=tmp_path / "gold",
        coverage_store_path=tmp_path / "coverage",
        feature_store_path=tmp_path / "features",
        region_boundary_path=tmp_path / "boundary.geojson",
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


@pytest.fixture
def client(tmp_path: Path) -> Iterator[TestClient]:
    settings = _settings(tmp_path)
    _seed_gold(settings)
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
