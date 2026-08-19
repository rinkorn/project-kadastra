"""Integration tests for the ADR-0027 «ЦОФ-сетка» API surface.

`GET /api/cell_tsorf` and the `cell_tsorf_*` fields of
`/api/feature_options` — the map UI's «ЦОФ-сетка» mode reads Слой 1 cell
ЦОФ straight from the feature store (keyed by h3_index).
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import polars as pl
import pytest
from fastapi.testclient import TestClient

from kadastra.composition_root import create_app
from kadastra.config import Settings


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        region_code="RU-TA",
        coverage_store_path=tmp_path / "coverage",
        feature_store_path=tmp_path / "features",
        region_boundary_path=tmp_path / "boundary.geojson",
        object_predictions_store_path=tmp_path / "object_preds",
        valuation_object_store_path=tmp_path / "valuation_objects",
        hex_aggregates_base_path=tmp_path / "hex_aggregates",
        model_registry_path=tmp_path / "models",
        emiss_silver_base_path=tmp_path / "emiss",
        emiss_market_reference_year=2025,
    )


def _seed_cell_tsorf(settings: Settings) -> None:
    """Two Слой 1 feature sets so the tests can switch between them."""
    part = settings.feature_store_path / f"region={settings.region_code}" / "feature_set=walk_dist" / "resolution=10"
    part.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        [
            {"h3_index": "8a10a84a4d97fff", "resolution": 10, "walk_dist_to_school_m": 538.0},
            {"h3_index": "8a10a84a4db7fff", "resolution": 10, "walk_dist_to_school_m": None},
            {"h3_index": "8a10a84a4daffff", "resolution": 10, "walk_dist_to_school_m": 1.0e9},
        ]
    ).write_parquet(part / "data.parquet")

    part2 = (
        settings.feature_store_path / f"region={settings.region_code}" / "feature_set=road_density" / "resolution=10"
    )
    part2.mkdir(parents=True, exist_ok=True)
    pl.DataFrame([{"h3_index": "8a10a84a4d97fff", "resolution": 10, "road_length_500m": 950.0}]).write_parquet(
        part2 / "data.parquet"
    )


@pytest.fixture
def client(tmp_path: Path) -> Iterator[TestClient]:
    settings = _settings(tmp_path)
    _seed_cell_tsorf(settings)
    app = create_app(settings)
    with TestClient(app) as c:
        yield c


def test_cell_tsorf_returns_hex_value_pairs(client: TestClient) -> None:
    response = client.get(
        "/api/cell_tsorf",
        params={
            "resolution": 10,
            "feature_set": "walk_dist",
            "feature": "walk_dist_to_school_m",
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["region"] == "RU-TA"
    assert payload["feature_set"] == "walk_dist"
    assert payload["feature"] == "walk_dist_to_school_m"
    # null row dropped; sentinel value kept (it is a real value).
    values = {r["hex"]: r["value"] for r in payload["data"]}
    assert values == {
        "8a10a84a4d97fff": 538.0,
        "8a10a84a4daffff": 1.0e9,
    }


def test_cell_tsorf_400_for_unknown_feature(client: TestClient) -> None:
    response = client.get(
        "/api/cell_tsorf",
        params={"resolution": 10, "feature_set": "walk_dist", "feature": "nope"},
    )
    assert response.status_code == 400
    assert "nope" in response.json()["detail"]


def test_cell_tsorf_404_for_missing_feature_set(client: TestClient) -> None:
    response = client.get(
        "/api/cell_tsorf",
        params={"resolution": 10, "feature_set": "zonal", "feature": "x"},
    )
    assert response.status_code == 404


def test_feature_options_includes_cell_tsorf_block(client: TestClient) -> None:
    response = client.get("/api/feature_options")
    assert response.status_code == 200
    opts = response.json()
    assert opts["cell_tsorf_resolution"] == 10
    assert "walk_dist" in opts["cell_tsorf_feature_sets"]
    # {feature_set: [features]} — built sets non-empty, missing sets empty.
    feats = opts["cell_tsorf_features"]
    assert "walk_dist_to_school_m" in feats["walk_dist"]
    assert feats["zonal"] == []
    # Tooltip covers the Слой 1 feature name.
    assert "walk_dist_to_school_m" in opts["feature_descriptions"]
    assert "пешеходная" in opts["feature_descriptions"]["walk_dist_to_school_m"].lower()
