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
        emiss_silver_base_path=tmp_path / "emiss",
        emiss_market_reference_year=2025,
    )


def _seed_hex_aggregates(settings: Settings) -> None:
    path = (
        settings.hex_aggregates_base_path
        / f"region={settings.region_code}"
        / "resolution=8"
        / "model=catboost"
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
    # WKT below is a tiny square in EPSG:3857 around lon≈49.12, lat≈55.78
    # (Kazan). After reprojection to WGS84 it must land in lon[48..50]
    # lat[55..56].
    wkt_way1 = (
        "POLYGON ((5468000 7530000, 5468100 7530000, "
        "5468100 7530100, 5468000 7530100, 5468000 7530000))"
    )
    df = pl.DataFrame(
        [
            {
                "object_id": "way/1", "asset_class": "apartment",
                "lat": 55.78, "lon": 49.12,
                "synthetic_target_rub_per_m2": 100_000.0,
                "intra_city_raion": "Советский",
                "polygon_wkt_3857": wkt_way1,
                # ADR-0018 geometry features (square 100×100 m).
                "polygon_area_m2": 10_000.0,
                "polygon_perimeter_m": 400.0,
                "polygon_compactness": 0.7854,
                "polygon_convexity": 1.0,
                "bbox_aspect_ratio": 1.0,
                "polygon_orientation_deg": 0.0,
                "polygon_n_vertices": 4,
            },
            {
                "object_id": "way/2", "asset_class": "apartment",
                "lat": 55.79, "lon": 49.13,
                "synthetic_target_rub_per_m2": 110_000.0,
                "intra_city_raion": "Вахитовский",
                "polygon_wkt_3857": None,
                "polygon_area_m2": None,
                "polygon_perimeter_m": None,
                "polygon_compactness": None,
                "polygon_convexity": None,
                "bbox_aspect_ratio": None,
                "polygon_orientation_deg": None,
                "polygon_n_vertices": None,
            },
        ],
        schema={
            "object_id": pl.Utf8, "asset_class": pl.Utf8,
            "lat": pl.Float64, "lon": pl.Float64,
            "synthetic_target_rub_per_m2": pl.Float64,
            "intra_city_raion": pl.Utf8,
            "polygon_wkt_3857": pl.Utf8,
            "polygon_area_m2": pl.Float64,
            "polygon_perimeter_m": pl.Float64,
            "polygon_compactness": pl.Float64,
            "polygon_convexity": pl.Float64,
            "bbox_aspect_ratio": pl.Float64,
            "polygon_orientation_deg": pl.Float64,
            "polygon_n_vertices": pl.Int64,
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


def _seed_emiss(settings: Settings) -> None:
    """Two rows for Tatarstan center-city Q1 2025: secondary 156k +
    primary 240k."""
    base = settings.emiss_silver_base_path / "61781"
    base.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "indicator_id": "61781", "region_okato": "92000000000",
            "region_name": "Татарстан", "mestdom_code": "2",
            "mestdom_name": "Центр субъекта РФ", "unit_code": "rub",
            "unit_name": "руб/м²", "period_code": "q1", "period_name": "Q1",
            "period_quarter": 1, "rynzhel_code": "3",
            "rynzhel_name": "Вторичный рынок жилья",
            "tipkvartir_code": "1", "tipkvartir_name": "Все типы квартир",
            "year": 2025, "period_label": "2025-Q1",
            "value_rub_per_m2": 156_000.0,
        },
        {
            "indicator_id": "61781", "region_okato": "92000000000",
            "region_name": "Татарстан", "mestdom_code": "2",
            "mestdom_name": "Центр субъекта РФ", "unit_code": "rub",
            "unit_name": "руб/м²", "period_code": "q1", "period_name": "Q1",
            "period_quarter": 1, "rynzhel_code": "1",
            "rynzhel_name": "Первичный рынок жилья",
            "tipkvartir_code": "1", "tipkvartir_name": "Все типы квартир",
            "year": 2025, "period_label": "2025-Q1",
            "value_rub_per_m2": 240_000.0,
        },
    ]
    pl.DataFrame(
        rows,
        schema={
            "indicator_id": pl.Utf8, "region_okato": pl.Utf8,
            "region_name": pl.Utf8, "mestdom_code": pl.Utf8,
            "mestdom_name": pl.Utf8, "unit_code": pl.Utf8,
            "unit_name": pl.Utf8, "period_code": pl.Utf8,
            "period_name": pl.Utf8, "period_quarter": pl.Int64,
            "rynzhel_code": pl.Utf8, "rynzhel_name": pl.Utf8,
            "tipkvartir_code": pl.Utf8, "tipkvartir_name": pl.Utf8,
            "year": pl.Int64, "period_label": pl.Utf8,
            "value_rub_per_m2": pl.Float64,
        },
    ).write_parquet(base / "data.parquet")


@pytest.fixture
def client(tmp_path: Path) -> Iterator[TestClient]:
    settings = _settings(tmp_path)
    _seed_hex_aggregates(settings)
    _seed_valuation_objects(settings)
    _seed_oof_predictions(settings)
    _seed_emiss(settings)
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


def test_hex_aggregates_default_model_is_catboost(client: TestClient) -> None:
    """Without ``?model=…`` the endpoint must return the catboost
    partition (so the existing UI keeps working before it's wired)."""
    response = client.get(
        "/api/hex_aggregates",
        params={"resolution": 8, "asset_class": "apartment", "feature": "count"},
    )
    assert response.status_code == 200
    assert response.json()["model"] == "catboost"


def test_hex_aggregates_rejects_unknown_model(client: TestClient) -> None:
    response = client.get(
        "/api/hex_aggregates",
        params={
            "resolution": 8, "asset_class": "apartment",
            "feature": "count", "model": "magic",
        },
    )
    assert response.status_code == 400


def test_hex_aggregates_returns_404_for_missing_model_partition(
    client: TestClient,
) -> None:
    """Test fixture seeds only catboost partition, so ?model=ebm
    must surface as 404, not silently fall back."""
    response = client.get(
        "/api/hex_aggregates",
        params={
            "resolution": 8, "asset_class": "apartment",
            "feature": "count", "model": "ebm",
        },
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


def test_inspection_list_emits_geometry_per_row(client: TestClient) -> None:
    """The scatter list must surface each object's polygon as GeoJSON
    (WGS84) so the map can render the object as a polygon instead of
    a point. WKT stays internal — the response carries `geometry`
    directly and drops `polygon_wkt_3857`. Objects without a polygon
    surface `geometry: null` so the frontend can fall back to a point."""
    response = client.get("/api/inspection", params={"asset_class": "apartment"})
    assert response.status_code == 200
    rows = sorted(response.json()["data"], key=lambda r: r["object_id"])

    way1 = rows[0]
    assert "polygon_wkt_3857" not in way1
    geom = way1["geometry"]
    assert geom is not None
    assert geom["type"] == "Polygon"
    ring = geom["coordinates"][0]
    # Closed ring of the 100×100 m fixture square: 4 corners + closure.
    assert len(ring) == 5
    for lon, lat in ring:
        assert 48.0 <= lon <= 50.0
        assert 55.0 <= lat <= 56.0

    way2 = rows[1]
    assert "polygon_wkt_3857" not in way2
    assert way2["geometry"] is None


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


def test_inspection_detail_emits_geometry_as_geojson_wgs84(
    client: TestClient,
) -> None:
    """The detail card must convert ``polygon_wkt_3857`` (WKT in
    web-mercator metres) into a GeoJSON polygon in WGS84 — the UI
    feeds this directly to deck.gl PolygonLayer which expects
    [lon, lat] in degrees. The raw WKT is stripped (no need to
    ship to the browser)."""
    response = client.get(
        "/api/inspection/way/1", params={"asset_class": "apartment"}
    )
    assert response.status_code == 200
    detail = response.json()["data"]
    geometry = detail["geometry"]
    assert geometry["type"] == "Polygon"
    rings = geometry["coordinates"]
    assert len(rings) == 1
    ring = rings[0]
    # Closed ring: 5 vertices (4 corners + closure).
    assert len(ring) == 5
    for lon, lat in ring:
        assert 48.0 <= lon <= 50.0, f"lon out of Kazan range: {lon}"
        assert 55.0 <= lat <= 56.0, f"lat out of Kazan range: {lat}"
    assert "polygon_wkt_3857" not in detail


def test_inspection_detail_carries_geometry_features(
    client: TestClient,
) -> None:
    """ADR-0018: the 7 derived geometry features (polygon_area_m2,
    perimeter, compactness, convexity, bbox_aspect_ratio, orientation,
    n_vertices) must reach the inspector — they are gold columns and
    surface in the side panel as a dedicated section."""
    response = client.get(
        "/api/inspection/way/1", params={"asset_class": "apartment"}
    )
    assert response.status_code == 200
    detail = response.json()["data"]
    assert detail["polygon_area_m2"] == 10_000.0
    assert detail["polygon_perimeter_m"] == 400.0
    assert detail["polygon_compactness"] == 0.7854
    assert detail["polygon_convexity"] == 1.0
    assert detail["bbox_aspect_ratio"] == 1.0
    assert detail["polygon_orientation_deg"] == 0.0
    assert detail["polygon_n_vertices"] == 4


def test_inspection_detail_geometry_features_null_when_no_polygon(
    client: TestClient,
) -> None:
    """When the underlying object has no polygon (way/2 in fixture has
    polygon_wkt_3857=None), the 7 geometry features must surface as
    null — the UI shows '—' but doesn't crash."""
    response = client.get(
        "/api/inspection/way/2", params={"asset_class": "apartment"}
    )
    assert response.status_code == 200
    detail = response.json()["data"]
    for field in (
        "polygon_area_m2",
        "polygon_perimeter_m",
        "polygon_compactness",
        "polygon_convexity",
        "bbox_aspect_ratio",
        "polygon_orientation_deg",
        "polygon_n_vertices",
    ):
        assert detail[field] is None, f"expected null for {field}"


def test_inspection_detail_geometry_null_when_wkt_missing(
    client: TestClient,
) -> None:
    """When ``polygon_wkt_3857`` is null in gold (e.g. silver row had
    no geometry), the API still returns the detail card with
    ``geometry: null`` — UI must tolerate this."""
    response = client.get(
        "/api/inspection/way/2", params={"asset_class": "apartment"}
    )
    assert response.status_code == 200
    detail = response.json()["data"]
    assert detail["geometry"] is None


def test_inspection_detail_returns_404_for_unknown_object(
    client: TestClient,
) -> None:
    response = client.get(
        "/api/inspection/way/missing", params={"asset_class": "apartment"}
    )
    assert response.status_code == 404


def test_inspection_detail_quartet_returns_per_model_breakdown(
    client: TestClient,
) -> None:
    """Side-panel comparison endpoint: shared gold features +
    geometry at top level + ``models`` dict with one entry per
    ADR-0016 model. Test fixture seeds only catboost OOF, so ebm /
    grey_tree / naive_linear must come back as null entries — the
    UI relies on the column existing to render an empty cell."""
    response = client.get(
        "/api/inspection/way/1/quartet", params={"asset_class": "apartment"}
    )
    assert response.status_code == 200
    payload = response.json()
    detail = payload["data"]
    assert detail["object_id"] == "way/1"
    assert detail["intra_city_raion"] == "Советский"
    assert detail["y_true"] == 100_000.0
    assert detail["geometry"]["type"] == "Polygon"
    models = detail["models"]
    assert set(models.keys()) == {"catboost", "ebm", "grey_tree", "naive_linear"}
    assert models["catboost"]["y_pred_oof"] == 95_000.0
    assert models["catboost"]["fold_id"] == 0
    assert models["catboost"]["residual"] == -5_000.0
    assert models["ebm"]["y_pred_oof"] is None
    assert models["grey_tree"]["y_pred_oof"] is None
    assert models["naive_linear"]["y_pred_oof"] is None


def test_inspection_detail_quartet_returns_404_for_unknown_object(
    client: TestClient,
) -> None:
    response = client.get(
        "/api/inspection/way/missing/quartet",
        params={"asset_class": "apartment"},
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
    # ADR-0016 quartet — UI uses this list to populate the model picker.
    assert {"catboost", "ebm", "grey_tree", "naive_linear"} <= set(payload["models"])


def test_market_reference_returns_emiss_anchor_for_apartment(
    client: TestClient,
) -> None:
    """ADR-0010 anchor: /api/market_reference returns EMISS-61781
    average ₽/м² for the configured year (2025 in fixture). The UI
    quartet panel renders this as «EMISS Казань вторичный/первичный»
    next to the four model OOFs."""
    response = client.get(
        "/api/market_reference",
        params={"asset_class": "apartment"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["region"] == "RU-TA"
    assert payload["asset_class"] == "apartment"
    assert payload["year"] == 2025
    data = payload["data"]
    assert data is not None
    assert data["source"] == "EMISS-61781"
    assert data["secondary_rub_per_m2"] == 156_000.0
    assert data["primary_rub_per_m2"] == 240_000.0


def test_market_reference_returns_null_data_for_non_apartment(
    client: TestClient,
) -> None:
    """EMISS #61781 is apartment-only — house/commercial/landplot
    requests succeed with status 200 but ``data: null`` so the UI
    can hide the row gracefully."""
    response = client.get(
        "/api/market_reference",
        params={"asset_class": "house"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["data"] is None


def test_market_reference_uses_explicit_year_query_param(
    client: TestClient,
) -> None:
    """Caller-supplied ``year`` overrides the configured default —
    used by future UI controls that let the human pick the reference
    year. Out-of-range years return data: null (no EMISS coverage)."""
    response = client.get(
        "/api/market_reference",
        params={"asset_class": "apartment", "year": 2010},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["year"] == 2010
    assert payload["data"] is None


def test_inspection_rejects_unknown_model(client: TestClient) -> None:
    response = client.get(
        "/api/inspection",
        params={"asset_class": "apartment", "model": "magic"},
    )
    assert response.status_code == 400


def test_inspection_default_model_is_catboost(client: TestClient) -> None:
    """Without ``?model=…`` the endpoint must keep returning the
    CatBoost OOF (so the existing UI doesn't break)."""
    response = client.get("/api/inspection", params={"asset_class": "apartment"})
    assert response.status_code == 200
    assert response.json()["model"] == "catboost"


def test_index_serves_html_shell(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    body = response.text
    assert "kadastra" in body
    # Map UI loads deck.gl + maplibre.
    assert "deck.gl" in body or "MapboxOverlay" in body or "H3HexagonLayer" in body
