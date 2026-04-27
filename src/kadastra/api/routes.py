"""HTTP API for the per-object inspector and per-hex aggregate map.

Endpoints:

- ``GET /api/hex_aggregates`` — return per-hex aggregate values for a
  given (resolution, asset_class, feature) triple. Drives the hex
  layer of the map UI.
- ``GET /api/inspection`` — slim per-object scatter payload
  ``{object_id, lat, lon, y_true, y_pred_oof, residual, fold_id}``
  for the requested asset class. Drives the scatter layer.
- ``GET /api/inspection/{object_id}`` — full feature dict for a
  single object (every gold column + OOF columns) plus a
  ``geometry`` field with the object polygon in GeoJSON WGS84
  (re-projected from the gold ``polygon_wkt_3857`` column at the
  API edge). Powers the side panel + deck.gl PolygonLayer.

The legacy ``/api/hex_features`` endpoint (sourced from the old gold
hex feature store, never re-built after the move to the per-object
pipeline) is retired. The map UI now reads ``/api/hex_aggregates``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pyproj
import shapely
from fastapi import APIRouter, HTTPException, Query
from shapely.geometry import mapping

from kadastra.domain.asset_class import AssetClass
from kadastra.domain.feature_descriptions import describe_feature
from kadastra.usecases.get_hex_aggregates import (
    ASSET_CLASS_VALUES,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    GetHexAggregates,
)
from kadastra.usecases.get_market_reference import GetMarketReference
from kadastra.usecases.load_object_inspection import (
    OBJECT_FEATURE_COLUMNS,
    LoadObjectInspection,
)

# ADR-0016 quartet — model selector accepted by /api/inspection and
# (later) /api/hex_aggregates. ``catboost`` is the default to keep the
# UI working before any quartet run lands.
QUARTET_MODELS = ("catboost", "ebm", "grey_tree", "naive_linear")

# ADR-0017 geometry — converted once per detail request. Constructed once
# at module load: web-mercator metres (silver/gold storage CRS) → WGS84
# lon/lat degrees (deck.gl + maplibre input).
_WGS84_FROM_3857 = pyproj.Transformer.from_crs(3857, 4326, always_xy=True)


def _convert_wkt_3857_to_geojson_wgs84(wkt: str | None) -> dict[str, Any] | None:
    if wkt is None:
        return None
    geom = shapely.from_wkt(wkt)

    def _reproject(coords: np.ndarray) -> np.ndarray:
        lons, lats = _WGS84_FROM_3857.transform(coords[:, 0], coords[:, 1])
        return np.column_stack([lons, lats])

    return mapping(shapely.transform(geom, _reproject))


def make_api_router(
    *,
    region_code: str,
    get_hex_aggregates: GetHexAggregates,
    load_inspection: LoadObjectInspection,
    get_market_reference: GetMarketReference,
    market_reference_year: int,
) -> APIRouter:
    router = APIRouter(prefix="/api")

    @router.get("/hex_aggregates")
    def hex_aggregates(
        resolution: int = Query(..., ge=0, le=15),
        asset_class: str = Query(...),
        feature: str = Query(...),
        model: str = Query("catboost"),
    ) -> dict[str, Any]:
        if asset_class not in ASSET_CLASS_VALUES:
            raise HTTPException(
                status_code=400,
                detail=f"unknown asset_class: {asset_class!r}; expected one of {ASSET_CLASS_VALUES}",
            )
        _validate_model(model)
        try:
            data = get_hex_aggregates.execute(region_code, resolution, asset_class, feature, model=model)
        except KeyError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {
            "region": region_code,
            "resolution": resolution,
            "asset_class": asset_class,
            "feature": feature,
            "model": model,
            "is_categorical": feature in CATEGORICAL_FEATURES,
            "is_numeric": feature in NUMERIC_FEATURES,
            "data": data,
        }

    @router.get("/inspection")
    def inspection_list(
        asset_class: str = Query(...),
        model: str = Query("catboost"),
    ) -> dict[str, Any]:
        ac = _parse_asset_class(asset_class)
        _validate_model(model)
        rows = load_inspection.list_for_map(region_code, ac, model=model)
        # Convert polygon WKT (gold's storage CRS, EPSG:3857) to GeoJSON
        # WGS84 once per row — same convention as the detail endpoint.
        # WKT itself is dropped from the payload (no consumer needs it).
        for row in rows:
            wkt = row.pop("polygon_wkt_3857", None)
            row["geometry"] = _convert_wkt_3857_to_geojson_wgs84(wkt)
        return {
            "region": region_code,
            "asset_class": ac.value,
            "model": model,
            "data": rows,
        }

    @router.get("/inspection/{object_id:path}/quartet")
    def inspection_detail_quartet(
        object_id: str,
        asset_class: str = Query(...),
    ) -> dict[str, Any]:
        ac = _parse_asset_class(asset_class)
        detail = load_inspection.get_detail_quartet(region_code, ac, object_id)
        if detail is None:
            raise HTTPException(
                status_code=404,
                detail=f"object {object_id!r} not found for asset_class={ac.value}",
            )
        wkt = detail.pop("polygon_wkt_3857", None)
        detail["geometry"] = _convert_wkt_3857_to_geojson_wgs84(wkt)
        return {
            "region": region_code,
            "asset_class": ac.value,
            "data": detail,
        }

    @router.get("/inspection/{object_id:path}")
    def inspection_detail(
        object_id: str,
        asset_class: str = Query(...),
        model: str = Query("catboost"),
    ) -> dict[str, Any]:
        ac = _parse_asset_class(asset_class)
        _validate_model(model)
        detail = load_inspection.get_detail(region_code, ac, object_id, model=model)
        if detail is None:
            raise HTTPException(
                status_code=404,
                detail=f"object {object_id!r} not found for asset_class={ac.value}",
            )
        wkt = detail.pop("polygon_wkt_3857", None)
        detail["geometry"] = _convert_wkt_3857_to_geojson_wgs84(wkt)
        return {
            "region": region_code,
            "asset_class": ac.value,
            "model": model,
            "data": detail,
        }

    @router.get("/market_reference")
    def market_reference(
        asset_class: str = Query(...),
        year: int | None = Query(None),
    ) -> dict[str, Any]:
        """ADR-0010 anchor: ЕМИСС/Росстат #61781 average ₽/м² for the
        region's center city, both primary and secondary apartment markets.
        Used by the inspector quartet panel as «вот рынок, а вот наша
        ЕГРН-основанная модель» reference. Apartments only; non-apartment
        classes return ``data: null`` with status 200 (UI treats as
        «no reference available» and hides the row)."""
        ac = _parse_asset_class(asset_class)
        ref_year = year if year is not None else market_reference_year
        data = get_market_reference.execute(
            region_code=region_code,
            asset_class=ac.value,
            year=ref_year,
        )
        return {
            "region": region_code,
            "asset_class": ac.value,
            "year": ref_year,
            "data": data,
        }

    @router.get("/feature_options")
    def feature_options() -> dict[str, Any]:
        all_feature_names = list(NUMERIC_FEATURES) + list(CATEGORICAL_FEATURES) + list(OBJECT_FEATURE_COLUMNS)
        return {
            "asset_classes": list(ASSET_CLASS_VALUES),
            "numeric_features": list(NUMERIC_FEATURES),
            "categorical_features": list(CATEGORICAL_FEATURES),
            "object_features": list(OBJECT_FEATURE_COLUMNS),
            "models": list(QUARTET_MODELS),
            # Single source of truth for per-feature tooltips. The map UI
            # reads this dict and falls back to nothing if a key is
            # missing — see domain/feature_descriptions.py.
            "feature_descriptions": {name: describe_feature(name) for name in all_feature_names},
        }

    return router


def _parse_asset_class(asset_class: str) -> AssetClass:
    try:
        return AssetClass(asset_class)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"unknown asset_class: {asset_class!r}") from exc


def _validate_model(model: str) -> None:
    if model not in QUARTET_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"unknown model: {model!r}; expected one of {QUARTET_MODELS}",
        )
