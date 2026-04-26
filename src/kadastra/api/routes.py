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
from kadastra.usecases.get_hex_aggregates import (
    ASSET_CLASS_VALUES,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    GetHexAggregates,
)
from kadastra.usecases.load_object_inspection import LoadObjectInspection

# Constructed once: web-mercator metres (silver/gold storage CRS) →
# WGS84 lon/lat degrees (deck.gl + maplibre input).
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
) -> APIRouter:
    router = APIRouter(prefix="/api")

    @router.get("/hex_aggregates")
    def hex_aggregates(
        resolution: int = Query(..., ge=0, le=15),
        asset_class: str = Query(...),
        feature: str = Query(...),
    ) -> dict[str, Any]:
        if asset_class not in ASSET_CLASS_VALUES:
            raise HTTPException(
                status_code=400,
                detail=f"unknown asset_class: {asset_class!r}; expected one of {ASSET_CLASS_VALUES}",
            )
        try:
            data = get_hex_aggregates.execute(
                region_code, resolution, asset_class, feature
            )
        except KeyError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {
            "region": region_code,
            "resolution": resolution,
            "asset_class": asset_class,
            "feature": feature,
            "is_categorical": feature in CATEGORICAL_FEATURES,
            "is_numeric": feature in NUMERIC_FEATURES,
            "data": data,
        }

    @router.get("/inspection")
    def inspection_list(asset_class: str = Query(...)) -> dict[str, Any]:
        ac = _parse_asset_class(asset_class)
        return {
            "region": region_code,
            "asset_class": ac.value,
            "data": load_inspection.list_for_map(region_code, ac),
        }

    @router.get("/inspection/{object_id:path}")
    def inspection_detail(
        object_id: str,
        asset_class: str = Query(...),
    ) -> dict[str, Any]:
        ac = _parse_asset_class(asset_class)
        detail = load_inspection.get_detail(region_code, ac, object_id)
        if detail is None:
            raise HTTPException(
                status_code=404,
                detail=f"object {object_id!r} not found for asset_class={ac.value}",
            )
        wkt = detail.pop("polygon_wkt_3857", None)
        detail["geometry"] = _convert_wkt_3857_to_geojson_wgs84(wkt)
        return {"region": region_code, "asset_class": ac.value, "data": detail}

    @router.get("/feature_options")
    def feature_options() -> dict[str, Any]:
        return {
            "asset_classes": list(ASSET_CLASS_VALUES),
            "numeric_features": list(NUMERIC_FEATURES),
            "categorical_features": list(CATEGORICAL_FEATURES),
        }

    return router


def _parse_asset_class(asset_class: str) -> AssetClass:
    try:
        return AssetClass(asset_class)
    except ValueError as exc:
        raise HTTPException(
            status_code=400, detail=f"unknown asset_class: {asset_class!r}"
        ) from exc
