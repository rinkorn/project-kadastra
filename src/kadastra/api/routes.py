"""HTTP API for the per-object inspector and per-hex aggregate map.

Endpoints:

- ``GET /api/hex_aggregates`` — return per-hex aggregate values for a
  given (resolution, asset_class, feature) triple. Drives the hex
  layer of the map UI.
- ``GET /api/hex_aggregates/{h3_index}`` — full aggregate row for a
  single hex (every aggregate column), powering the hex inspector.
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
from kadastra.usecases.get_cell_tsorf import (
    CELL_TSORF_FEATURE_SETS,
    GetCellTsorf,
)
from kadastra.usecases.get_hex_aggregates import (
    ASSET_CLASS_VALUES,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    GetHexAggregates,
)
from kadastra.usecases.get_market_reference import GetMarketReference
from kadastra.usecases.load_object_inspection import (
    OBJECT_FEATURE_COLUMNS,
    RAW_OBJECT_FEATURE_COLUMNS,
    LoadObjectInspection,
)

# ADR-0016 quartet — model selector accepted by /api/inspection and
# /api/hex_aggregates. ``ebm`` (White Box) is the default everywhere, so
# the map and the inspector center the interpretable model by default.
QUARTET_MODELS = ("catboost", "ebm", "grey_tree", "naive_linear")

# ADR-0017 geometry — converted once per detail request. Constructed once
# at module load: web-mercator metres (silver/gold storage CRS) → WGS84
# lon/lat degrees (deck.gl + maplibre input).
_WGS84_FROM_3857 = pyproj.Transformer.from_crs(3857, 4326, always_xy=True)
# ADR-0027: the Слой 1 grid resolution the «ЦОФ-сетка» mode reads. Matches
# Settings.cell_tsorf_resolution default; the API doesn't import Settings
# to keep the web layer free of config coupling, so the constant is local.
_CELL_TSORF_RESOLUTION = 10


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
    get_cell_tsorf: GetCellTsorf,
) -> APIRouter:
    router = APIRouter(prefix="/api")

    @router.get("/hex_aggregates")
    def hex_aggregates(
        resolution: int = Query(..., ge=0, le=15),
        asset_class: str = Query(...),
        feature: str = Query(...),
        model: str = Query("ebm"),
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

    @router.get("/hex_aggregates/{h3_index}")
    def hex_aggregate_detail(
        h3_index: str,
        resolution: int = Query(..., ge=0, le=15),
        asset_class: str = Query(...),
        model: str = Query("ebm"),
    ) -> dict[str, Any]:
        """Full aggregate row for a single hex (every aggregate column
        for that h3_index/asset_class), powering the hex inspector."""
        if asset_class not in ASSET_CLASS_VALUES:
            raise HTTPException(
                status_code=400,
                detail=f"unknown asset_class: {asset_class!r}; expected one of {ASSET_CLASS_VALUES}",
            )
        _validate_model(model)
        try:
            data = get_hex_aggregates.get_detail(region_code, resolution, asset_class, h3_index, model=model)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        if data is None:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"no hex aggregate for h3_index={h3_index!r} asset_class={asset_class!r} resolution={resolution}"
                ),
            )
        return {
            "region": region_code,
            "resolution": resolution,
            "asset_class": asset_class,
            "model": model,
            "data": data,
        }

    @router.get("/inspection")
    def inspection_list(
        asset_class: str = Query(...),
        model: str = Query("ebm"),
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

    @router.get("/inspection/{object_id:path}/explain")
    def inspection_detail_explain(
        object_id: str,
        asset_class: str = Query(...),
    ) -> dict[str, Any]:
        """EBM (White Box) per-feature contribution breakdown for one
        object: ``{intercept, terms: [{feature, value, contribution}]}``.
        ``intercept + Σ contribution`` equals the EBM prediction."""
        ac = _parse_asset_class(asset_class)
        explanation = load_inspection.get_explanation(region_code, ac, object_id)
        if explanation is None:
            raise HTTPException(
                status_code=404,
                detail=f"no EBM explanation for object {object_id!r} asset_class={ac.value}",
            )
        return {
            "region": region_code,
            "asset_class": ac.value,
            "model": "ebm",
            "data": explanation,
        }

    @router.get("/inspection/{object_id:path}")
    def inspection_detail(
        object_id: str,
        asset_class: str = Query(...),
        model: str = Query("ebm"),
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
        all_feature_names = (
            list(NUMERIC_FEATURES)
            + list(CATEGORICAL_FEATURES)
            + list(OBJECT_FEATURE_COLUMNS)
            + list(RAW_OBJECT_FEATURE_COLUMNS)
        )
        # Слой 1 cell ЦОФ (ADR-0027) — the «ЦОФ-сетка» mode's two-level
        # selector: {feature_set: [features]}. Built once so the frontend
        # gets everything in one round-trip; empty for sets not yet built.
        cell_tsorf = get_cell_tsorf.feature_set_map(region_code, _CELL_TSORF_RESOLUTION)
        all_feature_names += [f for feats in cell_tsorf.values() for f in feats]
        return {
            "asset_classes": list(ASSET_CLASS_VALUES),
            "numeric_features": list(NUMERIC_FEATURES),
            "categorical_features": list(CATEGORICAL_FEATURES),
            "object_features": list(OBJECT_FEATURE_COLUMNS),
            "models": list(QUARTET_MODELS),
            "cell_tsorf_resolution": _CELL_TSORF_RESOLUTION,
            "cell_tsorf_feature_sets": list(CELL_TSORF_FEATURE_SETS),
            "cell_tsorf_features": cell_tsorf,
            # Single source of truth for per-feature tooltips. The map UI
            # reads this dict and falls back to nothing if a key is
            # missing — see domain/feature_descriptions.py.
            "feature_descriptions": {name: describe_feature(name) for name in all_feature_names},
        }

    @router.get("/cell_tsorf")
    def cell_tsorf(
        resolution: int = Query(..., ge=0, le=15),
        feature_set: str = Query(...),
        feature: str = Query(...),
    ) -> dict[str, Any]:
        """Слой 1 cell ЦОФ (ADR-0027) for the map UI's «ЦОФ-сетка» mode:
        ``[{hex, value}]`` for one feature column across all res cells.
        Not asset-class / model scoped — this is input location factors,
        not object predictions."""
        try:
            data = get_cell_tsorf.execute(region_code, resolution, feature_set, feature)
        except KeyError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {
            "region": region_code,
            "resolution": resolution,
            "feature_set": feature_set,
            "feature": feature,
            "data": data,
        }

    @router.get("/cell_tsorf/{h3_index}")
    def cell_tsorf_detail(
        h3_index: str,
        resolution: int = Query(..., ge=0, le=15),
    ) -> dict[str, Any]:
        """Full Layer 1 cell ЦОФ across all built feature sets for a single cell.

        Feeds the cell inspector panel when clicking on a hex in «ЦОФ-сетка» mode.
        """
        data = get_cell_tsorf.get_cell_detail(region_code, resolution, h3_index)
        if not data:
            raise HTTPException(
                status_code=404,
                detail=f"cell {h3_index!r} not found at resolution={resolution} in region={region_code!r}",
            )
        return {
            "region": region_code,
            "resolution": resolution,
            "h3_index": h3_index,
            "data": data,
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
