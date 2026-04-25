from typing import Any

from fastapi import APIRouter, HTTPException, Query

from kadastra.domain.asset_class import AssetClass
from kadastra.usecases.get_hex_features import GetHexFeatures
from kadastra.usecases.get_object_predictions import GetObjectPredictions


def make_api_router(
    get_hex_features: GetHexFeatures,
    region_code: str,
    get_object_predictions: GetObjectPredictions | None = None,
) -> APIRouter:
    router = APIRouter(prefix="/api")

    @router.get("/hex_features")
    def hex_features(
        resolution: int = Query(8),
        feature: str = Query("building_count"),
    ) -> dict[str, Any]:
        try:
            data = get_hex_features.execute(region_code, resolution, feature)
        except KeyError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=404,
                detail=f"gold features not built for resolution={resolution}",
            ) from exc
        return {
            "region": region_code,
            "resolution": resolution,
            "feature": feature,
            "data": data,
        }

    @router.get("/object_predictions")
    def object_predictions(
        asset_class: str = Query(...),
    ) -> dict[str, Any]:
        if get_object_predictions is None:
            raise HTTPException(
                status_code=404, detail="object predictions are not configured"
            )
        try:
            ac = AssetClass(asset_class)
        except ValueError as exc:
            raise HTTPException(
                status_code=400, detail=f"unknown asset_class: {asset_class!r}"
            ) from exc
        try:
            data = get_object_predictions.execute(region_code, ac)
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=404,
                detail=f"object predictions not built for asset_class={asset_class}",
            ) from exc
        return {
            "region": region_code,
            "asset_class": ac.value,
            "data": data,
        }

    return router
