from typing import Any

from fastapi import APIRouter, HTTPException, Query

from kadastra.usecases.get_hex_features import GetHexFeatures


def make_api_router(get_hex_features: GetHexFeatures, region_code: str) -> APIRouter:
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

    return router
