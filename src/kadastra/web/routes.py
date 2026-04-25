from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

DEFAULT_FEATURES = [
    "building_count",
    "building_count_apartments",
    "building_count_detached",
    "levels_mean",
    "flats_total",
    "dist_metro_m",
    "dist_entrance_m",
    "count_stations_1km",
    "count_entrances_500m",
    "road_length_m",
]


def make_web_router(templates_dir: Path) -> APIRouter:
    router = APIRouter()
    templates = Jinja2Templates(directory=str(templates_dir))

    @router.get("/", response_class=HTMLResponse)
    def index(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(
            request=request,
            name="map.html",
            context={"features": DEFAULT_FEATURES},
        )

    return router
