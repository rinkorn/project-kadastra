"""Web routes for the kadastra inspector + map UI.

The single page at ``/`` renders the map shell + side panel. All
feature lists, asset class options, and per-object data come from
the JSON API at ``/api/...`` — there's no server-side rendering of
feature dropdowns anymore (they used to be a static snapshot of
the long-retired hex feature store, which gave the false impression
of currency).
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates


def make_web_router(templates_dir: Path) -> APIRouter:
    router = APIRouter()
    templates = Jinja2Templates(directory=str(templates_dir))

    @router.get("/", response_class=HTMLResponse)
    def index(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(request=request, name="map.html", context={})

    return router
