"""Web routes for the kadastra inspector + map UI.

The single page at ``/`` renders the map shell + side panel. All
feature lists, asset class options, and per-object data come from
the JSON API at ``/api/...`` — there's no server-side rendering of
feature dropdowns anymore (they used to be a static snapshot of
the long-retired hex feature store, which gave the false impression
of currency).

When ``Settings.auth_token`` is set, a single shared token gates
everything except ``/health``. ``/login`` accepts the token via a
form POST and stores it as an ``auth_token`` cookie that subsequent
``fetch('/api/...')`` calls inherit automatically.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates


def make_web_router(templates_dir: Path) -> APIRouter:
    router = APIRouter()
    templates = Jinja2Templates(directory=str(templates_dir))

    @router.get("/", response_class=HTMLResponse)
    def index(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(request=request, name="map.html", context={})

    @router.get("/login", response_class=HTMLResponse)
    def login_page(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(request=request, name="login.html", context={})

    @router.post("/login")
    def login_submit(token: str = Form(...)) -> RedirectResponse:
        # The middleware validates the token on the next request — we
        # just persist whatever was submitted. 30-day cookie life so a
        # logged-in pilot user doesn't have to re-auth daily.
        response = RedirectResponse("/", status_code=302)
        response.set_cookie(
            "auth_token",
            token,
            httponly=True,
            samesite="strict",
            max_age=86400 * 30,
        )
        return response

    @router.get("/logout")
    def logout() -> RedirectResponse:
        response = RedirectResponse("/login", status_code=302)
        response.delete_cookie("auth_token")
        return response

    return router
