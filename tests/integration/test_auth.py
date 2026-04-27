"""BearerAuthMiddleware end-to-end behaviour.

Mirrors test_app.py's fixture pattern but builds the app with
``auth_token`` set, so every request goes through the middleware.
Covers:
- public-prefix bypass (/health, /login, /logout, /favicon.ico)
- /api/* accepts both Bearer header and cookie, rejects with 401
- browser pages accept cookie, redirect to /login otherwise
- auth_token=None disables the middleware entirely
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from kadastra.composition_root import create_app
from kadastra.config import Settings

_TOKEN = "test-secret"


def _settings(tmp_path: Path, *, token: str | None) -> Settings:
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
        auth_token=token,
    )


@pytest.fixture
def client(tmp_path: Path) -> Iterator[TestClient]:
    app = create_app(_settings(tmp_path, token=_TOKEN))
    # follow_redirects=False: we want to assert on the 302 itself.
    with TestClient(app, follow_redirects=False) as c:
        yield c


@pytest.fixture
def open_client(tmp_path: Path) -> Iterator[TestClient]:
    """No auth_token → middleware not registered at all."""
    app = create_app(_settings(tmp_path, token=None))
    with TestClient(app, follow_redirects=False) as c:
        yield c


# ---------- public-prefix bypass ----------


def test_health_is_public(client: TestClient) -> None:
    """Docker/uptime probes hit /health without credentials. Must
    always return 200 — regardless of auth_token configuration."""
    assert client.get("/health").status_code == 200


def test_login_get_is_public(client: TestClient) -> None:
    """The login form itself must be reachable without auth, otherwise
    the user can never enter a token in the first place."""
    assert client.get("/login").status_code == 200


def test_logout_is_public(client: TestClient) -> None:
    """Logout is a public 302 → /login + cookie deletion. Treating it
    as protected would brick the «forgot my session» path."""
    response = client.get("/logout")
    assert response.status_code == 302
    assert response.headers["location"] == "/login"


def test_favicon_is_public(client: TestClient) -> None:
    """Browsers fetch /favicon.ico before the user can authenticate.
    A 302 here would surface a console error on every page load."""
    # 404 is fine (no file shipped) — what matters is *no* redirect to /login.
    response = client.get("/favicon.ico")
    assert response.status_code != 302


# ---------- /api/* protection ----------


def test_api_without_credentials_returns_401(client: TestClient) -> None:
    response = client.get("/api/feature_options")
    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid or missing token"}


def test_api_with_valid_bearer_passes(client: TestClient) -> None:
    response = client.get(
        "/api/feature_options",
        headers={"Authorization": f"Bearer {_TOKEN}"},
    )
    assert response.status_code == 200


def test_api_with_wrong_bearer_returns_401(client: TestClient) -> None:
    response = client.get(
        "/api/feature_options",
        headers={"Authorization": "Bearer wrong-token"},
    )
    assert response.status_code == 401


def test_api_with_valid_cookie_passes(client: TestClient) -> None:
    """The login form sets an ``auth_token`` cookie; the SPA's fetch()
    calls inherit it automatically. Must work without a Bearer header."""
    client.cookies.set("auth_token", _TOKEN)
    response = client.get("/api/feature_options")
    assert response.status_code == 200


def test_api_with_wrong_cookie_returns_401(client: TestClient) -> None:
    client.cookies.set("auth_token", "wrong-token")
    response = client.get("/api/feature_options")
    assert response.status_code == 401


# ---------- browser pages ----------


def test_index_without_cookie_redirects_to_login(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 302
    assert response.headers["location"] == "/login"


def test_index_with_valid_cookie_serves_html(client: TestClient) -> None:
    client.cookies.set("auth_token", _TOKEN)
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_index_with_wrong_cookie_redirects_to_login(client: TestClient) -> None:
    client.cookies.set("auth_token", "wrong-token")
    response = client.get("/")
    assert response.status_code == 302
    assert response.headers["location"] == "/login"


def test_index_does_not_accept_bearer_header(client: TestClient) -> None:
    """/api/* takes Bearer; browser pages take cookie only. Mixing
    must not let a Bearer header bypass the cookie check on /, since
    browser navigations cannot send Authorization headers anyway."""
    response = client.get("/", headers={"Authorization": f"Bearer {_TOKEN}"})
    assert response.status_code == 302
    assert response.headers["location"] == "/login"


# ---------- /login POST ----------


def test_login_post_sets_cookie_and_redirects(client: TestClient) -> None:
    """The form posts the token; the response sets an httponly cookie
    and 302s back to /. The cookie value is whatever the form sent —
    the middleware validates on the *next* request."""
    response = client.post("/login", data={"token": _TOKEN})
    assert response.status_code == 302
    assert response.headers["location"] == "/"
    set_cookie = response.headers.get("set-cookie", "")
    assert "auth_token=" in set_cookie
    assert "HttpOnly" in set_cookie
    assert "samesite=strict" in set_cookie.lower()


# ---------- middleware disabled when auth_token=None ----------


def test_disabled_auth_lets_api_through(open_client: TestClient) -> None:
    """When ``auth_token`` is unset, BearerAuthMiddleware is not
    registered at all (local dev default). Every endpoint behaves as
    if there were no auth layer."""
    assert open_client.get("/api/feature_options").status_code == 200


def test_disabled_auth_lets_index_through(open_client: TestClient) -> None:
    response = open_client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
