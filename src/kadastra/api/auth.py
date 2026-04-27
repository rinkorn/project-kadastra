"""Bearer token authentication middleware.

When ``Settings.auth_token`` is set, every request goes through this
middleware. /api/* requests must carry ``Authorization: Bearer <token>``
or an ``auth_token`` cookie; browser requests check the cookie alone
and redirect to ``/login`` when missing.

When ``auth_token`` is unset (None), the middleware is not registered
at all — useful for local dev. Production stages always have a token
in their .env.
"""

from __future__ import annotations

from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import RedirectResponse, Response
from starlette.types import ASGIApp

# /health: docker + uptime monitors hit this without credentials.
# /login, /logout: auth endpoints themselves.
# /favicon.ico: browsers fetch it before the user has a chance to log in.
_PUBLIC_PREFIXES: tuple[str, ...] = ("/health", "/login", "/logout", "/favicon.ico")


class BearerAuthMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp, *, token: str) -> None:
        super().__init__(app)
        self._token = token

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        path = request.url.path
        if any(path.startswith(p) for p in _PUBLIC_PREFIXES):
            return await call_next(request)
        # /api/* — JSON clients use Bearer header; the browser SPA also
        # accepts the cookie so a logged-in user's fetch() calls succeed
        # without any client-side token plumbing.
        if path.startswith("/api/"):
            auth = request.headers.get("Authorization", "")
            if auth == f"Bearer {self._token}":
                return await call_next(request)
            if request.cookies.get("auth_token") == self._token:
                return await call_next(request)
            return JSONResponse({"detail": "Invalid or missing token"}, status_code=401)
        # Browser pages — cookie only; redirect to the login form.
        if request.cookies.get("auth_token") != self._token:
            return RedirectResponse("/login", status_code=302)
        return await call_next(request)
