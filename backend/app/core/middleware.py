import time

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from backend.app.core.errors import ApiError, error_response
from backend.app.core.logging import get_logger

logger = get_logger("http")

CONTENT_SECURITY_POLICY = (
    "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data: blob:; font-src 'self'; "
    "connect-src 'self'; object-src 'none'; base-uri 'self'; form-action 'self'; frame-ancestors 'none'"
)
SECURITY_HEADERS = [
    (b"x-content-type-options", b"nosniff"),
    (b"x-frame-options", b"DENY"),
    (b"referrer-policy", b"no-referrer"),
    (b"content-security-policy", CONTENT_SECURITY_POLICY.encode()),
    (b"permissions-policy", b"camera=(), microphone=(), geolocation=()"),
]


class AccessMiddleware:
    """Outermost layer: adds security headers and writes one log line per HTTP request."""

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        started = time.perf_counter()
        status = 500

        async def send_with_headers(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
                message = {**message, "headers": [*message.get("headers", []), *SECURITY_HEADERS]}
            await send(message)

        try:
            await self.app(scope, receive, send_with_headers)
        finally:
            fields = {
                "method": scope["method"],
                "path": scope["path"],
                "status": status,
                "duration_ms": round((time.perf_counter() - started) * 1000, 2),
            }
            logger.info("request", extra={"fields": fields})


class InternalErrorMiddleware:
    """Innermost user middleware, so the uniform 500 body still passes through CORS and the access layer.

    Starlette would otherwise answer unexpected exceptions from its outermost layer, bypassing both.
    """

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        response_started = False

        async def tracking_send(message: Message) -> None:
            nonlocal response_started
            response_started |= message["type"] == "http.response.start"
            await send(message)

        try:
            await self.app(scope, receive, tracking_send)
        except Exception:
            logger.exception("unhandled error", extra={"fields": {"path": scope.get("path")}})
            if response_started or scope["type"] != "http":
                raise
            await error_response(ApiError("internal_error", "Internal server error."))(scope, receive, send)
