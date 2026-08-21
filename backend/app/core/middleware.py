"""Cross-cutting HTTP middleware: request correlation and security headers."""

from __future__ import annotations

import time
import uuid

import structlog
from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

REQUEST_ID_HEADER = "x-request-id"

#: Applied to every response. The API serves JSON to a separate origin, so the
#: CSP only has to cover the case where a browser is pointed at an endpoint
#: directly -- there is no first-party HTML to break.
SECURITY_HEADERS = {
    "x-content-type-options": "nosniff",
    "x-frame-options": "DENY",
    "referrer-policy": "no-referrer",
    "cross-origin-opener-policy": "same-origin",
    "content-security-policy": "default-src 'none'; frame-ancestors 'none'; sandbox",
    "permissions-policy": "geolocation=(), microphone=(), camera=()",
}


class RequestContextMiddleware:
    """Give every request an id, bind it to the logger, and echo it back.

    Without this, a structured log line cannot be tied to the request that
    produced it, which is the first thing anyone wants when a user reports a
    failure. An inbound X-Request-ID is honoured so a trace survives a proxy hop.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = _inbound_request_id(scope) or uuid.uuid4().hex
        started = time.perf_counter()

        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(
            request_id=request_id,
            method=scope.get("method", ""),
            path=scope.get("path", ""),
        )

        status_code = 500

        async def send_with_request_id(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
                MutableHeaders(scope=message).append(REQUEST_ID_HEADER, request_id)
            await send(message)

        try:
            await self.app(scope, receive, send_with_request_id)
        finally:
            logger.info(
                "request_completed",
                status=status_code,
                duration_ms=int((time.perf_counter() - started) * 1000),
            )
            structlog.contextvars.clear_contextvars()


def _inbound_request_id(scope: Scope) -> str | None:
    for key, value in scope.get("headers", []):
        if key.decode("latin-1").lower() == REQUEST_ID_HEADER:
            candidate = value.decode("latin-1").strip()
            # Bounded and alphanumeric: this value ends up in log lines, and an
            # attacker-supplied one should not be able to forge log structure.
            if 0 < len(candidate) <= 128 and candidate.replace("-", "").isalnum():
                return candidate
    return None


class SecurityHeadersMiddleware:
    """Add hardening headers to every response."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_with_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                for name, value in SECURITY_HEADERS.items():
                    if name not in headers:
                        headers.append(name, value)
            await send(message)

        await self.app(scope, receive, send_with_headers)


class BodySizeLimitMiddleware:
    """Reject oversized bodies before they are buffered.

    Upload size is already validated in the papers endpoint, but that check runs
    after the body has been read. This one refuses on the declared
    Content-Length, so a hostile client cannot make the process hold a gigabyte
    in memory on the way to a 413.
    """

    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        declared = _content_length(scope)
        if declared is not None and declared > self.max_bytes:
            logger.warning("request_body_too_large", declared=declared, limit=self.max_bytes)
            await _reject(send, 413, "Request body too large")
            return

        await self.app(scope, receive, send)


def _content_length(scope: Scope) -> int | None:
    for key, value in scope.get("headers", []):
        if key.decode("latin-1").lower() == "content-length":
            try:
                return int(value)
            except ValueError:
                return None
    return None


async def _reject(send: Send, status: int, message: str) -> None:
    body = f'{{"error":{{"type":"payload_too_large","message":"{message}"}}}}'.encode()
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})


def install(app) -> None:  # noqa: ANN001
    """Attach the middleware stack.

    Order matters and is the reverse of execution: the request-context
    middleware is added last so it runs first, and every log line emitted
    downstream -- including one from a rejected oversized body -- carries a
    request id.
    """
    settings = get_settings()

    app.add_middleware(BodySizeLimitMiddleware, max_bytes=settings.max_request_body_bytes)
    if settings.security_headers_enabled:
        app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(RequestContextMiddleware)
