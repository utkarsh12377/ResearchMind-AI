"""Prometheus instrumentation.

The registry is built by hand rather than by an auto-instrumentation library so
the label sets stay deliberate. Cardinality is the failure mode that kills a
Prometheus install, and a middleware that labels by raw URL path will happily
create one series per paper id. Routes are labelled by their template, and
nothing here is labelled by anything user-supplied.

prometheus_client is optional: when it is missing every helper becomes a no-op
so the application still starts.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from contextlib import contextmanager

from starlette.requests import Request
from starlette.responses import PlainTextResponse, Response
from starlette.types import ASGIApp

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

try:
    from prometheus_client import (
        CONTENT_TYPE_LATEST,
        CollectorRegistry,
        Counter,
        Gauge,
        Histogram,
        generate_latest,
    )

    PROMETHEUS_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without the extra
    PROMETHEUS_AVAILABLE = False
    CONTENT_TYPE_LATEST = "text/plain"

#: Buckets chosen for this workload rather than the library default. Retrieval
#: answers land in tens of milliseconds; a full agent run takes tens of seconds.
#: The default buckets top out at 10s and would put every agent run in +Inf.
LATENCY_BUCKETS = (0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0)
TOKEN_BUCKETS = (100, 500, 1000, 2500, 5000, 10000, 25000, 50000, 100000)


class _NoOp:
    """Stand-in used when prometheus_client is not installed."""

    def labels(self, *args, **kwargs) -> _NoOp:  # noqa: ANN002, ANN003
        return self

    def inc(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
        return None

    def observe(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
        return None

    def set(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
        return None


if PROMETHEUS_AVAILABLE:
    REGISTRY = CollectorRegistry()

    http_requests_total = Counter(
        "researchmind_http_requests_total",
        "HTTP requests by route template, method, and status class.",
        ("method", "route", "status"),
        registry=REGISTRY,
    )
    http_request_duration = Histogram(
        "researchmind_http_request_duration_seconds",
        "HTTP request latency by route template.",
        ("method", "route"),
        buckets=LATENCY_BUCKETS,
        registry=REGISTRY,
    )
    llm_requests_total = Counter(
        "researchmind_llm_requests_total",
        "LLM completions by provider, model, and outcome.",
        ("provider", "model", "outcome"),
        registry=REGISTRY,
    )
    llm_tokens_total = Counter(
        "researchmind_llm_tokens_total",
        "Tokens consumed by provider and direction.",
        ("provider", "direction"),
        registry=REGISTRY,
    )
    llm_tokens_per_request = Histogram(
        "researchmind_llm_tokens_per_request",
        "Total tokens per completion.",
        ("provider",),
        buckets=TOKEN_BUCKETS,
        registry=REGISTRY,
    )
    retrieval_duration = Histogram(
        "researchmind_retrieval_duration_seconds",
        "Retrieval pipeline latency by stage.",
        ("stage",),
        buckets=LATENCY_BUCKETS,
        registry=REGISTRY,
    )
    agent_runs_total = Counter(
        "researchmind_agent_runs_total",
        "Agent executions by agent name and outcome.",
        ("agent", "outcome"),
        registry=REGISTRY,
    )
    ingestion_total = Counter(
        "researchmind_ingestion_total",
        "Papers processed by terminal status.",
        ("status",),
        registry=REGISTRY,
    )
    indexed_chunks = Gauge(
        "researchmind_indexed_chunks",
        "Chunks currently present in the sparse index.",
        registry=REGISTRY,
    )
else:  # pragma: no cover - exercised only without the extra
    REGISTRY = None
    http_requests_total = _NoOp()
    http_request_duration = _NoOp()
    llm_requests_total = _NoOp()
    llm_tokens_total = _NoOp()
    llm_tokens_per_request = _NoOp()
    retrieval_duration = _NoOp()
    agent_runs_total = _NoOp()
    ingestion_total = _NoOp()
    indexed_chunks = _NoOp()


def route_template(request: Request) -> str:
    """The route's path template, never the concrete URL.

    /papers/{paper_id} stays one series. Labelling by request.url.path would
    create a new one for every paper ever fetched.
    """
    route = request.scope.get("route")
    template = getattr(route, "path", None)
    return template or "unmatched"


class MetricsMiddleware:
    """Times every request and records it against its route template."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope, receive, send) -> None:  # noqa: ANN001
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        started = time.perf_counter()
        status_holder = {"code": 500}

        async def wrapped_send(message) -> None:  # noqa: ANN001
            if message["type"] == "http.response.start":
                status_holder["code"] = message["status"]
            await send(message)

        try:
            await self.app(scope, receive, wrapped_send)
        finally:
            request = Request(scope)
            route = route_template(request)
            method = scope.get("method", "GET")
            elapsed = time.perf_counter() - started

            http_request_duration.labels(method=method, route=route).observe(elapsed)
            http_requests_total.labels(
                method=method, route=route, status=_status_class(status_holder["code"])
            ).inc()


def _status_class(status_code: int) -> str:
    """Bucket by class rather than exact code.

    A dashboard cares that 5xx is rising, not that it was specifically a 503,
    and exact codes multiply series for no operational gain.
    """
    return f"{status_code // 100}xx"


def record_llm_call(
    *, provider: str, model: str, prompt_tokens: int, completion_tokens: int, ok: bool = True
) -> None:
    outcome = "success" if ok else "error"
    llm_requests_total.labels(provider=provider, model=model, outcome=outcome).inc()
    llm_tokens_total.labels(provider=provider, direction="prompt").inc(prompt_tokens)
    llm_tokens_total.labels(provider=provider, direction="completion").inc(completion_tokens)
    llm_tokens_per_request.labels(provider=provider).observe(prompt_tokens + completion_tokens)


def record_agent_run(agent: str, *, ok: bool = True) -> None:
    agent_runs_total.labels(agent=agent, outcome="success" if ok else "error").inc()


def record_ingestion(status: str) -> None:
    ingestion_total.labels(status=status).inc()


def set_indexed_chunks(count: int) -> None:
    indexed_chunks.set(count)


@contextmanager
def time_stage(stage: str):  # noqa: ANN201
    started = time.perf_counter()
    try:
        yield
    finally:
        retrieval_duration.labels(stage=stage).observe(time.perf_counter() - started)


def render_metrics() -> Response:
    if not PROMETHEUS_AVAILABLE:
        return PlainTextResponse(
            "prometheus_client is not installed; metrics are disabled.\n",
            status_code=503,
        )
    return Response(content=generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)


def install(app) -> None:  # noqa: ANN001
    """Attach the middleware and the scrape endpoint, if metrics are enabled."""
    settings = get_settings()
    if not settings.metrics_enabled:
        logger.info("metrics_disabled")
        return

    app.add_middleware(MetricsMiddleware)

    @app.get(settings.metrics_path, include_in_schema=False)
    async def metrics() -> Response:
        return render_metrics()

    logger.info(
        "metrics_enabled",
        path=settings.metrics_path,
        backend="prometheus_client" if PROMETHEUS_AVAILABLE else "noop",
    )


def observe(stage: str) -> Callable:
    """Decorator form of time_stage for async functions."""

    def decorator(func: Callable) -> Callable:
        async def wrapper(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
            with time_stage(stage):
                return await func(*args, **kwargs)

        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper

    return decorator
