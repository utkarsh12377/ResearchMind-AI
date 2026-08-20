"""FastAPI application entrypoint."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from app.api.v1.router import api_router
from app.core.config import get_settings
from app.core.exceptions import register_exception_handlers
from app.core.logging import configure_logging, get_logger
from app.core.rate_limit import limiter

configure_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ANN201
    """Warm the in-memory sparse index before serving traffic.

    Dense vectors are persisted by the vector store, but BM25 term statistics
    are rebuilt from Postgres on every start.
    """
    from app.db.session import async_session_factory
    from app.retrieval.service import rebuild_sparse_index

    try:
        async with async_session_factory() as session:
            await rebuild_sparse_index(session)
    except Exception as exc:  # noqa: BLE001 - never block startup on a warm cache
        logger.warning("sparse_index_warmup_failed", error=str(exc))

    yield


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="ResearchMind AI",
        description="AI-powered research assistant: ingestion, retrieval, agents, knowledge graph.",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.backend_cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    register_exception_handlers(app)
    app.include_router(api_router, prefix=settings.api_v1_prefix)

    @app.get("/health", tags=["system"])
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    logger.info("app_initialized", env=settings.app_env)
    return app


app = create_app()
