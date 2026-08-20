"""Shared pytest fixtures: an isolated async SQLite engine per test.

Production/dev target Postgres (see app.core.config.Settings.database_url),
but tests run against SQLite (aiosqlite) for speed and zero external
dependencies. Models only use portable SQLAlchemy types (Uuid, Enum, etc.)
so behavior matches Postgres for what these tests actually exercise.
"""

from collections.abc import AsyncGenerator

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import StaticPool

import app.models  # noqa: F401  (registers all models on Base.metadata)
from app.core.config import get_settings
from app.core.rate_limit import limiter
from app.db.base import Base
from app.db.session import get_db
from app.main import app as fastapi_app


@pytest_asyncio.fixture
async def engine() -> AsyncGenerator[AsyncEngine, None]:
    # StaticPool keeps a single shared connection alive for the engine's
    # lifetime, which in-memory SQLite requires (each new connection would
    # otherwise see a fresh, empty database).
    test_engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
        # echo=False regardless of app_debug: statement logging buries the
        # actual assertion in failure output.
        echo=False,
    )
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield test_engine

    await test_engine.dispose()


@pytest_asyncio.fixture
async def db_session(engine: AsyncEngine) -> AsyncGenerator[AsyncSession, None]:
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    async with session_factory() as session:
        yield session


@pytest.fixture(autouse=True)
def _isolated_storage(tmp_path, monkeypatch) -> None:  # noqa: ANN001
    """Point blob storage at a per-test temp directory.

    get_settings() is lru_cached, so the cache is cleared on both sides of the
    override to keep the patched path from leaking into other tests.
    """
    monkeypatch.setenv("STORAGE_LOCAL_PATH", str(tmp_path / "storage"))
    # Debug logging floods failure output with every SQL statement.
    monkeypatch.setenv("APP_DEBUG", "false")
    get_settings.cache_clear()
    yield
    monkeypatch.delenv("STORAGE_LOCAL_PATH", raising=False)
    get_settings.cache_clear()


@pytest.fixture(autouse=True)
def _reset_rate_limiter() -> None:
    # The limiter's in-memory storage is process-global; without resetting it,
    # request counts would leak between tests that hit the same endpoint.
    limiter.reset()


@pytest_asyncio.fixture
async def client(engine: AsyncEngine) -> AsyncGenerator[AsyncClient, None]:
    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    async def override_get_db() -> AsyncGenerator[AsyncSession, None]:
        async with session_factory() as session:
            yield session

    fastapi_app.dependency_overrides[get_db] = override_get_db
    transport = ASGITransport(app=fastapi_app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac

    fastapi_app.dependency_overrides.clear()
