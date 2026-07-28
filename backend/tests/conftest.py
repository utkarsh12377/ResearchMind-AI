"""Shared pytest fixtures: an isolated async SQLite engine per test.

Production/dev target Postgres (see app.core.config.Settings.database_url),
but tests run against SQLite (aiosqlite) for speed and zero external
dependencies. Models only use portable SQLAlchemy types (Uuid, Enum, etc.)
so behavior matches Postgres for what these tests actually exercise.
"""

from collections.abc import AsyncGenerator

import pytest_asyncio
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import StaticPool

import app.models  # noqa: F401  (registers all models on Base.metadata)
from app.db.base import Base


@pytest_asyncio.fixture
async def engine() -> AsyncGenerator[AsyncEngine, None]:
    # StaticPool keeps a single shared connection alive for the engine's
    # lifetime, which in-memory SQLite requires (each new connection would
    # otherwise see a fresh, empty database).
    test_engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
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
