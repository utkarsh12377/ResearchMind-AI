import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.core.exceptions import ConflictError, NotFoundError, register_exception_handlers


def _build_test_app() -> FastAPI:
    app = FastAPI()
    register_exception_handlers(app)

    @app.get("/not-found")
    async def raise_not_found() -> None:
        raise NotFoundError("Paper not found")

    @app.get("/conflict")
    async def raise_conflict() -> None:
        raise ConflictError("Email already registered")

    @app.get("/boom")
    async def raise_unhandled() -> None:
        raise RuntimeError("something broke")

    return app


@pytest.mark.asyncio
async def test_app_error_returns_structured_json_and_status_code() -> None:
    transport = ASGITransport(app=_build_test_app())
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/not-found")

    assert response.status_code == 404
    assert response.json() == {
        "error": {"type": "not_found", "message": "Paper not found"}
    }


@pytest.mark.asyncio
async def test_conflict_error_maps_to_409() -> None:
    transport = ASGITransport(app=_build_test_app())
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/conflict")

    assert response.status_code == 409
    assert response.json()["error"]["type"] == "conflict"


@pytest.mark.asyncio
async def test_unhandled_exception_returns_generic_500() -> None:
    transport = ASGITransport(app=_build_test_app(), raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/boom")

    assert response.status_code == 500
    assert response.json() == {
        "error": {"type": "internal_error", "message": "An unexpected error occurred."}
    }
