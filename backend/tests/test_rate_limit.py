import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_login_is_rate_limited_after_five_attempts_per_minute(client: AsyncClient) -> None:
    await client.post(
        "/api/v1/auth/register",
        json={"email": "ratelimited@example.com", "password": "supersecret123"},
    )

    for _ in range(5):
        response = await client.post(
            "/api/v1/auth/login",
            data={"username": "ratelimited@example.com", "password": "wrong-password"},
        )
        assert response.status_code == 401

    response = await client.post(
        "/api/v1/auth/login",
        data={"username": "ratelimited@example.com", "password": "wrong-password"},
    )

    assert response.status_code == 429
