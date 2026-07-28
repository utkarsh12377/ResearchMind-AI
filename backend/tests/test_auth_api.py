import pytest
from httpx import AsyncClient


async def _register(client: AsyncClient, email: str = "researcher@example.com") -> None:
    response = await client.post(
        "/api/v1/auth/register",
        json={"email": email, "password": "supersecret123", "full_name": "Ada Researcher"},
    )
    assert response.status_code == 201, response.text
    return response


@pytest.mark.asyncio
async def test_register_creates_user(client: AsyncClient) -> None:
    response = await _register(client)

    body = response.json()
    assert body["email"] == "researcher@example.com"
    assert "hashed_password" not in body
    assert "id" in body


@pytest.mark.asyncio
async def test_register_duplicate_email_is_conflict(client: AsyncClient) -> None:
    await _register(client)

    response = await client.post(
        "/api/v1/auth/register",
        json={"email": "researcher@example.com", "password": "anotherpassword"},
    )

    assert response.status_code == 409
    assert response.json()["error"]["type"] == "conflict"


@pytest.mark.asyncio
async def test_login_with_correct_credentials_returns_token(client: AsyncClient) -> None:
    await _register(client)

    response = await client.post(
        "/api/v1/auth/login",
        data={"username": "researcher@example.com", "password": "supersecret123"},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["token_type"] == "bearer"
    assert body["access_token"]


@pytest.mark.asyncio
async def test_login_with_wrong_password_is_unauthorized(client: AsyncClient) -> None:
    await _register(client)

    response = await client.post(
        "/api/v1/auth/login",
        data={"username": "researcher@example.com", "password": "wrong-password"},
    )

    assert response.status_code == 401
    assert response.json()["error"]["type"] == "unauthorized"


@pytest.mark.asyncio
async def test_me_requires_authentication(client: AsyncClient) -> None:
    response = await client.get("/api/v1/auth/me")

    assert response.status_code == 401


@pytest.mark.asyncio
async def test_me_returns_current_user_with_bearer_token(client: AsyncClient) -> None:
    await _register(client)
    login_response = await client.post(
        "/api/v1/auth/login",
        data={"username": "researcher@example.com", "password": "supersecret123"},
    )
    token = login_response.json()["access_token"]

    response = await client.get(
        "/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"}
    )

    assert response.status_code == 200
    assert response.json()["email"] == "researcher@example.com"
