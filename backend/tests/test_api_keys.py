import pytest
from httpx import AsyncClient


async def _register_and_login(client: AsyncClient) -> str:
    await client.post(
        "/api/v1/auth/register",
        json={"email": "keys@example.com", "password": "supersecret123"},
    )
    login_response = await client.post(
        "/api/v1/auth/login",
        data={"username": "keys@example.com", "password": "supersecret123"},
    )
    return login_response.json()["access_token"]


@pytest.mark.asyncio
async def test_create_api_key_returns_raw_key_once(client: AsyncClient) -> None:
    token = await _register_and_login(client)

    response = await client.post(
        "/api/v1/auth/api-keys",
        json={"name": "CI pipeline"},
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 201, response.text
    body = response.json()
    assert body["api_key"].startswith("rma_")
    assert body["prefix"] in body["api_key"]


@pytest.mark.asyncio
async def test_api_key_authenticates_requests(client: AsyncClient) -> None:
    token = await _register_and_login(client)
    create_response = await client.post(
        "/api/v1/auth/api-keys",
        json={"name": "CI pipeline"},
        headers={"Authorization": f"Bearer {token}"},
    )
    raw_key = create_response.json()["api_key"]

    response = await client.get("/api/v1/auth/me", headers={"X-API-Key": raw_key})

    assert response.status_code == 200
    assert response.json()["email"] == "keys@example.com"


@pytest.mark.asyncio
async def test_listing_api_keys_never_exposes_raw_key(client: AsyncClient) -> None:
    token = await _register_and_login(client)
    await client.post(
        "/api/v1/auth/api-keys",
        json={"name": "CI pipeline"},
        headers={"Authorization": f"Bearer {token}"},
    )

    response = await client.get(
        "/api/v1/auth/api-keys", headers={"Authorization": f"Bearer {token}"}
    )

    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    assert "api_key" not in body[0]
    assert "hashed_key" not in body[0]


@pytest.mark.asyncio
async def test_revoked_api_key_can_no_longer_authenticate(client: AsyncClient) -> None:
    token = await _register_and_login(client)
    create_response = await client.post(
        "/api/v1/auth/api-keys",
        json={"name": "CI pipeline"},
        headers={"Authorization": f"Bearer {token}"},
    )
    raw_key = create_response.json()["api_key"]
    key_id = create_response.json()["id"]

    revoke_response = await client.delete(
        f"/api/v1/auth/api-keys/{key_id}", headers={"Authorization": f"Bearer {token}"}
    )
    assert revoke_response.status_code == 204

    response = await client.get("/api/v1/auth/me", headers={"X-API-Key": raw_key})
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_revoking_unknown_key_is_not_found(client: AsyncClient) -> None:
    token = await _register_and_login(client)

    response = await client.delete(
        "/api/v1/auth/api-keys/00000000-0000-0000-0000-000000000000",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 404
