import uuid

import pytest
from httpx import AsyncClient

ROUTES = (
    "/api/v1/insights/comparison",
    "/api/v1/insights/matrix",
    "/api/v1/insights/trends",
    "/api/v1/insights/consistency",
    "/api/v1/insights/gaps",
)


async def _auth_headers(client: AsyncClient, email: str = "insights@example.com") -> dict:
    await client.post(
        "/api/v1/auth/register", json={"email": email, "password": "supersecret123"}
    )
    login = await client.post(
        "/api/v1/auth/login", data={"username": email, "password": "supersecret123"}
    )
    return {"Authorization": f"Bearer {login.json()['access_token']}"}


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ROUTES)
async def test_every_route_requires_authentication(client: AsyncClient, route: str) -> None:
    assert (await client.post(route, json={})).status_code == 401


@pytest.mark.asyncio
async def test_review_requires_authentication(client: AsyncClient) -> None:
    response = await client.post("/api/v1/insights/review", json={"topic": "retrieval"})

    assert response.status_code == 401


@pytest.mark.asyncio
async def test_comparison_on_an_empty_library(client: AsyncClient) -> None:
    headers = await _auth_headers(client)

    response = await client.post("/api/v1/insights/comparison", json={}, headers=headers)

    assert response.status_code == 200
    body = response.json()
    assert body["rows"] == []
    assert "No comparable results" in body["markdown"]


@pytest.mark.asyncio
async def test_matrix_on_an_empty_library(client: AsyncClient) -> None:
    headers = await _auth_headers(client)

    response = await client.post("/api/v1/insights/matrix", json={}, headers=headers)

    assert response.status_code == 200
    assert response.json()["rows"] == []


@pytest.mark.asyncio
async def test_trends_on_an_empty_library(client: AsyncClient) -> None:
    headers = await _auth_headers(client)

    response = await client.post("/api/v1/insights/trends", json={}, headers=headers)

    assert response.status_code == 200
    body = response.json()
    assert body["timeline"] == []
    assert body["narrative"] == ""


@pytest.mark.asyncio
async def test_consistency_on_an_empty_library(client: AsyncClient) -> None:
    headers = await _auth_headers(client)

    response = await client.post(
        "/api/v1/insights/consistency", json={"include_claims": False}, headers=headers
    )

    assert response.status_code == 200
    assert response.json()["is_consistent"] is True


@pytest.mark.asyncio
async def test_gaps_on_an_empty_library(client: AsyncClient) -> None:
    headers = await _auth_headers(client)

    response = await client.post("/api/v1/insights/gaps", json={}, headers=headers)

    assert response.status_code == 200
    assert response.json()["structural"] == []


@pytest.mark.asyncio
async def test_another_users_paper_id_is_silently_dropped(client: AsyncClient) -> None:
    """Scoping is a convenience; the accessible set is the actual boundary."""
    headers = await _auth_headers(client)

    response = await client.post(
        "/api/v1/insights/comparison",
        json={"paper_ids": [str(uuid.uuid4())]},
        headers=headers,
    )

    assert response.status_code == 200
    assert response.json()["rows"] == []


@pytest.mark.asyncio
async def test_too_many_paper_ids_is_rejected(client: AsyncClient) -> None:
    headers = await _auth_headers(client)

    response = await client.post(
        "/api/v1/insights/comparison",
        json={"paper_ids": [str(uuid.uuid4()) for _ in range(80)]},
        headers=headers,
    )

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_review_requires_a_topic(client: AsyncClient) -> None:
    headers = await _auth_headers(client)

    response = await client.post("/api/v1/insights/review", json={}, headers=headers)

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_review_section_count_is_bounded(client: AsyncClient) -> None:
    headers = await _auth_headers(client)

    response = await client.post(
        "/api/v1/insights/review",
        json={"topic": "retrieval", "max_sections": 40},
        headers=headers,
    )

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_review_over_an_empty_library_returns_no_sections(client: AsyncClient) -> None:
    headers = await _auth_headers(client)

    response = await client.post(
        "/api/v1/insights/review", json={"topic": "dense retrieval"}, headers=headers
    )

    assert response.status_code == 200
    body = response.json()
    assert body["sections"] == []
    assert body["sources"] == []
