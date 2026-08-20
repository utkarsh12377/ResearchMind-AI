import pytest
from httpx import AsyncClient

from tests.factories import build_pdf


@pytest.fixture(autouse=True)
def _no_celery_dispatch(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setattr(
        "app.api.v1.endpoints.papers.process_paper.delay", lambda paper_id: None
    )


async def _auth_headers(client: AsyncClient, email: str = "search@example.com") -> dict:
    await client.post(
        "/api/v1/auth/register", json={"email": email, "password": "supersecret123"}
    )
    login = await client.post(
        "/api/v1/auth/login", data={"username": email, "password": "supersecret123"}
    )
    return {"Authorization": f"Bearer {login.json()['access_token']}"}


@pytest.mark.asyncio
async def test_search_requires_authentication(client: AsyncClient) -> None:
    response = await client.post("/api/v1/search", json={"query": "retrieval"})

    assert response.status_code == 401


@pytest.mark.asyncio
async def test_empty_query_is_rejected_by_validation(client: AsyncClient) -> None:
    headers = await _auth_headers(client)

    response = await client.post("/api/v1/search", json={"query": ""}, headers=headers)

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_search_with_no_indexed_papers_returns_empty(client: AsyncClient) -> None:
    headers = await _auth_headers(client)

    response = await client.post(
        "/api/v1/search", json={"query": "retrieval augmented generation"}, headers=headers
    )

    assert response.status_code == 200
    body = response.json()
    assert body["results"] == []
    assert body["total"] == 0
    assert body["query"] == "retrieval augmented generation"


@pytest.mark.asyncio
async def test_limit_is_validated(client: AsyncClient) -> None:
    headers = await _auth_headers(client)

    response = await client.post(
        "/api/v1/search", json={"query": "retrieval", "limit": 500}, headers=headers
    )

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_filters_are_accepted(client: AsyncClient) -> None:
    headers = await _auth_headers(client)
    await client.post("/api/v1/papers", files={"file": ("p.pdf", build_pdf(), "application/pdf")},
                      headers=headers)

    response = await client.post(
        "/api/v1/search",
        json={
            "query": "retrieval",
            "filters": {"kinds": ["table"], "sections": ["Results"], "year_from": 2020},
        },
        headers=headers,
    )

    assert response.status_code == 200
