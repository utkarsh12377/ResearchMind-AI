import pytest
from httpx import AsyncClient

from app.graph.store import reset_graph_store


@pytest.fixture(autouse=True)
def _fresh_graph_store() -> None:
    """The store is process-global, so it must not leak between tests."""
    reset_graph_store()
    yield
    reset_graph_store()


async def _auth_headers(client: AsyncClient, email: str = "graph@example.com") -> dict:
    await client.post(
        "/api/v1/auth/register", json={"email": email, "password": "supersecret123"}
    )
    login = await client.post(
        "/api/v1/auth/login", data={"username": email, "password": "supersecret123"}
    )
    return {"Authorization": f"Bearer {login.json()['access_token']}"}


@pytest.mark.asyncio
async def test_overview_requires_authentication(client: AsyncClient) -> None:
    assert (await client.get("/api/v1/graph/overview")).status_code == 401


@pytest.mark.asyncio
async def test_network_requires_authentication(client: AsyncClient) -> None:
    assert (await client.get("/api/v1/graph/network")).status_code == 401


@pytest.mark.asyncio
async def test_query_requires_authentication(client: AsyncClient) -> None:
    response = await client.post("/api/v1/graph/query", json={"question": "who cites what"})

    assert response.status_code == 401


@pytest.mark.asyncio
async def test_overview_on_an_empty_corpus(client: AsyncClient) -> None:
    headers = await _auth_headers(client)

    response = await client.get("/api/v1/graph/overview", headers=headers)

    assert response.status_code == 200
    body = response.json()
    assert body["papers"] == 0
    assert body["top_entities"] == []


@pytest.mark.asyncio
async def test_network_defaults_to_the_entity_view(client: AsyncClient) -> None:
    headers = await _auth_headers(client)

    response = await client.get("/api/v1/graph/network", headers=headers)

    assert response.status_code == 200
    assert response.json()["view"] == "entities"


@pytest.mark.asyncio
@pytest.mark.parametrize("view", ["entities", "citations", "authors"])
async def test_every_declared_view_is_served(client: AsyncClient, view: str) -> None:
    headers = await _auth_headers(client)

    response = await client.get(f"/api/v1/graph/network?view={view}", headers=headers)

    assert response.status_code == 200
    assert response.json()["view"] == view


@pytest.mark.asyncio
async def test_an_unknown_view_is_rejected(client: AsyncClient) -> None:
    headers = await _auth_headers(client)

    response = await client.get("/api/v1/graph/network?view=telepathy", headers=headers)

    assert response.status_code == 422
    assert "Unknown view" in response.json()["error"]["message"]


@pytest.mark.asyncio
async def test_an_unknown_label_filter_is_rejected(client: AsyncClient) -> None:
    headers = await _auth_headers(client)

    response = await client.get("/api/v1/graph/network?labels=Sandwich", headers=headers)

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_neighbors_rejects_an_unknown_label(client: AsyncClient) -> None:
    headers = await _auth_headers(client)

    response = await client.get(
        "/api/v1/graph/neighbors?label=Sandwich&key=blt", headers=headers
    )

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_neighbors_of_a_missing_node_is_empty(client: AsyncClient) -> None:
    headers = await _auth_headers(client)

    response = await client.get(
        "/api/v1/graph/neighbors?label=Dataset&key=nothing-here", headers=headers
    )

    assert response.status_code == 200
    assert response.json()["neighbors"] == []


@pytest.mark.asyncio
async def test_neighbor_depth_is_bounded(client: AsyncClient) -> None:
    headers = await _auth_headers(client)

    response = await client.get(
        "/api/v1/graph/neighbors?label=Dataset&key=squad&depth=99", headers=headers
    )

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_query_returns_the_generated_cypher_even_on_failure(client: AsyncClient) -> None:
    """A graph answer nobody can inspect is not one worth trusting."""
    headers = await _auth_headers(client)

    response = await client.post(
        "/api/v1/graph/query", json={"question": "which papers cite each other"}, headers=headers
    )

    assert response.status_code == 200
    body = response.json()
    assert "cypher" in body
    assert body["rows"] == []
    assert body["error"]


@pytest.mark.asyncio
async def test_query_rejects_an_empty_question(client: AsyncClient) -> None:
    headers = await _auth_headers(client)

    response = await client.post("/api/v1/graph/query", json={"question": ""}, headers=headers)

    assert response.status_code == 422
