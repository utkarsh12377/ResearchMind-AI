import pytest
from httpx import AsyncClient

from tests.factories import build_pdf


@pytest.fixture(autouse=True)
def _no_celery_dispatch(monkeypatch) -> list[str]:  # noqa: ANN001
    """Capture task dispatches instead of requiring a live broker.

    The task body itself is covered by tests/test_ingestion_task.py; here we
    only care that the endpoint enqueues exactly one job for the new paper.
    """
    dispatched: list[str] = []
    monkeypatch.setattr(
        "app.api.v1.endpoints.papers.process_paper.delay",
        lambda paper_id: dispatched.append(paper_id),
    )
    return dispatched


async def _auth_headers(client: AsyncClient, email: str = "papers@example.com") -> dict[str, str]:
    await client.post(
        "/api/v1/auth/register", json={"email": email, "password": "supersecret123"}
    )
    login = await client.post(
        "/api/v1/auth/login", data={"username": email, "password": "supersecret123"}
    )
    return {"Authorization": f"Bearer {login.json()['access_token']}"}


def _pdf_upload(name: str = "paper.pdf") -> dict[str, tuple[str, object, str]]:
    return {"file": (name, build_pdf(), "application/pdf")}


@pytest.mark.asyncio
async def test_upload_stores_paper_and_enqueues_processing(
    client: AsyncClient, _no_celery_dispatch: list[str]
) -> None:
    headers = await _auth_headers(client)

    response = await client.post("/api/v1/papers", files=_pdf_upload(), headers=headers)

    assert response.status_code == 201, response.text
    body = response.json()
    assert body["status"] == "pending"
    assert body["original_filename"] == "paper.pdf"
    assert body["size_bytes"] > 0
    assert len(body["checksum"]) == 64
    assert _no_celery_dispatch == [body["id"]]


@pytest.mark.asyncio
async def test_upload_requires_authentication(client: AsyncClient) -> None:
    response = await client.post("/api/v1/papers", files=_pdf_upload())

    assert response.status_code == 401


@pytest.mark.asyncio
async def test_non_pdf_upload_is_rejected(client: AsyncClient) -> None:
    headers = await _auth_headers(client)

    response = await client.post(
        "/api/v1/papers",
        files={"file": ("notes.txt", b"just text", "text/plain")},
        headers=headers,
    )

    assert response.status_code == 422
    assert response.json()["error"]["type"] == "validation_error"


@pytest.mark.asyncio
async def test_empty_upload_is_rejected(client: AsyncClient) -> None:
    headers = await _auth_headers(client)

    response = await client.post(
        "/api/v1/papers",
        files={"file": ("empty.pdf", b"", "application/pdf")},
        headers=headers,
    )

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_uploading_the_same_file_twice_is_a_conflict(client: AsyncClient) -> None:
    headers = await _auth_headers(client)
    # Reuse the exact same bytes: PDF generation embeds timestamps, so calling
    # build_pdf() twice would produce different checksums and not collide.
    pdf_bytes = build_pdf().getvalue()

    first = await client.post(
        "/api/v1/papers",
        files={"file": ("paper.pdf", pdf_bytes, "application/pdf")},
        headers=headers,
    )
    assert first.status_code == 201

    response = await client.post(
        "/api/v1/papers",
        files={"file": ("paper.pdf", pdf_bytes, "application/pdf")},
        headers=headers,
    )

    assert response.status_code == 409
    assert response.json()["error"]["type"] == "conflict"


@pytest.mark.asyncio
async def test_list_returns_only_the_callers_papers(client: AsyncClient) -> None:
    owner_headers = await _auth_headers(client, "owner@example.com")
    await client.post("/api/v1/papers", files=_pdf_upload(), headers=owner_headers)

    other_headers = await _auth_headers(client, "other@example.com")
    response = await client.get("/api/v1/papers", headers=other_headers)

    assert response.status_code == 200
    assert response.json() == {"items": [], "total": 0}


@pytest.mark.asyncio
async def test_get_paper_by_id(client: AsyncClient) -> None:
    headers = await _auth_headers(client)
    created = await client.post("/api/v1/papers", files=_pdf_upload(), headers=headers)
    paper_id = created.json()["id"]

    response = await client.get(f"/api/v1/papers/{paper_id}", headers=headers)

    assert response.status_code == 200
    assert response.json()["id"] == paper_id


@pytest.mark.asyncio
async def test_another_users_paper_is_not_found(client: AsyncClient) -> None:
    owner_headers = await _auth_headers(client, "owner2@example.com")
    created = await client.post("/api/v1/papers", files=_pdf_upload(), headers=owner_headers)
    paper_id = created.json()["id"]

    other_headers = await _auth_headers(client, "intruder@example.com")
    response = await client.get(f"/api/v1/papers/{paper_id}", headers=other_headers)

    # 404 rather than 403, so the API doesn't confirm the paper exists.
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_delete_removes_the_paper(client: AsyncClient) -> None:
    headers = await _auth_headers(client)
    created = await client.post("/api/v1/papers", files=_pdf_upload(), headers=headers)
    paper_id = created.json()["id"]

    delete_response = await client.delete(f"/api/v1/papers/{paper_id}", headers=headers)
    assert delete_response.status_code == 204

    assert (await client.get(f"/api/v1/papers/{paper_id}", headers=headers)).status_code == 404


@pytest.mark.asyncio
async def test_assets_endpoint_requires_ownership(client: AsyncClient) -> None:
    owner_headers = await _auth_headers(client, "assetowner@example.com")
    created = await client.post("/api/v1/papers", files=_pdf_upload(), headers=owner_headers)
    paper_id = created.json()["id"]

    intruder_headers = await _auth_headers(client, "assetintruder@example.com")
    response = await client.get(f"/api/v1/papers/{paper_id}/assets", headers=intruder_headers)

    assert response.status_code == 404


@pytest.mark.asyncio
async def test_assets_endpoint_returns_empty_list_before_processing(client: AsyncClient) -> None:
    headers = await _auth_headers(client, "assets@example.com")
    created = await client.post("/api/v1/papers", files=_pdf_upload(), headers=headers)
    paper_id = created.json()["id"]

    response = await client.get(f"/api/v1/papers/{paper_id}/assets", headers=headers)

    assert response.status_code == 200
    assert response.json() == []
