import pytest
from httpx import AsyncClient
from sqlalchemy import select

from app.core.metrics import (
    LATENCY_BUCKETS,
    PROMETHEUS_AVAILABLE,
    _status_class,
    record_agent_run,
    record_llm_call,
    time_stage,
)
from app.core.middleware import SECURITY_HEADERS, _content_length, _inbound_request_id
from app.models import User


async def _auth_headers(client: AsyncClient, email: str = "obs@example.com") -> dict:
    await client.post(
        "/api/v1/auth/register", json={"email": email, "password": "supersecret123"}
    )
    login = await client.post(
        "/api/v1/auth/login", data={"username": email, "password": "supersecret123"}
    )
    return {"Authorization": f"Bearer {login.json()['access_token']}"}


# --- Metrics ---------------------------------------------------------------


def test_status_codes_are_bucketed_by_class() -> None:
    """A dashboard cares that 5xx is rising, not that it was exactly 503."""
    assert _status_class(200) == "2xx"
    assert _status_class(404) == "4xx"
    assert _status_class(503) == "5xx"


def test_latency_buckets_cover_a_full_agent_run() -> None:
    """The library defaults top out at 10s, which puts every agent run in +Inf."""
    assert max(LATENCY_BUCKETS) >= 60.0


def test_recording_helpers_do_not_raise() -> None:
    record_llm_call(provider="openai", model="gpt-4", prompt_tokens=10, completion_tokens=5)
    record_llm_call(
        provider="openai", model="gpt-4", prompt_tokens=0, completion_tokens=0, ok=False
    )
    record_agent_run("planner")
    record_agent_run("planner", ok=False)

    with time_stage("dense"):
        pass


@pytest.mark.asyncio
async def test_the_scrape_endpoint_is_served(client: AsyncClient) -> None:
    response = await client.get("/metrics")

    assert response.status_code == 200 if PROMETHEUS_AVAILABLE else 503


@pytest.mark.asyncio
async def test_requests_are_counted_by_route_template(client: AsyncClient) -> None:
    if not PROMETHEUS_AVAILABLE:
        pytest.skip("prometheus_client is not installed")

    await client.get("/health")

    body = (await client.get("/metrics")).text
    assert "researchmind_http_requests_total" in body
    assert 'route="/health"' in body


@pytest.mark.asyncio
async def test_paths_with_ids_do_not_create_a_series_each(client: AsyncClient) -> None:
    """Labelling by raw URL would open one time series per paper id."""
    if not PROMETHEUS_AVAILABLE:
        pytest.skip("prometheus_client is not installed")

    headers = await _auth_headers(client)
    for _ in range(3):
        await client.get(
            "/api/v1/papers/11111111-1111-1111-1111-111111111111", headers=headers
        )

    body = (await client.get("/metrics")).text
    assert "11111111-1111-1111-1111-111111111111" not in body


# --- Request context -------------------------------------------------------


@pytest.mark.asyncio
async def test_every_response_carries_a_request_id(client: AsyncClient) -> None:
    response = await client.get("/health")

    assert response.headers.get("x-request-id")


@pytest.mark.asyncio
async def test_an_inbound_request_id_is_preserved(client: AsyncClient) -> None:
    """A trace has to survive a proxy hop to be worth anything."""
    response = await client.get("/health", headers={"X-Request-ID": "abc123"})

    assert response.headers["x-request-id"] == "abc123"


def test_a_malformed_inbound_request_id_is_ignored() -> None:
    """This value reaches log lines; it must not be able to forge structure."""
    scope = {"headers": [(b"x-request-id", b'evil" injected="yes')]}

    assert _inbound_request_id(scope) is None


def test_an_oversized_inbound_request_id_is_ignored() -> None:
    scope = {"headers": [(b"x-request-id", b"a" * 500)]}

    assert _inbound_request_id(scope) is None


def test_a_valid_inbound_request_id_is_accepted() -> None:
    scope = {"headers": [(b"x-request-id", b"7f3a-9c21")]}

    assert _inbound_request_id(scope) == "7f3a-9c21"


# --- Security headers ------------------------------------------------------


@pytest.mark.asyncio
async def test_security_headers_are_present(client: AsyncClient) -> None:
    response = await client.get("/health")

    for header in SECURITY_HEADERS:
        assert header in response.headers


@pytest.mark.asyncio
async def test_security_headers_apply_to_errors_too(client: AsyncClient) -> None:
    response = await client.get("/api/v1/papers")

    assert response.status_code == 401
    assert response.headers["x-content-type-options"] == "nosniff"


# --- Body size limit -------------------------------------------------------


def test_content_length_is_parsed() -> None:
    assert _content_length({"headers": [(b"content-length", b"512")]}) == 512


def test_a_malformed_content_length_is_ignored() -> None:
    assert _content_length({"headers": [(b"content-length", b"many")]}) is None


def test_a_missing_content_length_is_none() -> None:
    assert _content_length({"headers": []}) is None


@pytest.mark.asyncio
async def test_an_oversized_body_is_refused_before_buffering(client: AsyncClient) -> None:
    from app.core.config import get_settings

    limit = get_settings().max_request_body_bytes
    response = await client.post(
        "/api/v1/auth/login",
        content=b"x",
        headers={"content-length": str(limit + 1), "content-type": "text/plain"},
    )

    assert response.status_code == 413
    assert response.json()["error"]["type"] == "payload_too_large"


# --- Admin -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_admin_status_requires_authentication(client: AsyncClient) -> None:
    assert (await client.get("/api/v1/admin/status")).status_code == 401


@pytest.mark.asyncio
async def test_admin_status_requires_a_superuser(client: AsyncClient) -> None:
    headers = await _auth_headers(client, "plain@example.com")

    assert (await client.get("/api/v1/admin/status", headers=headers)).status_code == 403


@pytest.mark.asyncio
async def test_reindex_requires_a_superuser(client: AsyncClient) -> None:
    headers = await _auth_headers(client, "plain2@example.com")

    response = await client.post("/api/v1/admin/reindex-sparse", headers=headers)

    assert response.status_code == 403


@pytest.mark.asyncio
async def test_a_superuser_sees_component_and_corpus_state(
    client: AsyncClient, engine  # noqa: ANN001
) -> None:
    from sqlalchemy.ext.asyncio import async_sessionmaker

    headers = await _auth_headers(client, "root@example.com")

    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    async with session_factory() as session:
        user = await session.scalar(select(User).where(User.email == "root@example.com"))
        user.is_superuser = True
        await session.commit()

    response = await client.get("/api/v1/admin/status", headers=headers)

    assert response.status_code == 200
    body = response.json()
    assert body["components"]["embedding_provider"]
    assert body["corpus"]["users"] >= 1
    assert "sparse_documents" in body["indexes"]
