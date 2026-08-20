import httpx
import pytest

from app.retrieval.embeddings import (
    EmbeddingError,
    HashEmbeddingProvider,
    JinaEmbeddingProvider,
    OpenAIEmbeddingProvider,
    l2_normalize,
)


def test_l2_normalize_produces_unit_vectors() -> None:
    normalized = l2_normalize([3.0, 4.0])

    assert pytest.approx(sum(v * v for v in normalized) ** 0.5) == 1.0


def test_l2_normalize_leaves_the_zero_vector_alone() -> None:
    assert l2_normalize([0.0, 0.0]) == [0.0, 0.0]


@pytest.mark.asyncio
async def test_hash_provider_is_deterministic() -> None:
    provider = HashEmbeddingProvider(dimensions=64)

    first = await provider.embed_documents(["graph retrieval"])
    second = await provider.embed_documents(["graph retrieval"])

    assert first == second


@pytest.mark.asyncio
async def test_hash_provider_returns_unit_vectors_of_the_right_size() -> None:
    provider = HashEmbeddingProvider(dimensions=128)

    [vector] = await provider.embed_documents(["retrieval augmented generation"])

    assert len(vector) == 128
    assert pytest.approx(sum(v * v for v in vector) ** 0.5) == 1.0


@pytest.mark.asyncio
async def test_identical_text_has_similarity_one() -> None:
    provider = HashEmbeddingProvider(dimensions=256)
    text = "dense passage retrieval for open domain question answering"

    [document] = await provider.embed_documents([text])
    query = await provider.embed_query(text)

    assert pytest.approx(sum(a * b for a, b in zip(query, document, strict=True))) == 1.0


@pytest.mark.asyncio
async def test_shared_vocabulary_scores_higher_than_unrelated_text() -> None:
    provider = HashEmbeddingProvider(dimensions=512)

    vectors = await provider.embed_documents(
        [
            "dense passage retrieval for question answering",
            "dense passage retrieval improves question answering",
            "convolutional networks for image classification",
        ]
    )

    def dot(a, b):  # noqa: ANN001, ANN202
        return sum(x * y for x, y in zip(a, b, strict=True))

    assert dot(vectors[0], vectors[1]) > dot(vectors[0], vectors[2])


@pytest.mark.asyncio
async def test_empty_input_returns_no_vectors() -> None:
    provider = OpenAIEmbeddingProvider("sk-test")

    assert await provider.embed_documents([]) == []


def test_http_providers_require_an_api_key() -> None:
    with pytest.raises(EmbeddingError, match="requires an API key"):
        OpenAIEmbeddingProvider("")


@pytest.mark.asyncio
async def test_openai_provider_parses_and_normalizes_responses(monkeypatch) -> None:  # noqa: ANN001
    captured: dict[str, object] = {}

    async def fake_post(self, url, headers=None, json=None):  # noqa: ANN001, ANN202
        captured["url"] = url
        captured["json"] = json
        # Returned out of order to prove results are sorted by index.
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": 1, "embedding": [0.0, 3.0, 4.0]},
                    {"index": 0, "embedding": [3.0, 4.0, 0.0]},
                ]
            },
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
    provider = OpenAIEmbeddingProvider("sk-test", dimensions=3)

    vectors = await provider.embed_documents(["first", "second"])

    assert captured["json"]["input"] == ["first", "second"]
    assert vectors[0] == pytest.approx([0.6, 0.8, 0.0])
    assert vectors[1] == pytest.approx([0.0, 0.6, 0.8])


@pytest.mark.asyncio
async def test_client_errors_are_not_retried(monkeypatch) -> None:  # noqa: ANN001
    attempts = {"count": 0}

    async def fake_post(self, url, headers=None, json=None):  # noqa: ANN001, ANN202
        attempts["count"] += 1
        return httpx.Response(401, text="invalid key", request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

    with pytest.raises(EmbeddingError, match="401"):
        await OpenAIEmbeddingProvider("sk-bad").embed_documents(["text"])

    # A bad key will never succeed, so retrying only wastes time.
    assert attempts["count"] == 1


@pytest.mark.asyncio
async def test_rate_limits_are_retried_then_surface(monkeypatch) -> None:  # noqa: ANN001
    attempts = {"count": 0}

    async def fake_post(self, url, headers=None, json=None):  # noqa: ANN001, ANN202
        attempts["count"] += 1
        return httpx.Response(429, text="slow down", request=httpx.Request("POST", url))

    async def no_sleep(_delay):  # noqa: ANN001, ANN202
        return None

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
    monkeypatch.setattr("app.retrieval.embeddings.asyncio.sleep", no_sleep)

    with pytest.raises(EmbeddingError, match="after 3 attempts"):
        await OpenAIEmbeddingProvider("sk-test").embed_documents(["text"])

    assert attempts["count"] == 3


@pytest.mark.asyncio
async def test_transient_failure_then_success(monkeypatch) -> None:  # noqa: ANN001
    attempts = {"count": 0}

    async def fake_post(self, url, headers=None, json=None):  # noqa: ANN001, ANN202
        attempts["count"] += 1
        if attempts["count"] == 1:
            return httpx.Response(503, text="unavailable", request=httpx.Request("POST", url))
        return httpx.Response(
            200,
            json={"data": [{"index": 0, "embedding": [1.0, 0.0]}]},
            request=httpx.Request("POST", url),
        )

    async def no_sleep(_delay):  # noqa: ANN001, ANN202
        return None

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
    monkeypatch.setattr("app.retrieval.embeddings.asyncio.sleep", no_sleep)

    vectors = await OpenAIEmbeddingProvider("sk-test", dimensions=2).embed_documents(["text"])

    assert vectors == [[1.0, 0.0]]
    assert attempts["count"] == 2


@pytest.mark.asyncio
async def test_jina_uses_the_query_task_for_queries(monkeypatch) -> None:  # noqa: ANN001
    seen: list[str] = []

    async def fake_post(self, url, headers=None, json=None):  # noqa: ANN001, ANN202
        seen.append(json["task"])
        return httpx.Response(
            200,
            json={"data": [{"index": 0, "embedding": [1.0, 0.0]}]},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
    provider = JinaEmbeddingProvider("jina-key", dimensions=2)

    await provider.embed_documents(["a passage"])
    await provider.embed_query("a query")

    assert seen == ["retrieval.passage", "retrieval.query"]
