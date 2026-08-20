"""Pluggable embedding providers.

Every provider is reached through one interface so the embedding model can be
swapped without touching ingestion or retrieval. Providers are called over HTTP
with a shared client rather than through per-vendor SDKs: it gives one retry and
timeout story, keeps async semantics consistent, and avoids three dependencies
that disagree about connection pooling.

Vectors are L2-normalized on the way out, so cosine similarity reduces to a dot
product and every vector store backend can use the same metric.
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import struct
from abc import ABC, abstractmethod

import httpx

from app.core.logging import get_logger

logger = get_logger(__name__)

DEFAULT_TIMEOUT = 60.0
MAX_RETRIES = 3
RETRY_BASE_DELAY = 0.5


class EmbeddingError(RuntimeError):
    """Raised when embeddings cannot be produced."""


def l2_normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        return vector
    return [value / norm for value in vector]


class EmbeddingProvider(ABC):
    """Turns text into dense vectors."""

    model_name: str
    dimensions: int

    @abstractmethod
    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed passages for indexing."""

    async def embed_query(self, text: str) -> list[float]:
        """Embed a search query.

        Defaults to the document path; providers with asymmetric query/document
        encoders (BGE, for instance) override this to add their query prefix.
        """
        vectors = await self.embed_documents([text])
        return vectors[0]


class HashEmbeddingProvider(EmbeddingProvider):
    """Deterministic local embeddings with no network dependency.

    Not semantically meaningful — it hashes token n-grams into a fixed number of
    buckets. It exists so the whole retrieval stack (indexing, hybrid fusion,
    reranking, the API) can be developed and tested end-to-end without API keys
    or network access, and so tests stay fast and deterministic. Never select
    this in production; retrieval quality would be no better than lexical
    overlap.
    """

    model_name = "hash-local"

    def __init__(self, dimensions: int = 384) -> None:
        self.dimensions = dimensions

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        tokens = text.lower().split()

        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            bucket = struct.unpack_from(">I", digest)[0] % self.dimensions
            # Sign from a second digest byte keeps unrelated tokens from all
            # pushing the vector in the same direction.
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[bucket] += sign

        return l2_normalize(vector)

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]


class _HttpEmbeddingProvider(EmbeddingProvider):
    """Shared retry/transport behavior for HTTP-based providers."""

    endpoint: str

    def __init__(self, api_key: str, *, timeout: float = DEFAULT_TIMEOUT) -> None:
        if not api_key:
            raise EmbeddingError(
                f"{type(self).__name__} requires an API key; set it in .env or "
                "switch EMBEDDING_PROVIDER to 'hash' for local development."
            )
        self.api_key = api_key
        self.timeout = timeout

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def _payload(self, texts: list[str]) -> dict[str, object]:
        raise NotImplementedError

    def _parse(self, payload: dict) -> list[list[float]]:
        raise NotImplementedError

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        last_error: Exception | None = None
        for attempt in range(MAX_RETRIES):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(
                        self.endpoint, headers=self._headers(), json=self._payload(texts)
                    )

                # 429 and 5xx are transient; 4xx otherwise means a bad request
                # or key, which retrying will never fix.
                if response.status_code == 429 or response.status_code >= 500:
                    raise httpx.HTTPStatusError(
                        f"Retryable status {response.status_code}",
                        request=response.request,
                        response=response,
                    )
                response.raise_for_status()
                return [l2_normalize(vector) for vector in self._parse(response.json())]

            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                if status != 429 and status < 500:
                    raise EmbeddingError(
                        f"Embedding request failed ({status}): {exc.response.text[:300]}"
                    ) from exc
                last_error = exc
            except (httpx.TransportError, httpx.TimeoutException) as exc:
                last_error = exc

            if attempt < MAX_RETRIES - 1:
                delay = RETRY_BASE_DELAY * (2**attempt)
                logger.warning(
                    "embedding_retry", attempt=attempt + 1, delay=delay, error=str(last_error)
                )
                await asyncio.sleep(delay)

        raise EmbeddingError(f"Embedding request failed after {MAX_RETRIES} attempts: {last_error}")


class OpenAIEmbeddingProvider(_HttpEmbeddingProvider):
    endpoint = "https://api.openai.com/v1/embeddings"

    def __init__(
        self,
        api_key: str,
        *,
        model: str = "text-embedding-3-small",
        dimensions: int = 1536,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        super().__init__(api_key, timeout=timeout)
        self.model_name = model
        self.dimensions = dimensions

    def _payload(self, texts: list[str]) -> dict[str, object]:
        return {"model": self.model_name, "input": texts, "dimensions": self.dimensions}

    def _parse(self, payload: dict) -> list[list[float]]:
        # The API may return items out of order, so sort by the echoed index.
        items = sorted(payload["data"], key=lambda item: item["index"])
        return [item["embedding"] for item in items]


class JinaEmbeddingProvider(_HttpEmbeddingProvider):
    endpoint = "https://api.jina.ai/v1/embeddings"

    def __init__(
        self,
        api_key: str,
        *,
        model: str = "jina-embeddings-v3",
        dimensions: int = 1024,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        super().__init__(api_key, timeout=timeout)
        self.model_name = model
        self.dimensions = dimensions

    def _payload(self, texts: list[str]) -> dict[str, object]:
        return {"model": self.model_name, "input": texts, "task": "retrieval.passage"}

    def _parse(self, payload: dict) -> list[list[float]]:
        items = sorted(payload["data"], key=lambda item: item["index"])
        return [item["embedding"] for item in items]

    async def embed_query(self, text: str) -> list[float]:
        # Jina v3 is asymmetric: queries must be encoded with the query task or
        # similarity against passages degrades noticeably.
        original = self._payload
        try:
            self._payload = lambda texts: {  # type: ignore[method-assign]
                "model": self.model_name,
                "input": texts,
                "task": "retrieval.query",
            }
            vectors = await self.embed_documents([text])
        finally:
            self._payload = original  # type: ignore[method-assign]
        return vectors[0]


class BGEEmbeddingProvider(EmbeddingProvider):
    """Local BGE embeddings via sentence-transformers.

    Imported lazily: torch and sentence-transformers are heavy, and most
    deployments pick a hosted provider instead.
    """

    def __init__(self, model: str = "BAAI/bge-small-en-v1.5", dimensions: int = 384) -> None:
        self.model_name = model
        self.dimensions = dimensions
        self._model = None

    def _load(self):  # noqa: ANN202 - third-party type only available when installed
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise EmbeddingError(
                    "BGE embeddings require sentence-transformers. Install it, or set "
                    "EMBEDDING_PROVIDER to 'openai', 'jina', or 'hash'."
                ) from exc
            self._model = SentenceTransformer(self.model_name)
        return self._model

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        model = self._load()
        # sentence-transformers is synchronous and CPU/GPU bound; run it off the
        # event loop so the API stays responsive.
        vectors = await asyncio.to_thread(model.encode, texts, normalize_embeddings=True)
        return [list(map(float, vector)) for vector in vectors]

    async def embed_query(self, text: str) -> list[float]:
        # BGE expects this instruction prefix on queries only.
        vectors = await self.embed_documents(
            [f"Represent this sentence for searching relevant passages: {text}"]
        )
        return vectors[0]


def get_embedding_provider() -> EmbeddingProvider:
    from app.core.config import get_settings

    settings = get_settings()
    provider = settings.embedding_provider.lower()

    if provider == "openai":
        return OpenAIEmbeddingProvider(
            settings.openai_api_key,
            model=settings.embedding_model,
            dimensions=settings.embedding_dimensions,
        )
    if provider == "jina":
        return JinaEmbeddingProvider(
            settings.jina_api_key,
            model=settings.embedding_model,
            dimensions=settings.embedding_dimensions,
        )
    if provider == "bge":
        return BGEEmbeddingProvider(
            model=settings.embedding_model, dimensions=settings.embedding_dimensions
        )
    if provider == "hash":
        return HashEmbeddingProvider(dimensions=settings.embedding_dimensions)

    raise EmbeddingError(
        f"Unknown embedding provider {settings.embedding_provider!r}; "
        "expected one of: openai, jina, bge, hash"
    )
