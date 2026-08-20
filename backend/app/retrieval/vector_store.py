"""Vector store abstraction with in-memory/FAISS and Qdrant backends.

Retrieval code depends only on `VectorStore`, so the development backend and
the production one are interchangeable. Stored payloads carry just the
identifiers and filterable metadata — chunk text stays in Postgres, which keeps
the index small and means re-indexing never risks stale duplicated text.
"""

from __future__ import annotations

import math
import pickle
import threading
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class VectorRecord:
    """A vector plus the metadata needed to filter and resolve it."""

    chunk_id: uuid.UUID
    paper_id: uuid.UUID
    workspace_id: uuid.UUID
    vector: list[float]
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass
class SearchHit:
    chunk_id: uuid.UUID
    paper_id: uuid.UUID
    score: float
    metadata: dict[str, object] = field(default_factory=dict)


def _matches(metadata: dict[str, object], filters: dict[str, object] | None) -> bool:
    """Apply equality/membership filters to a record's metadata.

    A list filter value means "any of", which covers the common cases (a set of
    paper ids, a set of years) without needing a query language here; the richer
    filter DSL lives one layer up in the retrieval service.
    """
    if not filters:
        return True

    for key, expected in filters.items():
        actual = metadata.get(key)
        if isinstance(expected, list | tuple | set):
            if actual not in expected:
                return False
        elif actual != expected:
            return False
    return True


class VectorStore(ABC):
    @abstractmethod
    async def upsert(self, records: list[VectorRecord]) -> None:
        """Insert or replace vectors by chunk id."""

    @abstractmethod
    async def search(
        self,
        vector: list[float],
        *,
        limit: int = 10,
        filters: dict[str, object] | None = None,
    ) -> list[SearchHit]: ...

    @abstractmethod
    async def delete_by_paper(self, paper_id: uuid.UUID) -> int:
        """Remove every vector belonging to a paper. Returns the count removed."""

    @abstractmethod
    async def count(self) -> int: ...


class InMemoryVectorStore(VectorStore):
    """Exact brute-force search, optionally persisted to disk.

    Exact rather than approximate: for development corpora the difference is
    imperceptible, and exact results make retrieval tests deterministic. Vectors
    are assumed L2-normalized (the embedding layer guarantees this), so the dot
    product is cosine similarity.
    """

    def __init__(self, persist_path: str | Path | None = None) -> None:
        self._records: dict[uuid.UUID, VectorRecord] = {}
        self._lock = threading.Lock()
        self._persist_path = Path(persist_path) if persist_path else None
        if self._persist_path and self._persist_path.is_file():
            self._load()

    def _load(self) -> None:
        try:
            with self._persist_path.open("rb") as handle:
                self._records = pickle.load(handle)
            logger.info("vector_store_loaded", count=len(self._records))
        except Exception as exc:  # noqa: BLE001 - a corrupt cache must not block startup
            logger.warning("vector_store_load_failed", error=str(exc))
            self._records = {}

    def _persist(self) -> None:
        if not self._persist_path:
            return
        self._persist_path.parent.mkdir(parents=True, exist_ok=True)
        # Write-then-rename so a crash mid-write can't truncate the index.
        temporary = self._persist_path.with_suffix(".tmp")
        with temporary.open("wb") as handle:
            pickle.dump(self._records, handle)
        temporary.replace(self._persist_path)

    async def upsert(self, records: list[VectorRecord]) -> None:
        with self._lock:
            for record in records:
                self._records[record.chunk_id] = record
            self._persist()

    async def search(
        self,
        vector: list[float],
        *,
        limit: int = 10,
        filters: dict[str, object] | None = None,
    ) -> list[SearchHit]:
        with self._lock:
            candidates = list(self._records.values())

        hits = [
            SearchHit(
                chunk_id=record.chunk_id,
                paper_id=record.paper_id,
                score=sum(a * b for a, b in zip(vector, record.vector, strict=True)),
                metadata=record.metadata,
            )
            for record in candidates
            if _matches(record.metadata, filters)
        ]
        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits[:limit]

    async def delete_by_paper(self, paper_id: uuid.UUID) -> int:
        with self._lock:
            doomed = [key for key, rec in self._records.items() if rec.paper_id == paper_id]
            for key in doomed:
                del self._records[key]
            self._persist()
        return len(doomed)

    async def count(self) -> int:
        with self._lock:
            return len(self._records)


class FaissVectorStore(InMemoryVectorStore):
    """FAISS-backed search for larger development corpora.

    Subclasses the in-memory store so the record bookkeeping (payloads,
    filtering, persistence) is shared; only the similarity scan is replaced.
    FAISS has no notion of payload filters, so filtered queries over-fetch and
    post-filter, which is correct if occasionally wasteful.
    """

    def __init__(self, dimensions: int, persist_path: str | Path | None = None) -> None:
        super().__init__(persist_path=persist_path)
        self.dimensions = dimensions
        self._index = None
        self._ids: list[uuid.UUID] = []

    def _require_faiss(self):  # noqa: ANN202 - third-party type
        try:
            import faiss
        except ImportError as exc:
            raise RuntimeError(
                "FAISS backend requires faiss-cpu. Install it, or set "
                "VECTOR_STORE_BACKEND to 'memory' or 'qdrant'."
            ) from exc
        return faiss

    def _rebuild(self) -> None:
        faiss = self._require_faiss()
        import numpy as np

        self._ids = list(self._records)
        self._index = faiss.IndexFlatIP(self.dimensions)
        if self._ids:
            matrix = np.array(
                [self._records[key].vector for key in self._ids], dtype="float32"
            )
            self._index.add(matrix)

    async def upsert(self, records: list[VectorRecord]) -> None:
        await super().upsert(records)
        with self._lock:
            self._rebuild()

    async def search(
        self,
        vector: list[float],
        *,
        limit: int = 10,
        filters: dict[str, object] | None = None,
    ) -> list[SearchHit]:
        import numpy as np

        with self._lock:
            if self._index is None:
                self._rebuild()
            if not self._ids:
                return []

            # Over-fetch when filtering, since FAISS can't apply payload filters.
            fetch = min(len(self._ids), limit * 10 if filters else limit)
            scores, positions = self._index.search(
                np.array([vector], dtype="float32"), fetch
            )
            records = [(self._records[self._ids[pos]], float(score))
                       for pos, score in zip(positions[0], scores[0], strict=True)
                       if pos >= 0]

        return [
            SearchHit(
                chunk_id=record.chunk_id,
                paper_id=record.paper_id,
                score=score,
                metadata=record.metadata,
            )
            for record, score in records
            if _matches(record.metadata, filters)
        ][:limit]

    async def delete_by_paper(self, paper_id: uuid.UUID) -> int:
        removed = await super().delete_by_paper(paper_id)
        with self._lock:
            self._rebuild()
        return removed


class QdrantVectorStore(VectorStore):
    """Production backend.

    Filtering and deletion are pushed into Qdrant rather than done client-side,
    so recall is unaffected by how selective a filter is.
    """

    def __init__(
        self,
        url: str,
        collection: str,
        dimensions: int,
        api_key: str | None = None,
    ) -> None:
        self.url = url
        self.collection = collection
        self.dimensions = dimensions
        self.api_key = api_key or None
        self._client = None

    def _get_client(self):  # noqa: ANN202 - third-party type
        if self._client is None:
            try:
                from qdrant_client import AsyncQdrantClient
            except ImportError as exc:
                raise RuntimeError(
                    "Qdrant backend requires qdrant-client. Install it, or set "
                    "VECTOR_STORE_BACKEND to 'memory'."
                ) from exc
            self._client = AsyncQdrantClient(url=self.url, api_key=self.api_key)
        return self._client

    async def ensure_collection(self) -> None:
        from qdrant_client.models import Distance, VectorParams

        client = self._get_client()
        existing = {c.name for c in (await client.get_collections()).collections}
        if self.collection in existing:
            return

        await client.create_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(size=self.dimensions, distance=Distance.COSINE),
        )
        # Payload indexes make filtered search fast; without them Qdrant scans.
        for field_name in ("paper_id", "workspace_id"):
            await client.create_payload_index(
                collection_name=self.collection,
                field_name=field_name,
                field_schema="keyword",
            )
        logger.info("qdrant_collection_created", collection=self.collection)

    async def upsert(self, records: list[VectorRecord]) -> None:
        if not records:
            return
        from qdrant_client.models import PointStruct

        await self.ensure_collection()
        client = self._get_client()
        await client.upsert(
            collection_name=self.collection,
            points=[
                PointStruct(
                    id=str(record.chunk_id),
                    vector=record.vector,
                    payload={
                        "chunk_id": str(record.chunk_id),
                        "paper_id": str(record.paper_id),
                        "workspace_id": str(record.workspace_id),
                        **record.metadata,
                    },
                )
                for record in records
            ],
        )

    def _build_filter(self, filters: dict[str, object] | None):  # noqa: ANN202
        if not filters:
            return None
        from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue

        conditions = []
        for key, expected in filters.items():
            if isinstance(expected, list | tuple | set):
                match = MatchAny(any=[str(value) for value in expected])
            else:
                match = MatchValue(value=expected)
            conditions.append(FieldCondition(key=key, match=match))
        return Filter(must=conditions)

    async def search(
        self,
        vector: list[float],
        *,
        limit: int = 10,
        filters: dict[str, object] | None = None,
    ) -> list[SearchHit]:
        await self.ensure_collection()
        client = self._get_client()
        results = await client.query_points(
            collection_name=self.collection,
            query=vector,
            limit=limit,
            query_filter=self._build_filter(filters),
            with_payload=True,
        )

        return [
            SearchHit(
                chunk_id=uuid.UUID(point.payload["chunk_id"]),
                paper_id=uuid.UUID(point.payload["paper_id"]),
                score=point.score,
                metadata=point.payload,
            )
            for point in results.points
        ]

    async def delete_by_paper(self, paper_id: uuid.UUID) -> int:
        from qdrant_client.models import FilterSelector

        await self.ensure_collection()
        client = self._get_client()
        await client.delete(
            collection_name=self.collection,
            points_selector=FilterSelector(
                filter=self._build_filter({"paper_id": str(paper_id)})
            ),
        )
        # Qdrant's delete response doesn't report a row count.
        return 0

    async def count(self) -> int:
        await self.ensure_collection()
        client = self._get_client()
        return (await client.count(collection_name=self.collection)).count


_store: VectorStore | None = None


def get_vector_store() -> VectorStore:
    """Return the process-wide vector store.

    Cached because the in-memory and FAISS backends *are* the index — building a
    new one per request would discard it.
    """
    global _store
    if _store is not None:
        return _store

    from app.core.config import get_settings

    settings = get_settings()
    backend = settings.vector_store_backend.lower()

    if backend == "qdrant":
        _store = QdrantVectorStore(
            url=settings.qdrant_url,
            collection=settings.qdrant_collection,
            dimensions=settings.embedding_dimensions,
            api_key=settings.qdrant_api_key,
        )
    elif backend == "faiss":
        _store = FaissVectorStore(
            dimensions=settings.embedding_dimensions,
            persist_path=Path(settings.storage_local_path) / "vectors" / "index.pkl",
        )
    elif backend == "memory":
        _store = InMemoryVectorStore()
    else:
        raise ValueError(
            f"Unknown vector store backend {settings.vector_store_backend!r}; "
            "expected one of: memory, faiss, qdrant"
        )

    logger.info("vector_store_initialized", backend=backend)
    return _store


def reset_vector_store() -> None:
    """Drop the cached store. Used by tests to isolate indexes."""
    global _store
    _store = None


def cosine_similarity(a: list[float], b: list[float]) -> float:
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return sum(x * y for x, y in zip(a, b, strict=True)) / (norm_a * norm_b)
