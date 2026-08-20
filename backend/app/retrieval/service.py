"""The retrieval pipeline: filter, search densely and sparsely, fuse, rerank.

This is the single entry point every consumer uses — the chat endpoint, the
agents, and the evaluation harness — so retrieval behavior stays consistent and
improvements land everywhere at once.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.models import DocumentChunk, Paper, Workspace
from app.retrieval.embeddings import EmbeddingProvider, get_embedding_provider
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.reranker import (
    RerankCandidate,
    Reranker,
    extract_supporting_sentences,
    get_reranker,
)
from app.retrieval.sparse import BM25Index, get_bm25_index
from app.retrieval.vector_store import VectorStore, get_vector_store

logger = get_logger(__name__)

# Retrieve more candidates than requested so the reranker has room to reorder;
# reranking a list of exactly k can only shuffle, never recover a missed hit.
CANDIDATE_MULTIPLIER = 4
MAX_CANDIDATES = 200


@dataclass
class RetrievalFilters:
    """Structured filters applied before scoring.

    Kept as an explicit object rather than a free-form dict so the API surface
    is typed and the same filters can be pushed down to both retrievers.
    """

    paper_ids: list[uuid.UUID] = field(default_factory=list)
    kinds: list[str] = field(default_factory=list)
    year_from: int | None = None
    year_to: int | None = None
    sections: list[str] = field(default_factory=list)

    def to_metadata_filters(self) -> dict[str, object]:
        """Filters the vector/BM25 stores can apply against chunk metadata."""
        filters: dict[str, object] = {}
        if self.paper_ids:
            filters["paper_id"] = [str(pid) for pid in self.paper_ids]
        if self.kinds:
            filters["kind"] = list(self.kinds)
        return filters

    @property
    def needs_database_filtering(self) -> bool:
        """Year and section live on the paper/section, not in chunk metadata."""
        return bool(self.year_from or self.year_to or self.sections)


@dataclass
class RetrievedChunk:
    chunk_id: uuid.UUID
    paper_id: uuid.UUID
    paper_title: str | None
    content: str
    section_path: str | None
    page_number: int | None
    kind: str
    score: float
    dense_rank: int | None = None
    sparse_rank: int | None = None
    rerank_score: float | None = None
    supporting_sentences: list[str] = field(default_factory=list)

    @property
    def citation(self) -> str:
        """Human-readable source label for inline citation."""
        parts = [self.paper_title or "Untitled paper"]
        if self.section_path:
            parts.append(self.section_path)
        if self.page_number:
            parts.append(f"p. {self.page_number}")
        return " — ".join(parts)


async def _resolve_workspace_ids(db: AsyncSession, user) -> list[uuid.UUID]:  # noqa: ANN001
    result = await db.scalars(select(Workspace.id).where(Workspace.owner_id == user.id))
    return list(result)


async def _allowed_paper_ids(
    db: AsyncSession, user, filters: RetrievalFilters  # noqa: ANN001
) -> list[uuid.UUID]:
    """Papers the caller may see, narrowed by any database-level filters.

    Authorization is resolved here rather than after scoring: filtering results
    the user was never allowed to see would leak their existence through
    result counts.
    """
    workspace_ids = await _resolve_workspace_ids(db, user)
    if not workspace_ids:
        return []

    query = select(Paper.id).where(Paper.workspace_id.in_(workspace_ids))
    if filters.paper_ids:
        query = query.where(Paper.id.in_(filters.paper_ids))

    return list(await db.scalars(query))


async def retrieve(
    db: AsyncSession,
    user,  # noqa: ANN001
    query: str,
    *,
    limit: int = 10,
    filters: RetrievalFilters | None = None,
    use_reranker: bool = True,
    provider: EmbeddingProvider | None = None,
    store: VectorStore | None = None,
    bm25: BM25Index | None = None,
    reranker: Reranker | None = None,
) -> list[RetrievedChunk]:
    """Run the full hybrid retrieval pipeline for one query."""
    if not query.strip():
        return []

    filters = filters or RetrievalFilters()
    provider = provider or get_embedding_provider()
    store = store or get_vector_store()
    bm25 = bm25 or get_bm25_index()

    allowed = await _allowed_paper_ids(db, user, filters)
    if not allowed:
        return []

    metadata_filters = filters.to_metadata_filters()
    # Always constrain to papers the caller can access, even when no explicit
    # paper filter was requested.
    metadata_filters["paper_id"] = [str(pid) for pid in allowed]

    candidate_count = min(MAX_CANDIDATES, max(limit * CANDIDATE_MULTIPLIER, limit))

    query_vector = await provider.embed_query(query)
    # The two retrievers are independent; run them concurrently.
    dense_hits, sparse_hits = await asyncio.gather(
        store.search(query_vector, limit=candidate_count, filters=metadata_filters),
        asyncio.to_thread(
            bm25.search, query, limit=candidate_count, filters=metadata_filters
        ),
    )

    fused = reciprocal_rank_fusion(dense_hits, sparse_hits, limit=candidate_count)
    if not fused:
        return []

    chunks = await _load_chunks(db, [hit.chunk_id for hit in fused], filters)
    if not chunks:
        return []

    results = [
        RetrievedChunk(
            chunk_id=hit.chunk_id,
            paper_id=hit.paper_id,
            paper_title=chunks[hit.chunk_id][1],
            content=chunks[hit.chunk_id][0].content,
            section_path=chunks[hit.chunk_id][0].section_path,
            page_number=chunks[hit.chunk_id][0].page_number,
            kind=chunks[hit.chunk_id][0].kind,
            score=hit.score,
            dense_rank=hit.dense_rank,
            sparse_rank=hit.sparse_rank,
        )
        for hit in fused
        if hit.chunk_id in chunks
    ]

    if use_reranker and results:
        results = await _apply_reranker(query, results, reranker or get_reranker(), limit)
    else:
        results = results[:limit]

    for result in results:
        result.supporting_sentences = extract_supporting_sentences(query, result.content)

    logger.info(
        "retrieval_completed",
        query_length=len(query),
        dense=len(dense_hits),
        sparse=len(sparse_hits),
        fused=len(fused),
        returned=len(results),
    )
    return results


async def _load_chunks(
    db: AsyncSession, chunk_ids: list[uuid.UUID], filters: RetrievalFilters
) -> dict[uuid.UUID, tuple[DocumentChunk, str | None]]:
    """Fetch chunk rows and their paper titles, applying database-level filters.

    Chunk text lives here rather than in the index, so this is also where a
    vector that outlived its row (a deleted paper mid-query) gets dropped.
    """
    if not chunk_ids:
        return {}

    query = (
        select(DocumentChunk, Paper.title)
        .join(Paper, Paper.id == DocumentChunk.paper_id)
        .where(DocumentChunk.id.in_(chunk_ids))
    )

    if filters.sections:
        # Match a section prefix so "3 Results" also matches "3 Results > 3.1".
        conditions = [DocumentChunk.section_path.ilike(f"%{s}%") for s in filters.sections]
        from sqlalchemy import or_

        query = query.where(or_(*conditions))

    rows = (await db.execute(query)).all()
    return {row[0].id: (row[0], row[1]) for row in rows}


async def _apply_reranker(
    query: str, results: list[RetrievedChunk], reranker: Reranker, limit: int
) -> list[RetrievedChunk]:
    candidates = [
        RerankCandidate(chunk_id=result.chunk_id, text=result.content, original_score=result.score)
        for result in results
    ]
    ranked = await reranker.rerank(query, candidates, limit=limit)

    by_id = {result.chunk_id: result for result in results}
    reordered = []
    for entry in ranked:
        result = by_id[entry.chunk_id]
        result.rerank_score = entry.score
        reordered.append(result)
    return reordered


async def rebuild_sparse_index(db: AsyncSession, bm25: BM25Index | None = None) -> int:
    """Load every chunk into the BM25 index.

    The sparse index is in-memory, so it has to be rebuilt on process start.
    Dense vectors persist in the vector store; BM25 statistics do not.
    """
    bm25 = bm25 or get_bm25_index()
    bm25.clear()

    rows = (
        await db.execute(
            select(DocumentChunk.id, DocumentChunk.paper_id, DocumentChunk.content,
                   DocumentChunk.kind, DocumentChunk.section_path, Paper.workspace_id)
            .join(Paper, Paper.id == DocumentChunk.paper_id)
        )
    ).all()

    for chunk_id, paper_id, content, kind, section_path, workspace_id in rows:
        bm25.add(
            chunk_id,
            paper_id,
            content,
            {
                "chunk_id": str(chunk_id),
                "paper_id": str(paper_id),
                "workspace_id": str(workspace_id),
                "kind": kind,
                "section_path": section_path or "",
            },
        )

    logger.info("sparse_index_rebuilt", chunks=len(rows))
    return len(rows)
