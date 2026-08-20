"""Embeds document chunks and writes them into the vector store."""

from __future__ import annotations

import uuid

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.models import DocumentChunk, Paper
from app.retrieval.embeddings import EmbeddingProvider, get_embedding_provider
from app.retrieval.vector_store import VectorRecord, VectorStore, get_vector_store

logger = get_logger(__name__)


def _batched(items: list, size: int):  # noqa: ANN201, ANN001
    for start in range(0, len(items), size):
        yield items[start : start + size]


async def index_paper(
    db: AsyncSession,
    paper_id: uuid.UUID,
    *,
    provider: EmbeddingProvider | None = None,
    store: VectorStore | None = None,
    batch_size: int = 64,
    reindex: bool = False,
) -> int:
    """Embed a paper's chunks and upsert them into the vector store.

    Returns the number of chunks indexed. Only unembedded chunks are processed
    unless `reindex` is set, so a retried task doesn't pay to re-embed work that
    already succeeded.
    """
    provider = provider or get_embedding_provider()
    store = store or get_vector_store()

    paper = await db.get(Paper, paper_id)
    if paper is None:
        logger.warning("index_paper_missing", paper_id=str(paper_id))
        return 0

    query = select(DocumentChunk).where(DocumentChunk.paper_id == paper_id)
    if not reindex:
        query = query.where(DocumentChunk.is_embedded.is_(False))

    chunks = list(await db.scalars(query.order_by(DocumentChunk.chunk_index)))
    if not chunks:
        return 0

    if reindex:
        # Drop the old vectors first: chunk ids change when text is re-chunked,
        # so stale entries would otherwise linger and be retrievable forever.
        await store.delete_by_paper(paper_id)

    indexed = 0
    for batch in _batched(chunks, batch_size):
        vectors = await provider.embed_documents([chunk.content for chunk in batch])

        await store.upsert(
            [
                VectorRecord(
                    chunk_id=chunk.id,
                    paper_id=paper.id,
                    workspace_id=paper.workspace_id,
                    vector=vector,
                    metadata={
                        "chunk_id": str(chunk.id),
                        "paper_id": str(paper.id),
                        "workspace_id": str(paper.workspace_id),
                        "kind": chunk.kind,
                        "section_path": chunk.section_path or "",
                        "page_number": chunk.page_number or 0,
                    },
                )
                for chunk, vector in zip(batch, vectors, strict=True)
            ]
        )

        await db.execute(
            update(DocumentChunk)
            .where(DocumentChunk.id.in_([chunk.id for chunk in batch]))
            .values(is_embedded=True)
        )
        # Commit per batch so an interrupted run keeps its completed work
        # rather than re-embedding everything on retry.
        await db.commit()
        indexed += len(batch)

    logger.info(
        "paper_indexed", paper_id=str(paper_id), chunks=indexed, model=provider.model_name
    )
    return indexed


async def remove_paper_from_index(
    paper_id: uuid.UUID, *, store: VectorStore | None = None
) -> int:
    store = store or get_vector_store()
    return await store.delete_by_paper(paper_id)
