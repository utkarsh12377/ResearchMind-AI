"""Indexing tests that run the real path: PDF -> chunks -> vectors -> search."""

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import DocumentChunk
from app.retrieval.embeddings import HashEmbeddingProvider
from app.retrieval.indexer import index_paper, remove_paper_from_index
from app.retrieval.vector_store import InMemoryVectorStore
from app.worker.tasks import _process_paper
from tests.factories import build_pdf
from tests.test_ingestion_task import _store_paper


@pytest.fixture
def provider() -> HashEmbeddingProvider:
    return HashEmbeddingProvider(dimensions=128)


@pytest.fixture
def store() -> InMemoryVectorStore:
    return InMemoryVectorStore()


@pytest.mark.asyncio
async def test_indexing_embeds_every_chunk(
    db_session: AsyncSession, provider, store  # noqa: ANN001
) -> None:
    paper = await _store_paper(db_session, build_pdf(pages=2), "indexed.pdf")
    await _process_paper(db_session, paper.id)

    indexed = await index_paper(db_session, paper.id, provider=provider, store=store)

    assert indexed > 0
    assert await store.count() == indexed


@pytest.mark.asyncio
async def test_indexed_chunks_are_marked_embedded(
    db_session: AsyncSession, provider, store  # noqa: ANN001
) -> None:
    paper = await _store_paper(db_session, build_pdf(), "marked.pdf")
    await _process_paper(db_session, paper.id)

    await index_paper(db_session, paper.id, provider=provider, store=store)

    chunks = list(
        await db_session.scalars(
            select(DocumentChunk).where(DocumentChunk.paper_id == paper.id)
        )
    )
    assert chunks and all(chunk.is_embedded for chunk in chunks)


@pytest.mark.asyncio
async def test_reindexing_skips_already_embedded_chunks(
    db_session: AsyncSession, provider, store  # noqa: ANN001
) -> None:
    paper = await _store_paper(db_session, build_pdf(), "skip.pdf")
    await _process_paper(db_session, paper.id)
    first = await index_paper(db_session, paper.id, provider=provider, store=store)

    second = await index_paper(db_session, paper.id, provider=provider, store=store)

    assert first > 0
    # Nothing left unembedded, so a retried task does no duplicate work.
    assert second == 0


@pytest.mark.asyncio
async def test_forced_reindex_replaces_the_old_vectors(
    db_session: AsyncSession, provider, store  # noqa: ANN001
) -> None:
    paper = await _store_paper(db_session, build_pdf(), "reindex.pdf")
    await _process_paper(db_session, paper.id)
    first = await index_paper(db_session, paper.id, provider=provider, store=store)

    second = await index_paper(db_session, paper.id, provider=provider, store=store, reindex=True)

    assert second == first
    # Stale vectors were dropped rather than accumulating alongside the new ones.
    assert await store.count() == first


@pytest.mark.asyncio
async def test_indexed_content_is_retrievable_by_meaningful_query(
    db_session: AsyncSession, provider, store  # noqa: ANN001
) -> None:
    paper = await _store_paper(
        db_session,
        build_pdf(
            title="Hybrid Retrieval for Scientific Corpora",
            abstract="We combine dense retrieval with sparse lexical matching.",
            body="Our reranking stage uses a cross encoder over candidate passages.",
        ),
        "searchable.pdf",
    )
    await _process_paper(db_session, paper.id)
    await index_paper(db_session, paper.id, provider=provider, store=store)

    query = await provider.embed_query("cross encoder reranking of candidate passages")
    hits = await store.search(query, limit=3)

    assert hits
    assert hits[0].paper_id == paper.id
    matched = await db_session.get(DocumentChunk, hits[0].chunk_id)
    assert matched is not None


@pytest.mark.asyncio
async def test_search_can_be_scoped_to_a_single_paper(
    db_session: AsyncSession, provider, store  # noqa: ANN001
) -> None:
    first = await _store_paper(db_session, build_pdf(body="Graph neural networks."), "a.pdf")
    await _process_paper(db_session, first.id)
    await index_paper(db_session, first.id, provider=provider, store=store)

    query = await provider.embed_query("graph neural networks")
    hits = await store.search(query, limit=10, filters={"paper_id": str(first.id)})

    assert hits
    assert all(hit.paper_id == first.id for hit in hits)


@pytest.mark.asyncio
async def test_indexing_a_missing_paper_is_a_no_op(
    db_session: AsyncSession, provider, store  # noqa: ANN001
) -> None:
    import uuid

    assert await index_paper(db_session, uuid.uuid4(), provider=provider, store=store) == 0


@pytest.mark.asyncio
async def test_paper_with_no_chunks_indexes_nothing(
    db_session: AsyncSession, provider, store  # noqa: ANN001
) -> None:
    paper = await _store_paper(db_session, build_pdf(), "unparsed.pdf")

    # Never processed, so no chunks exist yet.
    assert await index_paper(db_session, paper.id, provider=provider, store=store) == 0


@pytest.mark.asyncio
async def test_removing_a_paper_clears_its_vectors(
    db_session: AsyncSession, provider, store  # noqa: ANN001
) -> None:
    paper = await _store_paper(db_session, build_pdf(), "removed.pdf")
    await _process_paper(db_session, paper.id)
    await index_paper(db_session, paper.id, provider=provider, store=store)

    removed = await remove_paper_from_index(paper.id, store=store)

    assert removed > 0
    assert await store.count() == 0


@pytest.mark.asyncio
async def test_batching_indexes_every_chunk(
    db_session: AsyncSession, provider, store  # noqa: ANN001
) -> None:
    paper = await _store_paper(db_session, build_pdf(pages=3), "batched.pdf")
    await _process_paper(db_session, paper.id)

    indexed = await index_paper(
        db_session, paper.id, provider=provider, store=store, batch_size=1
    )

    assert indexed > 1
    assert await store.count() == indexed


@pytest.mark.asyncio
async def test_indexing_populates_the_sparse_index_too(
    db_session: AsyncSession, provider, store  # noqa: ANN001
) -> None:
    """Regression: papers were dense-indexed but not added to BM25.

    That left every newly uploaded paper invisible to sparse retrieval — and so
    to half of hybrid search — until the next process restart rebuilt the index.
    """
    from app.retrieval.sparse import BM25Index
    from tests.factories import build_pdf

    bm25 = BM25Index()
    paper = await _store_paper(db_session, build_pdf(body="Sparse indexing check."), "sparse.pdf")
    await _process_paper(db_session, paper.id)

    indexed = await index_paper(db_session, paper.id, provider=provider, store=store, bm25=bm25)

    assert indexed > 0
    assert bm25.size == indexed
    assert bm25.search("sparse indexing")


@pytest.mark.asyncio
async def test_removing_a_paper_clears_both_indexes(
    db_session: AsyncSession, provider, store  # noqa: ANN001
) -> None:
    from app.retrieval.sparse import BM25Index
    from tests.factories import build_pdf

    bm25 = BM25Index()
    paper = await _store_paper(db_session, build_pdf(body="Removable content."), "both.pdf")
    await _process_paper(db_session, paper.id)
    await index_paper(db_session, paper.id, provider=provider, store=store, bm25=bm25)

    await remove_paper_from_index(paper.id, store=store, bm25=bm25)

    assert await store.count() == 0
    assert bm25.size == 0
