"""Full retrieval pipeline tests: real PDFs through indexing to ranked results."""

import uuid

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import User, Workspace
from app.retrieval.embeddings import HashEmbeddingProvider
from app.retrieval.indexer import index_paper
from app.retrieval.reranker import LexicalOverlapReranker
from app.retrieval.service import (
    RetrievalFilters,
    rebuild_sparse_index,
    retrieve,
)
from app.retrieval.sparse import BM25Index
from app.retrieval.vector_store import InMemoryVectorStore
from app.worker.tasks import _process_paper
from tests.factories import build_pdf, make_paper
from tests.test_ingestion_task import _store_paper


@pytest.fixture
def provider() -> HashEmbeddingProvider:
    return HashEmbeddingProvider(dimensions=256)


@pytest.fixture
def store() -> InMemoryVectorStore:
    return InMemoryVectorStore()


@pytest.fixture
def bm25() -> BM25Index:
    return BM25Index()


async def _ingest(db: AsyncSession, provider, store, bm25, **pdf_kwargs):  # noqa: ANN001, ANN202
    """Upload, parse, chunk, and index one paper into both retrievers."""
    filename = pdf_kwargs.pop("filename", f"{uuid.uuid4().hex[:8]}.pdf")
    paper = await _store_paper(db, build_pdf(**pdf_kwargs), filename)
    await _process_paper(db, paper.id)
    await index_paper(db, paper.id, provider=provider, store=store)
    await rebuild_sparse_index(db, bm25=bm25)
    return paper


async def _owner(db: AsyncSession) -> User:
    user = await db.scalar(User.__table__.select().with_only_columns(User.id).limit(0))  # noqa: F841
    from sqlalchemy import select

    return await db.scalar(select(User).limit(1))


@pytest.mark.asyncio
async def test_retrieval_finds_the_relevant_paper(
    db_session: AsyncSession, provider, store, bm25  # noqa: ANN001
) -> None:
    target = await _ingest(
        db_session,
        provider,
        store,
        bm25,
        title="Cross Encoder Reranking for Scientific Search",
        body="We apply a cross encoder to rerank candidate passages.",
    )
    user = await _owner(db_session)

    results = await retrieve(
        db_session,
        user,
        "cross encoder reranking",
        provider=provider,
        store=store,
        bm25=bm25,
        reranker=LexicalOverlapReranker(),
    )

    assert results
    assert results[0].paper_id == target.id


@pytest.mark.asyncio
async def test_results_carry_citation_metadata(
    db_session: AsyncSession, provider, store, bm25  # noqa: ANN001
) -> None:
    await _ingest(
        db_session, provider, store, bm25, title="Hybrid Retrieval Systems", pages=2
    )
    user = await _owner(db_session)

    results = await retrieve(
        db_session, user, "hybrid retrieval", provider=provider, store=store, bm25=bm25
    )

    assert results
    top = results[0]
    assert top.paper_title
    assert top.citation.startswith(top.paper_title)
    assert top.content


@pytest.mark.asyncio
async def test_results_expose_which_retriever_found_them(
    db_session: AsyncSession, provider, store, bm25  # noqa: ANN001
) -> None:
    await _ingest(db_session, provider, store, bm25, body="Dense passage retrieval methods.")
    user = await _owner(db_session)

    results = await retrieve(
        db_session, user, "dense passage retrieval", provider=provider, store=store, bm25=bm25
    )

    assert results
    # Explainability: at least one retriever must account for every result.
    assert all(r.dense_rank is not None or r.sparse_rank is not None for r in results)


@pytest.mark.asyncio
async def test_empty_query_returns_nothing(
    db_session: AsyncSession, provider, store, bm25  # noqa: ANN001
) -> None:
    await _ingest(db_session, provider, store, bm25)
    user = await _owner(db_session)

    assert await retrieve(db_session, user, "   ", provider=provider, store=store, bm25=bm25) == []


@pytest.mark.asyncio
async def test_user_without_papers_gets_no_results(
    db_session: AsyncSession, provider, store, bm25  # noqa: ANN001
) -> None:
    await _ingest(db_session, provider, store, bm25, body="Indexed content about retrieval.")

    stranger = User(email="stranger@example.com", hashed_password="x")
    db_session.add(stranger)
    db_session.add(Workspace(name="Empty", owner=stranger))
    await db_session.commit()

    results = await retrieve(
        db_session, stranger, "retrieval", provider=provider, store=store, bm25=bm25
    )

    # Authorization is applied before scoring, so another user's indexed
    # content is not merely hidden -- it is never a candidate.
    assert results == []


@pytest.mark.asyncio
async def test_paper_filter_scopes_results(
    db_session: AsyncSession, provider, store, bm25  # noqa: ANN001
) -> None:
    first = await _ingest(
        db_session, provider, store, bm25, body="Retrieval augmented generation.", filename="a.pdf"
    )
    await _ingest(
        db_session, provider, store, bm25, body="Retrieval augmented pipelines.", filename="b.pdf"
    )
    user = await _owner(db_session)

    results = await retrieve(
        db_session,
        user,
        "retrieval",
        filters=RetrievalFilters(paper_ids=[first.id]),
        provider=provider,
        store=store,
        bm25=bm25,
    )

    assert results
    assert all(r.paper_id == first.id for r in results)


@pytest.mark.asyncio
async def test_limit_is_respected(
    db_session: AsyncSession, provider, store, bm25  # noqa: ANN001
) -> None:
    await _ingest(db_session, provider, store, bm25, pages=4)
    user = await _owner(db_session)

    results = await retrieve(
        db_session, user, "retrieval system", limit=2, provider=provider, store=store, bm25=bm25
    )

    assert len(results) <= 2


@pytest.mark.asyncio
async def test_supporting_sentences_are_extracted(
    db_session: AsyncSession, provider, store, bm25  # noqa: ANN001
) -> None:
    await _ingest(
        db_session,
        provider,
        store,
        bm25,
        abstract="We combine dense retrieval with sparse lexical matching for science.",
    )
    user = await _owner(db_session)

    results = await retrieve(
        db_session,
        user,
        "sparse lexical matching",
        provider=provider,
        store=store,
        bm25=bm25,
    )

    assert results
    assert any(r.supporting_sentences for r in results)


@pytest.mark.asyncio
async def test_disabling_the_reranker_still_returns_results(
    db_session: AsyncSession, provider, store, bm25  # noqa: ANN001
) -> None:
    await _ingest(db_session, provider, store, bm25, body="Vector databases for retrieval.")
    user = await _owner(db_session)

    results = await retrieve(
        db_session,
        user,
        "vector databases",
        use_reranker=False,
        provider=provider,
        store=store,
        bm25=bm25,
    )

    assert results
    assert all(r.rerank_score is None for r in results)


@pytest.mark.asyncio
async def test_rebuilding_the_sparse_index_counts_every_chunk(
    db_session: AsyncSession, provider, store, bm25  # noqa: ANN001
) -> None:
    await _ingest(db_session, provider, store, bm25, pages=3)

    count = await rebuild_sparse_index(db_session, bm25=bm25)

    assert count > 0
    assert bm25.size == count


@pytest.mark.asyncio
async def test_kind_filter_selects_tables(
    db_session: AsyncSession, provider, store, bm25  # noqa: ANN001
) -> None:
    from tests.factories import build_pdf_with_table

    paper = await _store_paper(db_session, build_pdf_with_table(), "withtable.pdf")
    await _process_paper(db_session, paper.id)
    await index_paper(db_session, paper.id, provider=provider, store=store)
    await rebuild_sparse_index(db_session, bm25=bm25)
    user = await _owner(db_session)

    results = await retrieve(
        db_session,
        user,
        "benchmark accuracy results",
        filters=RetrievalFilters(kinds=["table"]),
        provider=provider,
        store=store,
        bm25=bm25,
    )

    assert results
    assert all(r.kind == "table" for r in results)


@pytest.mark.asyncio
async def test_deleted_chunks_do_not_surface_from_a_stale_index(
    db_session: AsyncSession, provider, store, bm25  # noqa: ANN001
) -> None:
    """Vectors can outlive their rows; the join must drop orphans."""
    from sqlalchemy import delete

    from app.models import DocumentChunk

    paper = await _ingest(db_session, provider, store, bm25, body="Soon to be deleted content.")
    user = await _owner(db_session)

    await db_session.execute(delete(DocumentChunk).where(DocumentChunk.paper_id == paper.id))
    await db_session.commit()

    results = await retrieve(
        db_session, user, "deleted content", provider=provider, store=store, bm25=bm25
    )

    assert results == []


@pytest.mark.asyncio
async def test_query_matching_nothing_returns_empty(
    db_session: AsyncSession, provider, store, bm25  # noqa: ANN001
) -> None:
    await _ingest(db_session, provider, store, bm25, body="Graph neural networks.")
    user = await _owner(db_session)
    # Unindexed paper so the filter matches no accessible content.
    other = make_paper(workspace_id=uuid.uuid4(), uploaded_by_id=uuid.uuid4())

    results = await retrieve(
        db_session,
        user,
        "anything",
        filters=RetrievalFilters(paper_ids=[other.id]),
        provider=provider,
        store=store,
        bm25=bm25,
    )

    assert results == []
