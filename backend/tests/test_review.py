import json

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.llm.gateway import LLMGateway
from app.models import Paper, User
from app.research.review import (
    LiteratureReview,
    ReviewSection,
    ReviewSource,
    generate_literature_review,
    plan_outline,
)
from app.retrieval.embeddings import HashEmbeddingProvider
from app.retrieval.indexer import index_paper
from app.retrieval.service import rebuild_sparse_index
from app.retrieval.sparse import BM25Index
from app.retrieval.vector_store import InMemoryVectorStore
from app.worker.tasks import _process_paper
from tests.factories import build_pdf
from tests.test_ingestion_task import _store_paper
from tests.test_rag import ScriptedProvider


@pytest.fixture
def provider() -> HashEmbeddingProvider:
    return HashEmbeddingProvider(dimensions=256)


@pytest.fixture
def store() -> InMemoryVectorStore:
    return InMemoryVectorStore()


@pytest.fixture
def bm25() -> BM25Index:
    return BM25Index()


async def _ingest(db: AsyncSession, provider, store, bm25, *, filename, **kwargs) -> Paper:  # noqa: ANN001
    paper = await _store_paper(db, build_pdf(**kwargs), filename)
    await _process_paper(db, paper.id)
    await index_paper(db, paper.id, provider=provider, store=store)
    await rebuild_sparse_index(db, bm25=bm25)
    return paper


async def _owner(db: AsyncSession) -> User:
    return await db.scalar(select(User).limit(1))


def _source(index: int, **overrides) -> ReviewSource:  # noqa: ANN003
    import uuid

    defaults = {
        "index": index,
        "paper_id": uuid.uuid4(),
        "title": f"Paper {index}",
        "authors": "Ada Lovelace",
        "year": 2020 + index,
    }
    return ReviewSource(**{**defaults, **overrides})


def test_a_source_renders_a_full_citation() -> None:
    citation = _source(1, title="Dense Retrieval", authors="Ada Lovelace", year=2021).citation

    assert citation == "Ada Lovelace. Dense Retrieval. 2021"


def test_a_source_citation_survives_missing_metadata() -> None:
    assert _source(1, authors=None, year=None).citation == "Paper 1"


def test_bibtex_keys_are_derived_from_author_year_and_title() -> None:
    key = _source(1, title="Dense Retrieval", authors="Ada Lovelace", year=2021).bibtex_key()

    assert key == "lovelace2021dense"


def test_only_cited_sources_reach_the_reference_list() -> None:
    """A paper that was retrieved but never cited is not a reference."""
    review = LiteratureReview(
        topic="t",
        title="T",
        sources=[_source(1), _source(2)],
        sections=[ReviewSection(heading="H", focus="f", text="Claim [1].", source_indices=[1])],
    )

    assert [s.index for s in review.cited_sources] == [1]


def test_markdown_includes_headings_and_references() -> None:
    review = LiteratureReview(
        topic="t",
        title="A Review",
        sources=[_source(1)],
        sections=[
            ReviewSection(heading="Overview", focus="f", text="Claim [1].", source_indices=[1])
        ],
    )

    markdown = review.to_markdown()

    assert markdown.startswith("# A Review")
    assert "## Overview" in markdown
    assert "## References" in markdown


def test_markdown_omits_references_when_nothing_was_cited() -> None:
    review = LiteratureReview(
        topic="t",
        title="A Review",
        sources=[_source(1)],
        sections=[ReviewSection(heading="Overview", focus="f", text="No citations here.")],
    )

    assert "## References" not in review.to_markdown()


def test_bibtex_export_covers_the_cited_sources() -> None:
    review = LiteratureReview(
        topic="t",
        title="T",
        sources=[_source(1, title="Dense Retrieval", authors="Ada Lovelace", year=2021)],
        sections=[ReviewSection(heading="H", focus="f", text="[1]", source_indices=[1])],
    )

    bibtex = review.to_bibtex()

    assert "@article{lovelace2021dense" in bibtex
    assert "title = {Dense Retrieval}" in bibtex


def test_word_count_sums_across_sections() -> None:
    review = LiteratureReview(
        topic="t",
        title="T",
        sections=[
            ReviewSection(heading="A", focus="", text="one two three"),
            ReviewSection(heading="B", focus="", text="four five"),
        ],
    )

    assert review.word_count == 5


@pytest.mark.asyncio
async def test_the_outline_is_read_from_the_model() -> None:
    payload = json.dumps(
        {
            "title": "Advances in Dense Retrieval",
            "sections": [{"heading": "Encoders", "focus": "Architecture choices"}],
        }
    )
    gateway = LLMGateway(ScriptedProvider([payload]))

    title, sections, _ = await plan_outline("dense retrieval", [_source(1)], gateway=gateway)

    assert title == "Advances in Dense Retrieval"
    assert sections[0].heading == "Encoders"


@pytest.mark.asyncio
async def test_an_unusable_outline_falls_back_to_a_standard_structure() -> None:
    gateway = LLMGateway(ScriptedProvider(["I cannot do that."]))

    title, sections, _ = await plan_outline("dense retrieval", [_source(1)], gateway=gateway)

    assert title == "A Review of dense retrieval"
    assert [s.heading for s in sections] == [
        "Overview",
        "Approaches",
        "Evaluation",
        "Open problems",
    ]


@pytest.mark.asyncio
async def test_an_outline_failure_still_returns_sections() -> None:
    class Broken(ScriptedProvider):
        async def complete(self, messages, **kwargs):  # noqa: ANN001, ANN003, ANN201
            raise RuntimeError("down")

    _, sections, _ = await plan_outline("x", [_source(1)], gateway=LLMGateway(Broken([])))

    assert sections


@pytest.mark.asyncio
async def test_a_review_with_no_papers_produces_no_sections(db_session: AsyncSession) -> None:
    review = await generate_literature_review(
        db_session, None, "empty topic", [], gateway=LLMGateway(ScriptedProvider([]))
    )

    assert review.sections == []
    assert review.sources == []


@pytest.mark.asyncio
async def test_a_review_drafts_every_planned_section(
    db_session: AsyncSession, provider, store, bm25  # noqa: ANN001
) -> None:
    paper = await _ingest(
        db_session,
        provider,
        store,
        bm25,
        filename="a.pdf",
        title="Dense Retrieval Methods",
        body="Dense retrieval uses a dual encoder trained with in-batch negatives.",
    )
    user = await _owner(db_session)

    outline = json.dumps(
        {
            "title": "Dense Retrieval: A Review",
            "sections": [
                {"heading": "Encoders", "focus": "architectures"},
                {"heading": "Training", "focus": "objectives"},
            ],
        }
    )
    gateway = LLMGateway(
        ScriptedProvider([outline, "Dual encoders dominate [1].", "In-batch negatives [1]."])
    )

    review = await generate_literature_review(
        db_session, user, "dense retrieval", [paper.id], gateway=gateway
    )

    assert [s.heading for s in review.sections] == ["Encoders", "Training"]
    assert review.word_count > 0


@pytest.mark.asyncio
async def test_a_citation_outside_the_sections_sources_is_dropped(
    db_session: AsyncSession, provider, store, bm25  # noqa: ANN001
) -> None:
    """A marker pointing at a paper the section never saw is a fabricated reference."""
    paper = await _ingest(
        db_session,
        provider,
        store,
        bm25,
        filename="a.pdf",
        title="Dense Retrieval Methods",
        body="Dense retrieval uses a dual encoder.",
    )
    user = await _owner(db_session)

    outline = json.dumps({"title": "R", "sections": [{"heading": "H", "focus": "f"}]})
    gateway = LLMGateway(ScriptedProvider([outline, "A claim [1] and a fabricated one [7]."]))

    review = await generate_literature_review(
        db_session, user, "dense retrieval", [paper.id], gateway=gateway
    )

    assert review.sections[0].source_indices == [1]


@pytest.mark.asyncio
async def test_a_section_with_no_retrieval_says_so(
    db_session: AsyncSession, provider, store, bm25  # noqa: ANN001
) -> None:
    paper = await _ingest(
        db_session, provider, store, bm25, filename="a.pdf", body="Graph neural networks."
    )
    user = await _owner(db_session)

    outline = json.dumps({"title": "R", "sections": [{"heading": "H", "focus": "f"}]})
    gateway = LLMGateway(ScriptedProvider([outline, "unused"]))

    review = await generate_literature_review(
        db_session, user, "quantum error correction", [paper.id], gateway=gateway
    )

    assert review.sections
