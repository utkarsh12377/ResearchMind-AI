"""Corrective-RAG and verification tests.

The Echo provider performs no reasoning, so these use a scripted provider that
returns a queued response per call. That makes the *control flow* — when the
loop re-retrieves, how a flagged answer affects confidence, how malformed model
output is handled — testable deterministically, which is the part that carries
the risk.
"""

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.llm.gateway import Completion, LLMGateway, LLMProvider, Usage
from app.llm.rag import (
    Grade,
    RagAnswer,
    Verification,
    answer_question,
    extract_citations,
    grade_retrieval,
    stream_answer,
    verify_answer,
)
from app.retrieval.embeddings import HashEmbeddingProvider
from app.retrieval.indexer import index_paper
from app.retrieval.service import rebuild_sparse_index
from app.retrieval.sparse import BM25Index
from app.retrieval.vector_store import InMemoryVectorStore
from app.worker.tasks import _process_paper
from tests.factories import build_pdf
from tests.test_ingestion_task import _store_paper


class ScriptedProvider(LLMProvider):
    """Returns queued responses in order, so multi-step flows are deterministic."""

    name = "scripted"
    default_model = "scripted-1"

    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.prompts: list[str] = []

    async def complete(self, messages, **kwargs):  # noqa: ANN001, ANN003, ANN201
        self.prompts.append(messages[-1].content)
        text = self.responses.pop(0) if self.responses else "[exhausted]"
        return Completion(
            text=text,
            model=self.default_model,
            provider=self.name,
            finish_reason="stop",
            usage=Usage(prompt_tokens=10, completion_tokens=5),
        )

    async def stream(self, messages, **kwargs):  # noqa: ANN001, ANN003, ANN201
        self.prompts.append(messages[-1].content)
        for word in (self.responses.pop(0) if self.responses else "").split(" "):
            yield word + " "


@pytest.fixture
def provider() -> HashEmbeddingProvider:
    return HashEmbeddingProvider(dimensions=256)


@pytest.fixture
def store() -> InMemoryVectorStore:
    return InMemoryVectorStore()


@pytest.fixture
def bm25() -> BM25Index:
    return BM25Index()


async def _ingest(db, provider, store, bm25, **kwargs):  # noqa: ANN001, ANN202
    paper = await _store_paper(db, build_pdf(**kwargs), "rag.pdf")
    await _process_paper(db, paper.id)
    await index_paper(db, paper.id, provider=provider, store=store)
    await rebuild_sparse_index(db, bm25=bm25)
    from sqlalchemy import select

    from app.models import User

    return paper, await db.scalar(select(User).limit(1))


# --- Citation extraction ---------------------------------------------------


def test_extracts_cited_source_numbers() -> None:
    assert extract_citations("Claim one [1]. Claim two [2][3].", 3) == [1, 2, 3]


def test_ignores_citations_beyond_the_source_count() -> None:
    # A model citing [7] against 3 sources invented that reference.
    assert extract_citations("Supported [1] and invented [7].", 3) == [1]


def test_deduplicates_repeated_citations() -> None:
    assert extract_citations("[1] and again [1].", 2) == [1]


def test_uncited_answer_yields_no_citations() -> None:
    assert extract_citations("A confident answer with no sources.", 3) == []


# --- Confidence scoring ----------------------------------------------------


def test_confidence_is_zero_without_sources() -> None:
    assert RagAnswer(question="q", answer="a").confidence == 0.0


def test_unsupported_answers_are_capped_low_even_with_good_retrieval() -> None:
    answer = RagAnswer(
        question="q",
        answer="a [1]",
        sources=[object()],
        cited_indices=[1],
        grade=Grade(sufficient=True),
        verification=Verification(supported=False, confidence=0.9),
    )

    # Verification failure dominates: unsupported content is the failure mode
    # that matters most, so a high verifier confidence must not rescue it.
    assert answer.confidence <= 0.35


def test_supported_and_cited_answers_score_high() -> None:
    answer = RagAnswer(
        question="q",
        answer="a [1]",
        sources=[object()],
        cited_indices=[1],
        grade=Grade(sufficient=True),
        verification=Verification(supported=True, confidence=0.9),
    )

    assert answer.confidence >= 0.8


# --- Grading ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_grading_with_no_chunks_reports_insufficient() -> None:
    gateway = LLMGateway(ScriptedProvider([]))

    grade, usage = await grade_retrieval(gateway, "question", [])

    assert grade.sufficient is False
    # No model call is needed to know an empty result set is insufficient.
    assert usage.total_tokens == 0


@pytest.mark.asyncio
async def test_grading_parses_json_wrapped_in_prose() -> None:
    gateway = LLMGateway(
        ScriptedProvider(['Sure! ```json\n{"sufficient": false, "missing": "metrics"}\n```'])
    )

    grade, _ = await grade_retrieval(gateway, "q", [_FakeChunk("text")])

    assert grade.sufficient is False
    assert grade.missing == "metrics"


@pytest.mark.asyncio
async def test_unparseable_grade_does_not_block_answering() -> None:
    gateway = LLMGateway(ScriptedProvider(["I cannot comply."]))

    grade, _ = await grade_retrieval(gateway, "q", [_FakeChunk("text")])

    assert grade.sufficient is True
    assert "could not be parsed" in grade.reason


# --- Verification ----------------------------------------------------------


@pytest.mark.asyncio
async def test_verification_reports_unsupported_claims() -> None:
    gateway = LLMGateway(
        ScriptedProvider(
            ['{"supported": false, "confidence": 0.2, '
             '"unsupported_claims": ["the 99% figure"], "reason": "not in sources"}']
        )
    )

    verification, _ = await verify_answer(gateway, "q", "answer", [_FakeChunk("text")])

    assert verification.supported is False
    assert verification.unsupported_claims == ["the 99% figure"]


@pytest.mark.asyncio
async def test_unparseable_verification_is_reported_as_uncertain() -> None:
    gateway = LLMGateway(ScriptedProvider(["not json at all"]))

    verification, _ = await verify_answer(gateway, "q", "answer", [_FakeChunk("t")])

    assert verification.confidence == 0.5
    assert "could not be parsed" in verification.reason


class _FakeChunk:
    def __init__(self, content: str) -> None:
        self.content = content
        self.citation = "Fake Paper — 1 Introduction"


# --- End-to-end flow -------------------------------------------------------


@pytest.mark.asyncio
async def test_answer_includes_sources_and_citations(
    db_session: AsyncSession, provider, store, bm25  # noqa: ANN001
) -> None:
    _, user = await _ingest(
        db_session, provider, store, bm25, abstract="Dense retrieval improves recall on science."
    )
    gateway = LLMGateway(
        ScriptedProvider(
            ['{"sufficient": true}', "Dense retrieval improves recall [1].",
             '{"supported": true, "confidence": 0.9}']
        )
    )

    result = await answer_question(
        db_session, user, "dense retrieval recall",
        gateway=gateway, provider=provider, store=store, bm25=bm25,
    )

    assert result.sources
    assert result.cited_indices == [1]
    assert result.confidence >= 0.8


@pytest.mark.asyncio
async def test_insufficient_retrieval_triggers_a_rewritten_query(
    db_session: AsyncSession, provider, store, bm25  # noqa: ANN001
) -> None:
    _, user = await _ingest(db_session, provider, store, bm25, body="Retrieval systems overview.")
    gateway = LLMGateway(
        ScriptedProvider(
            [
                '{"sufficient": false, "missing": "benchmark numbers"}',
                "retrieval benchmark accuracy numbers",  # the rewritten query
                "Answer from merged sources [1].",
                '{"supported": true, "confidence": 0.8}',
            ]
        )
    )

    result = await answer_question(
        db_session, user, "how well does it do?",
        gateway=gateway, provider=provider, store=store, bm25=bm25,
    )

    assert result.grade is not None and result.grade.sufficient is False
    assert result.rewritten_query == "retrieval benchmark accuracy numbers"


@pytest.mark.asyncio
async def test_no_results_returns_an_honest_answer(
    db_session: AsyncSession, provider, store, bm25  # noqa: ANN001
) -> None:
    from app.models import User, Workspace

    stranger = User(email="empty@example.com", hashed_password="x")
    db_session.add_all([stranger, Workspace(name="Empty", owner=stranger)])
    await db_session.commit()

    gateway = LLMGateway(ScriptedProvider([]))

    result = await answer_question(
        db_session, stranger, "anything at all",
        gateway=gateway, provider=provider, store=store, bm25=bm25,
    )

    # It must say it doesn't know rather than answering from model priors.
    assert "could not find" in result.answer.lower()
    assert result.confidence == 0.0
    assert result.sources == []


@pytest.mark.asyncio
async def test_verification_can_be_disabled(
    db_session: AsyncSession, provider, store, bm25  # noqa: ANN001
) -> None:
    _, user = await _ingest(db_session, provider, store, bm25, body="Some indexed content.")
    gateway = LLMGateway(ScriptedProvider(['{"sufficient": true}', "An answer [1]."]))

    result = await answer_question(
        db_session, user, "indexed content", use_verification=False,
        gateway=gateway, provider=provider, store=store, bm25=bm25,
    )

    assert result.verification is None


@pytest.mark.asyncio
async def test_streaming_emits_sources_before_tokens(
    db_session: AsyncSession, provider, store, bm25  # noqa: ANN001
) -> None:
    _, user = await _ingest(db_session, provider, store, bm25, body="Streaming test content.")
    gateway = LLMGateway(ScriptedProvider(["Streamed answer [1]."]))

    events = [
        event
        async for event in stream_answer(
            db_session, user, "streaming test",
            gateway=gateway, provider=provider, store=store, bm25=bm25,
        )
    ]

    types = [event["type"] for event in events]
    assert types[0] == "sources"
    assert "token" in types
    assert types[-1] == "done"


@pytest.mark.asyncio
async def test_streaming_with_no_results_still_terminates(
    db_session: AsyncSession, provider, store, bm25  # noqa: ANN001
) -> None:
    from app.models import User, Workspace

    stranger = User(email="nostream@example.com", hashed_password="x")
    db_session.add_all([stranger, Workspace(name="Empty", owner=stranger)])
    await db_session.commit()

    events = [
        event
        async for event in stream_answer(
            db_session, stranger, "nothing here",
            gateway=LLMGateway(ScriptedProvider([])),
            provider=provider, store=store, bm25=bm25,
        )
    ]

    assert events[-1]["type"] == "done"
