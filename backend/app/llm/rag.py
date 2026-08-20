"""Grounded question answering with corrective re-retrieval and verification.

Implements the Self-RAG / Corrective-RAG pattern:

    retrieve -> grade sufficiency -> (rewrite + re-retrieve if weak)
             -> answer -> verify against sources -> confidence

The grading and verification steps are what separate this from a plain RAG
chain. Grading catches the case where retrieval simply missed, and lets the
system try a better query instead of confidently answering from irrelevant
context. Verification catches the opposite failure — fluent text that drifts
past what the sources actually support — and reports it rather than hiding it.

Both extra steps cost a model call, so both are individually switchable.
"""

from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.llm.context import pack_context
from app.llm.gateway import LLMGateway, Message, Role, Usage, get_gateway
from app.llm.prompts import (
    GRADE_RETRIEVAL,
    RAG_ANSWER,
    REWRITE_QUERY,
    VERIFY_ANSWER,
    format_sources,
)
from app.retrieval.service import RetrievalFilters, retrieve

logger = get_logger(__name__)

_CITATION_PATTERN = re.compile(r"\[(\d{1,2})\]")
_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class Grade:
    sufficient: bool
    reason: str = ""
    missing: str = ""


@dataclass
class Verification:
    supported: bool
    confidence: float
    unsupported_claims: list[str] = field(default_factory=list)
    reason: str = ""


@dataclass
class RagAnswer:
    question: str
    answer: str
    sources: list = field(default_factory=list)
    cited_indices: list[int] = field(default_factory=list)
    grade: Grade | None = None
    verification: Verification | None = None
    rewritten_query: str | None = None
    usage: Usage = field(default_factory=Usage)

    @property
    def confidence(self) -> float:
        """Overall confidence, combining retrieval strength and verification.

        Deliberately conservative: an answer the verifier flagged is capped low
        even when retrieval looked strong, because unsupported content is the
        failure mode that matters most here.
        """
        if not self.sources:
            return 0.0

        score = 0.5
        if self.grade and self.grade.sufficient:
            score += 0.2
        if self.cited_indices:
            score += 0.1
        if self.verification:
            if self.verification.supported:
                score = max(score, self.verification.confidence)
            else:
                score = min(score, 0.35)
        return round(min(score, 1.0), 2)


def _parse_json_response(text: str) -> dict:
    """Extract a JSON object from a model response.

    Models wrap JSON in prose or code fences often enough that requiring a bare
    object would fail routinely; this finds the object and gives up gracefully.
    """
    match = _JSON_BLOCK.search(text)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}


def extract_citations(answer: str, source_count: int) -> list[int]:
    """Return the 1-based source numbers actually cited in an answer.

    Out-of-range markers are discarded: a model citing [7] against 3 sources is
    hallucinating a reference, and silently keeping it would corrupt the
    citation mapping shown to the user.
    """
    cited = []
    for match in _CITATION_PATTERN.finditer(answer):
        index = int(match.group(1))
        if 1 <= index <= source_count and index not in cited:
            cited.append(index)
    return sorted(cited)


async def grade_retrieval(
    gateway: LLMGateway, question: str, chunks: list
) -> tuple[Grade, Usage]:
    if not chunks:
        empty = Grade(
            sufficient=False, reason="No sources were retrieved.", missing=question
        )
        return empty, Usage()

    completion = await gateway.complete(
        [
            Message(Role.SYSTEM, GRADE_RETRIEVAL.system),
            Message(
                Role.USER,
                GRADE_RETRIEVAL.render(question=question, sources=format_sources(chunks)),
            ),
        ],
        temperature=0.0,
        max_tokens=300,
    )

    data = _parse_json_response(completion.text)
    if not data:
        # An unparseable grade must not block answering; assume sufficient and
        # let verification catch problems downstream.
        unparsed = Grade(
            sufficient=True, reason="Grader response could not be parsed."
        )
        return unparsed, completion.usage

    return (
        Grade(
            sufficient=bool(data.get("sufficient", True)),
            reason=str(data.get("reason", "")),
            missing=str(data.get("missing", "")),
        ),
        completion.usage,
    )


async def rewrite_query(
    gateway: LLMGateway, question: str, missing: str
) -> tuple[str, Usage]:
    completion = await gateway.complete(
        [
            Message(Role.SYSTEM, REWRITE_QUERY.system),
            Message(Role.USER, REWRITE_QUERY.render(question=question, missing=missing or "")),
        ],
        temperature=0.3,
        max_tokens=200,
    )
    rewritten = completion.text.strip().strip('"')
    return (rewritten or question), completion.usage


async def verify_answer(
    gateway: LLMGateway, question: str, answer: str, chunks: list
) -> tuple[Verification, Usage]:
    completion = await gateway.complete(
        [
            Message(Role.SYSTEM, VERIFY_ANSWER.system),
            Message(
                Role.USER,
                VERIFY_ANSWER.render(
                    question=question, answer=answer, sources=format_sources(chunks)
                ),
            ),
        ],
        temperature=0.0,
        max_tokens=500,
    )

    data = _parse_json_response(completion.text)
    if not data:
        # Unknown support is reported as unverified with middling confidence
        # rather than silently claiming the answer is grounded.
        return (
            Verification(
                supported=True, confidence=0.5, reason="Verifier response could not be parsed."
            ),
            completion.usage,
        )

    return (
        Verification(
            supported=bool(data.get("supported", True)),
            confidence=float(data.get("confidence", 0.5)),
            unsupported_claims=list(data.get("unsupported_claims") or []),
            reason=str(data.get("reason", "")),
        ),
        completion.usage,
    )


async def answer_question(
    db: AsyncSession,
    user,  # noqa: ANN001
    question: str,
    *,
    limit: int = 8,
    filters: RetrievalFilters | None = None,
    gateway: LLMGateway | None = None,
    use_corrective_retrieval: bool = True,
    use_verification: bool = True,
    max_context_tokens: int = 6000,
    **retrieval_kwargs,
) -> RagAnswer:
    """Answer a question from the corpus, with citations and a confidence score."""
    gateway = gateway or get_gateway()
    usage = Usage()

    chunks = await retrieve(
        db, user, question, limit=limit, filters=filters, **retrieval_kwargs
    )

    grade: Grade | None = None
    rewritten: str | None = None

    if use_corrective_retrieval:
        grade, grade_usage = await grade_retrieval(gateway, question, chunks)
        usage = usage + grade_usage

        if not grade.sufficient:
            rewritten, rewrite_usage = await rewrite_query(gateway, question, grade.missing)
            usage = usage + rewrite_usage

            if rewritten and rewritten != question:
                retry_chunks = await retrieve(
                    db, user, rewritten, limit=limit, filters=filters, **retrieval_kwargs
                )
                # Merge rather than replace: the first attempt may still hold
                # the only passage covering part of the question.
                seen = {chunk.chunk_id for chunk in chunks}
                chunks = chunks + [c for c in retry_chunks if c.chunk_id not in seen]

    if not chunks:
        return RagAnswer(
            question=question,
            answer=(
                "I could not find anything in your library that addresses this question. "
                "Try uploading relevant papers, or rephrasing with more specific terms."
            ),
            grade=grade,
            usage=usage,
        )

    packed = pack_context(chunks, question, max_tokens=max_context_tokens)
    sources = packed.chunks

    completion = await gateway.complete(
        [
            Message(Role.SYSTEM, RAG_ANSWER.system),
            Message(
                Role.USER,
                RAG_ANSWER.render(question=question, sources=format_sources(sources)),
            ),
        ],
    )
    usage = usage + completion.usage

    verification: Verification | None = None
    if use_verification:
        verification, verify_usage = await verify_answer(
            gateway, question, completion.text, sources
        )
        usage = usage + verify_usage

    result = RagAnswer(
        question=question,
        answer=completion.text,
        sources=sources,
        cited_indices=extract_citations(completion.text, len(sources)),
        grade=grade,
        verification=verification,
        rewritten_query=rewritten,
        usage=usage,
    )

    logger.info(
        "rag_answered",
        sources=len(sources),
        cited=len(result.cited_indices),
        corrective=bool(rewritten),
        supported=verification.supported if verification else None,
        confidence=result.confidence,
        tokens=usage.total_tokens,
    )
    return result


async def stream_answer(
    db: AsyncSession,
    user,  # noqa: ANN001
    question: str,
    *,
    limit: int = 8,
    filters: RetrievalFilters | None = None,
    gateway: LLMGateway | None = None,
    max_context_tokens: int = 6000,
    **retrieval_kwargs,
) -> AsyncIterator[dict]:
    """Stream an answer as structured events.

    Sources are emitted before the first token so the UI can render citations
    immediately rather than waiting for the full completion. Verification is
    deliberately skipped here: it can only run on a finished answer, and
    blocking the stream to verify would defeat the point of streaming.
    """
    gateway = gateway or get_gateway()

    chunks = await retrieve(
        db, user, question, limit=limit, filters=filters, **retrieval_kwargs
    )

    if not chunks:
        yield {"type": "sources", "sources": []}
        yield {
            "type": "token",
            "text": "I could not find anything in your library that addresses this question.",
        }
        yield {"type": "done", "cited": []}
        return

    packed = pack_context(chunks, question, max_tokens=max_context_tokens)
    sources = packed.chunks

    yield {
        "type": "sources",
        "sources": [
            {
                "index": index,
                "chunk_id": str(chunk.chunk_id),
                "paper_id": str(chunk.paper_id),
                "citation": chunk.citation,
                "page_number": chunk.page_number,
            }
            for index, chunk in enumerate(sources, start=1)
        ],
    }

    collected: list[str] = []
    async for token in gateway.stream(
        [
            Message(Role.SYSTEM, RAG_ANSWER.system),
            Message(
                Role.USER,
                RAG_ANSWER.render(question=question, sources=format_sources(sources)),
            ),
        ]
    ):
        collected.append(token)
        yield {"type": "token", "text": token}

    yield {"type": "done", "cited": extract_citations("".join(collected), len(sources))}
