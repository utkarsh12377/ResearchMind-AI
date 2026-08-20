"""Cross-encoder reranking.

Bi-encoders (the embedding models used for first-stage retrieval) encode query
and passage independently, so they can only measure similarity in a shared
vector space. A cross-encoder reads query and passage *together*, which
captures interaction the bi-encoder structurally cannot — negation, argument
order, whether a number actually answers the question.

It is far too slow to score a whole corpus, so it runs only over the top-k
candidates that hybrid retrieval already narrowed down. That two-stage shape
(cheap recall, expensive precision) is the point.
"""

from __future__ import annotations

import asyncio
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

from app.core.logging import get_logger
from app.retrieval.sparse import tokenize

logger = get_logger(__name__)


@dataclass
class RerankCandidate:
    chunk_id: object
    text: str
    original_score: float = 0.0
    metadata: dict | None = None


@dataclass
class RerankResult:
    chunk_id: object
    score: float
    original_rank: int
    new_rank: int


class Reranker(ABC):
    @abstractmethod
    async def score(self, query: str, candidates: list[RerankCandidate]) -> list[float]:
        """Return a relevance score per candidate, higher is better."""

    async def rerank(
        self, query: str, candidates: list[RerankCandidate], *, limit: int | None = None
    ) -> list[RerankResult]:
        if not candidates:
            return []

        scores = await self.score(query, candidates)
        ordered = sorted(
            zip(range(len(candidates)), candidates, scores, strict=True),
            key=lambda item: item[2],
            reverse=True,
        )

        results = [
            RerankResult(
                chunk_id=candidate.chunk_id,
                score=score,
                original_rank=original_index + 1,
                new_rank=new_index + 1,
            )
            for new_index, (original_index, candidate, score) in enumerate(ordered)
        ]
        return results[:limit] if limit else results


class CrossEncoderReranker(Reranker):
    """A sentence-transformers cross-encoder.

    ms-marco-MiniLM is the default: it is small enough to run on CPU while
    still substantially outperforming bi-encoder ordering on passage ranking.
    Imported lazily because torch is a heavy dependency most deployments of the
    API process never touch.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        self.model_name = model_name
        self._model = None

    def _load(self):  # noqa: ANN202 - third-party type
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:
                raise RuntimeError(
                    "Cross-encoder reranking requires sentence-transformers. Install it, "
                    "or set RERANKER_BACKEND to 'lexical' or 'none'."
                ) from exc
            self._model = CrossEncoder(self.model_name)
        return self._model

    async def score(self, query: str, candidates: list[RerankCandidate]) -> list[float]:
        model = self._load()
        pairs = [(query, candidate.text) for candidate in candidates]
        # Scoring is CPU/GPU bound and synchronous; keep it off the event loop.
        scores = await asyncio.to_thread(model.predict, pairs)
        return [float(score) for score in scores]


class LexicalOverlapReranker(Reranker):
    """Deterministic offline reranker used when no model is available.

    This is *not* a cross-encoder and does not capture term interaction. It
    scores coverage of the query's terms, weighted toward rarer terms and
    rewarding contiguous phrase matches, so it is a meaningful improvement over
    raw fusion order for keyword-like queries while staying dependency-free and
    deterministic for tests.
    """

    def __init__(self, phrase_bonus: float = 0.5) -> None:
        self.model_name = "lexical-overlap"
        self.phrase_bonus = phrase_bonus

    async def score(self, query: str, candidates: list[RerankCandidate]) -> list[float]:
        query_terms = tokenize(query)
        if not query_terms:
            return [candidate.original_score for candidate in candidates]

        unique_terms = set(query_terms)
        # Terms appearing in fewer candidates are more discriminative.
        document_frequency = {
            term: sum(1 for c in candidates if term in tokenize(c.text)) or 1
            for term in unique_terms
        }

        query_phrase = " ".join(query_terms)
        scores = []
        for candidate in candidates:
            tokens = tokenize(candidate.text)
            token_set = set(tokens)

            coverage = sum(
                1.0 / document_frequency[term] for term in unique_terms if term in token_set
            )
            normalized = coverage / sum(1.0 / document_frequency[t] for t in unique_terms)

            if query_phrase and query_phrase in " ".join(tokens):
                normalized += self.phrase_bonus

            scores.append(normalized)

        return scores


class NoOpReranker(Reranker):
    """Preserves the incoming order. Used when reranking is disabled."""

    model_name = "none"

    async def score(self, query: str, candidates: list[RerankCandidate]) -> list[float]:
        # Descending so the existing order is preserved exactly.
        return [float(len(candidates) - index) for index in range(len(candidates))]


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def extract_supporting_sentences(query: str, text: str, *, limit: int = 3) -> list[str]:
    """Pick the sentences within a passage most relevant to the query.

    Used for citation snippets: quoting the two or three sentences that
    actually support an answer is far more useful than the whole chunk.
    """
    query_terms = set(tokenize(query))
    if not query_terms:
        return []

    scored = []
    for sentence in _SENTENCE_SPLIT.split(text):
        cleaned = sentence.strip()
        if len(cleaned) < 20:
            continue
        overlap = len(query_terms & set(tokenize(cleaned)))
        if overlap:
            scored.append((overlap, cleaned))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [sentence for _, sentence in scored[:limit]]


def get_reranker() -> Reranker:
    from app.core.config import get_settings

    settings = get_settings()
    backend = settings.reranker_backend.lower()

    if backend == "cross-encoder":
        return CrossEncoderReranker(settings.reranker_model)
    if backend == "lexical":
        return LexicalOverlapReranker()
    if backend == "none":
        return NoOpReranker()

    raise ValueError(
        f"Unknown reranker backend {settings.reranker_backend!r}; "
        "expected one of: cross-encoder, lexical, none"
    )
