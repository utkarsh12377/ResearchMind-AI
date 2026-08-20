"""BM25 sparse retrieval.

Dense retrieval generalizes across paraphrase but reliably misses exact
lexical matches — model names, dataset names, metric names, and numbers, which
is exactly the vocabulary scientific queries are full of. BM25 covers that gap,
and the two are fused rather than chosen between.

Implemented directly rather than pulled from a library: the scoring is a dozen
lines, and owning it means term statistics can be maintained incrementally as
papers are added instead of rebuilding a whole index per upload.
"""

from __future__ import annotations

import math
import re
import threading
import uuid
from collections import Counter
from dataclasses import dataclass, field

# Very common words carry no discriminative signal and inflate scores for long
# documents. Deliberately short: over-aggressive stopword removal hurts phrase
# queries like "attention is all you need".
_STOPWORDS = frozenset(
    """
    a an and are as at be by for from has have in is it its of on or that the
    to was were will with this these those we our us you your they their
    """.split()
)

_TOKEN_PATTERN = re.compile(r"[a-z0-9][a-z0-9\-_.]*")

# Standard BM25 parameters. k1 controls term-frequency saturation, b controls
# length normalization.
BM25_K1 = 1.5
BM25_B = 0.75


def tokenize(text: str) -> list[str]:
    """Lowercase, split on non-word characters, and drop stopwords.

    Hyphens, underscores, and dots are kept inside tokens so identifiers like
    "bert-base", "f1_score", and "10.5" survive as single terms.
    """
    return [
        token
        for token in _TOKEN_PATTERN.findall(text.lower())
        if token not in _STOPWORDS and len(token) > 1
    ]


@dataclass
class SparseHit:
    chunk_id: uuid.UUID
    paper_id: uuid.UUID
    score: float
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass
class _Document:
    chunk_id: uuid.UUID
    paper_id: uuid.UUID
    length: int
    term_frequencies: Counter
    metadata: dict[str, object]


class BM25Index:
    """In-memory BM25 index supporting incremental updates."""

    def __init__(self, k1: float = BM25_K1, b: float = BM25_B) -> None:
        self.k1 = k1
        self.b = b
        self._documents: dict[uuid.UUID, _Document] = {}
        self._document_frequency: Counter = Counter()
        self._total_length = 0
        self._lock = threading.Lock()

    @property
    def size(self) -> int:
        return len(self._documents)

    @property
    def average_length(self) -> float:
        if not self._documents:
            return 0.0
        return self._total_length / len(self._documents)

    def _remove_unlocked(self, chunk_id: uuid.UUID) -> bool:
        existing = self._documents.pop(chunk_id, None)
        if existing is None:
            return False

        self._total_length -= existing.length
        for term in existing.term_frequencies:
            self._document_frequency[term] -= 1
            # Terms that no longer appear anywhere must be dropped, or IDF
            # would be computed from a stale document count.
            if self._document_frequency[term] <= 0:
                del self._document_frequency[term]
        return True

    def add(
        self,
        chunk_id: uuid.UUID,
        paper_id: uuid.UUID,
        text: str,
        metadata: dict[str, object] | None = None,
    ) -> None:
        tokens = tokenize(text)
        frequencies = Counter(tokens)

        with self._lock:
            # Re-adding the same chunk replaces it, so corpus statistics stay
            # correct when a paper is reprocessed.
            self._remove_unlocked(chunk_id)

            self._documents[chunk_id] = _Document(
                chunk_id=chunk_id,
                paper_id=paper_id,
                length=len(tokens),
                term_frequencies=frequencies,
                metadata=metadata or {},
            )
            self._total_length += len(tokens)
            for term in frequencies:
                self._document_frequency[term] += 1

    def remove_paper(self, paper_id: uuid.UUID) -> int:
        with self._lock:
            doomed = [cid for cid, doc in self._documents.items() if doc.paper_id == paper_id]
            for chunk_id in doomed:
                self._remove_unlocked(chunk_id)
            return len(doomed)

    def clear(self) -> None:
        with self._lock:
            self._documents.clear()
            self._document_frequency.clear()
            self._total_length = 0

    def _idf(self, term: str, corpus_size: int) -> float:
        frequency = self._document_frequency.get(term, 0)
        if frequency == 0:
            return 0.0
        # Robertson/Sparck-Jones IDF with the +0.5 smoothing that keeps terms
        # appearing in most documents from going negative.
        return math.log(1 + (corpus_size - frequency + 0.5) / (frequency + 0.5))

    def search(
        self,
        query: str,
        *,
        limit: int = 10,
        filters: dict[str, object] | None = None,
    ) -> list[SparseHit]:
        query_terms = tokenize(query)
        if not query_terms:
            return []

        with self._lock:
            corpus_size = len(self._documents)
            if corpus_size == 0:
                return []
            average_length = self._total_length / corpus_size
            idf = {term: self._idf(term, corpus_size) for term in set(query_terms)}
            documents = list(self._documents.values())

        from app.retrieval.vector_store import _matches

        hits: list[SparseHit] = []
        for document in documents:
            if not _matches(document.metadata, filters):
                continue

            score = 0.0
            for term in query_terms:
                term_frequency = document.term_frequencies.get(term)
                if not term_frequency:
                    continue
                denominator = term_frequency + self.k1 * (
                    1 - self.b + self.b * (document.length / average_length)
                )
                score += idf[term] * (term_frequency * (self.k1 + 1)) / denominator

            if score > 0:
                hits.append(
                    SparseHit(
                        chunk_id=document.chunk_id,
                        paper_id=document.paper_id,
                        score=score,
                        metadata=document.metadata,
                    )
                )

        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits[:limit]


_index: BM25Index | None = None


def get_bm25_index() -> BM25Index:
    """Process-wide BM25 index (it *is* the index, so it must be shared)."""
    global _index
    if _index is None:
        _index = BM25Index()
    return _index


def reset_bm25_index() -> None:
    global _index
    _index = None
