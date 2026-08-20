"""Hybrid retrieval: fusing dense and sparse result lists.

Dense cosine scores and BM25 scores live on incompatible scales — cosine is
bounded in [-1, 1] while BM25 is unbounded and corpus-dependent — so they can't
simply be added or averaged. Reciprocal Rank Fusion sidesteps the problem by
using only each item's *rank* in its own list, which needs no normalization and
no per-corpus tuning.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

# The standard RRF constant from Cormack et al. It damps the influence of the
# very top ranks so a single list can't dominate the fused ordering.
RRF_K = 60


@dataclass
class FusedHit:
    chunk_id: uuid.UUID
    paper_id: uuid.UUID
    score: float
    dense_rank: int | None = None
    sparse_rank: int | None = None
    dense_score: float | None = None
    sparse_score: float | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def found_by_both(self) -> bool:
        """Agreement between two independent retrievers is a strong signal."""
        return self.dense_rank is not None and self.sparse_rank is not None


def reciprocal_rank_fusion(
    dense: list,
    sparse: list,
    *,
    limit: int = 10,
    k: int = RRF_K,
    dense_weight: float = 1.0,
    sparse_weight: float = 1.0,
) -> list[FusedHit]:
    """Fuse two ranked lists by reciprocal rank.

    Weights let one retriever be favored (a lexical-heavy query might weight
    sparse higher) without reintroducing score-scale problems, since they scale
    the rank contribution rather than the raw score.
    """
    fused: dict[uuid.UUID, FusedHit] = {}

    def merge(results: list, weight: float, is_dense: bool) -> None:
        for rank, hit in enumerate(results, start=1):
            contribution = weight / (k + rank)
            entry = fused.get(hit.chunk_id)

            if entry is None:
                entry = FusedHit(
                    chunk_id=hit.chunk_id,
                    paper_id=hit.paper_id,
                    score=0.0,
                    metadata=dict(hit.metadata),
                )
                fused[hit.chunk_id] = entry
            else:
                # Keep whichever retriever supplied richer metadata.
                entry.metadata = {**hit.metadata, **entry.metadata}

            entry.score += contribution
            if is_dense:
                entry.dense_rank, entry.dense_score = rank, hit.score
            else:
                entry.sparse_rank, entry.sparse_score = rank, hit.score

    merge(dense, dense_weight, is_dense=True)
    merge(sparse, sparse_weight, is_dense=False)

    ordered = sorted(
        fused.values(),
        # Ties broken toward results both retrievers agreed on.
        key=lambda hit: (hit.score, hit.found_by_both),
        reverse=True,
    )
    return ordered[:limit]
