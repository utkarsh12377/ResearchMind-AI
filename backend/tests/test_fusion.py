import uuid
from dataclasses import dataclass, field

import pytest

from app.retrieval.fusion import RRF_K, reciprocal_rank_fusion


@dataclass
class FakeHit:
    chunk_id: uuid.UUID
    paper_id: uuid.UUID
    score: float
    metadata: dict = field(default_factory=dict)


def _hit(score: float = 1.0, **metadata):  # noqa: ANN001, ANN202
    return FakeHit(
        chunk_id=uuid.uuid4(), paper_id=uuid.uuid4(), score=score, metadata=metadata
    )


def test_empty_inputs_produce_no_results() -> None:
    assert reciprocal_rank_fusion([], []) == []


def test_a_single_list_passes_through_in_order() -> None:
    hits = [_hit(), _hit(), _hit()]

    fused = reciprocal_rank_fusion(hits, [])

    assert [f.chunk_id for f in fused] == [h.chunk_id for h in hits]


def test_agreement_between_retrievers_outranks_a_single_top_result() -> None:
    agreed = _hit()
    dense_only = _hit()

    fused = reciprocal_rank_fusion(
        dense=[dense_only, agreed],
        sparse=[agreed],
        limit=2,
    )

    # Rank 2 in dense plus rank 1 in sparse beats rank 1 in dense alone.
    assert fused[0].chunk_id == agreed.chunk_id
    assert fused[0].found_by_both


def test_fusion_records_both_ranks_and_scores() -> None:
    shared = _hit(score=0.9)

    [fused] = reciprocal_rank_fusion([shared], [FakeHit(shared.chunk_id, shared.paper_id, 4.2)])

    assert fused.dense_rank == 1
    assert fused.sparse_rank == 1
    assert fused.dense_score == 0.9
    assert fused.sparse_score == 4.2


def test_score_matches_the_rrf_formula() -> None:
    hit = _hit()

    [fused] = reciprocal_rank_fusion([hit], [])

    assert fused.score == pytest.approx(1.0 / (RRF_K + 1))


def test_incompatible_score_scales_do_not_distort_ranking() -> None:
    """BM25 scores are unbounded; RRF must ignore magnitude and use rank only."""
    dense_top = _hit(score=0.99)
    sparse_top = _hit(score=250.0)

    fused = reciprocal_rank_fusion([dense_top], [sparse_top], limit=2)

    # Both are rank 1 in their own list, so neither dominates on magnitude.
    assert fused[0].score == pytest.approx(fused[1].score)


def test_weights_shift_the_balance_between_retrievers() -> None:
    dense_top = _hit()
    sparse_top = _hit()

    fused = reciprocal_rank_fusion(
        [dense_top], [sparse_top], limit=2, sparse_weight=3.0
    )

    assert fused[0].chunk_id == sparse_top.chunk_id


def test_results_only_in_one_list_are_still_included() -> None:
    dense_only = _hit()
    sparse_only = _hit()

    fused = reciprocal_rank_fusion([dense_only], [sparse_only], limit=10)

    assert {f.chunk_id for f in fused} == {dense_only.chunk_id, sparse_only.chunk_id}
    assert not any(f.found_by_both for f in fused)


def test_limit_is_respected() -> None:
    hits = [_hit() for _ in range(10)]

    assert len(reciprocal_rank_fusion(hits, [], limit=3)) == 3


def test_metadata_from_both_retrievers_is_merged() -> None:
    dense = _hit(kind="text")
    sparse = FakeHit(dense.chunk_id, dense.paper_id, 2.0, {"section_path": "3 Results"})

    [fused] = reciprocal_rank_fusion([dense], [sparse])

    assert fused.metadata["kind"] == "text"
    assert fused.metadata["section_path"] == "3 Results"
