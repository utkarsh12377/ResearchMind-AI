import pytest

from app.retrieval.reranker import (
    LexicalOverlapReranker,
    NoOpReranker,
    RerankCandidate,
    extract_supporting_sentences,
)


def _candidate(text: str, chunk_id: str, score: float = 0.0) -> RerankCandidate:
    return RerankCandidate(chunk_id=chunk_id, text=text, original_score=score)


@pytest.mark.asyncio
async def test_empty_candidate_list_returns_nothing() -> None:
    assert await LexicalOverlapReranker().rerank("query", []) == []


@pytest.mark.asyncio
async def test_noop_reranker_preserves_incoming_order() -> None:
    candidates = [_candidate("a", "1"), _candidate("b", "2"), _candidate("c", "3")]

    ranked = await NoOpReranker().rerank("query", candidates)

    assert [r.chunk_id for r in ranked] == ["1", "2", "3"]


@pytest.mark.asyncio
async def test_reranker_promotes_the_more_relevant_passage() -> None:
    candidates = [
        _candidate("Graph neural networks applied to molecular data.", "irrelevant"),
        _candidate("Cross encoder reranking improves passage ranking quality.", "relevant"),
    ]

    ranked = await LexicalOverlapReranker().rerank("cross encoder reranking", candidates)

    assert ranked[0].chunk_id == "relevant"
    # The promoted result records where it came from, for explainability.
    assert ranked[0].original_rank == 2
    assert ranked[0].new_rank == 1


@pytest.mark.asyncio
async def test_ranks_are_reported_for_every_result() -> None:
    candidates = [_candidate("alpha retrieval", "1"), _candidate("beta", "2")]

    ranked = await LexicalOverlapReranker().rerank("retrieval", candidates)

    assert sorted(r.new_rank for r in ranked) == [1, 2]
    assert sorted(r.original_rank for r in ranked) == [1, 2]


@pytest.mark.asyncio
async def test_limit_truncates_after_reordering() -> None:
    candidates = [
        _candidate("unrelated content here", "1"),
        _candidate("unrelated content there", "2"),
        _candidate("dense passage retrieval methods", "3"),
    ]

    ranked = await LexicalOverlapReranker().rerank("dense passage retrieval", candidates, limit=1)

    # Truncation happens after reranking, so the best result survives even
    # though it started last.
    assert len(ranked) == 1
    assert ranked[0].chunk_id == "3"


@pytest.mark.asyncio
async def test_rarer_query_terms_carry_more_weight() -> None:
    candidates = [
        _candidate("retrieval systems overview", "common-only"),
        _candidate("retrieval systems using rotary embeddings", "has-rare-term"),
        _candidate("retrieval systems in practice", "common-only-2"),
    ]

    ranked = await LexicalOverlapReranker().rerank("retrieval rotary", candidates)

    assert ranked[0].chunk_id == "has-rare-term"


@pytest.mark.asyncio
async def test_exact_phrase_match_is_rewarded() -> None:
    candidates = [
        _candidate("encoder cross methods scattered across the passage", "scattered"),
        _candidate("the cross encoder is described here", "phrase"),
    ]

    ranked = await LexicalOverlapReranker().rerank("cross encoder", candidates)

    assert ranked[0].chunk_id == "phrase"


@pytest.mark.asyncio
async def test_empty_query_falls_back_to_original_scores() -> None:
    candidates = [_candidate("a", "low", score=0.1), _candidate("b", "high", score=0.9)]

    ranked = await LexicalOverlapReranker().rerank("   ", candidates)

    assert ranked[0].chunk_id == "high"


def test_supporting_sentences_pick_the_relevant_ones() -> None:
    text = (
        "This paragraph introduces the general topic at some length. "
        "Our cross encoder reranker improves precision on scientific queries. "
        "Unrelated discussion of deployment logistics follows here."
    )

    sentences = extract_supporting_sentences("cross encoder reranker", text, limit=1)

    assert len(sentences) == 1
    assert "cross encoder reranker" in sentences[0]


def test_supporting_sentences_ignore_short_fragments() -> None:
    sentences = extract_supporting_sentences("retrieval", "Yes. No. Retrieval matters here a lot.")

    assert all(len(s) >= 20 for s in sentences)


def test_supporting_sentences_return_empty_for_an_empty_query() -> None:
    assert extract_supporting_sentences("", "Some passage text about retrieval.") == []


def test_supporting_sentences_return_empty_when_nothing_overlaps() -> None:
    assert extract_supporting_sentences("quantum", "A passage about classical optimization.") == []
