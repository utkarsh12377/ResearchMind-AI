from dataclasses import dataclass

from app.llm.context import estimate_tokens, pack_context


@dataclass
class Chunk:
    content: str
    citation: str = "Paper — Section"


def test_everything_fits_within_a_generous_budget() -> None:
    chunks = [Chunk("short passage " * 5) for _ in range(3)]

    packed = pack_context(chunks, "query", max_tokens=10_000, reserved_tokens=0)

    assert len(packed.chunks) == 3
    assert not packed.was_reduced


def test_budget_reserves_room_for_the_prompt_and_output() -> None:
    chunks = [Chunk("word " * 400)]

    packed = pack_context(chunks, "query", max_tokens=600, reserved_tokens=500)

    # Only 100 tokens remain, far less than the 500-token passage.
    assert packed.used_tokens <= 100


def test_highest_ranked_chunks_are_kept_first() -> None:
    chunks = [Chunk("first " * 100), Chunk("second " * 100), Chunk("third " * 100)]

    packed = pack_context(chunks, "first", max_tokens=200, reserved_tokens=0)

    assert packed.chunks[0].content.startswith("first")


def test_oversized_chunk_is_reduced_to_relevant_sentences_not_dropped() -> None:
    text = (
        "Unrelated background about deployment logistics and history. "
        + "Filler sentence with no bearing on the question. " * 40
        + "The cross encoder reranker improved precision substantially."
    )

    packed = pack_context(
        [Chunk(text)], "cross encoder reranker", max_tokens=120, reserved_tokens=0
    )

    assert packed.compressed_count == 1
    assert "cross encoder reranker" in packed.chunks[0].content


def test_chunk_with_no_relevant_sentences_is_dropped() -> None:
    big = Chunk("Entirely unrelated content about gardening. " * 60)

    packed = pack_context([big], "quantum error correction", max_tokens=60, reserved_tokens=0)

    assert packed.dropped_count == 1
    assert packed.chunks == []


def test_no_budget_drops_everything() -> None:
    packed = pack_context([Chunk("content " * 50)], "q", max_tokens=100, reserved_tokens=100)

    assert packed.chunks == []


def test_empty_input_is_handled() -> None:
    packed = pack_context([], "query", max_tokens=1000)

    assert packed.chunks == []
    assert not packed.was_reduced


def test_token_estimate_matches_the_chunking_heuristic() -> None:
    assert estimate_tokens("a" * 400) == 100
