import pytest

from app.ingestion.chunking import (
    CHARS_PER_TOKEN,
    chunk_asset,
    chunk_document,
    chunk_text,
    estimate_tokens,
    split_sections,
)

PAPER = """Deep Retrieval for Science

ABSTRACT

We present a system for retrieval over scientific corpora.

1 Introduction

Retrieval augmented generation has become standard practice in recent work.

2 Methods

We describe the overall approach in this section of the paper.

2.1 Encoder

The encoder is a transformer trained with a contrastive objective.

2.2 Training

Training uses in-batch negatives sampled from a large corpus.

3 Results

Our system outperforms the baseline on every benchmark we evaluated.
"""


def test_splits_on_numbered_and_known_headings() -> None:
    titles = [section.title for section in split_sections(PAPER)]

    assert "Abstract" in titles
    assert "1 Introduction" in titles
    assert "2.1 Encoder" in titles


def test_text_before_the_first_heading_is_kept_as_front_matter() -> None:
    sections = split_sections(PAPER)

    assert sections[0].title == "Front Matter"
    assert "Deep Retrieval for Science" in sections[0].text


def test_heading_levels_reflect_numbering_depth() -> None:
    levels = {section.title: section.level for section in split_sections(PAPER)}

    assert levels["2 Methods"] == 1
    assert levels["2.1 Encoder"] == 2


def test_document_without_headings_becomes_a_single_section() -> None:
    sections = split_sections("Just a plain paragraph of text with no structure at all.")

    assert len(sections) == 1
    assert sections[0].title == "Document"


def test_empty_document_yields_no_sections() -> None:
    assert split_sections("   \n\n  ") == []


def test_subsection_chunks_carry_their_parent_heading() -> None:
    paths = {chunk.section_path for chunk in chunk_document(PAPER)}

    assert "2 Methods > 2.1 Encoder" in paths
    assert "2 Methods > 2.2 Training" in paths


def test_a_new_top_level_section_resets_the_breadcrumb() -> None:
    results_paths = [
        chunk.section_path
        for chunk in chunk_document(PAPER)
        if "Results" in (chunk.section_path or "")
    ]

    assert results_paths == ["3 Results"]


def test_chunk_indices_are_contiguous() -> None:
    chunks = chunk_document(PAPER)

    assert [chunk.index for chunk in chunks] == list(range(len(chunks)))


def test_chunks_respect_the_token_budget() -> None:
    max_tokens = 30
    long_section = "This sentence is here to make the section long. " * 60

    chunks = chunk_text(long_section, max_tokens=max_tokens, overlap_tokens=5)

    assert len(chunks) > 1
    for chunk in chunks:
        # Allow a small margin: packing works in characters, and the estimate
        # rounds down.
        assert len(chunk.content) <= max_tokens * CHARS_PER_TOKEN + CHARS_PER_TOKEN


def test_paragraphs_are_packed_together_rather_than_split_one_per_chunk() -> None:
    text = "\n\n".join(["A short paragraph about retrieval systems."] * 4)

    chunks = chunk_text(text, max_tokens=512)

    assert len(chunks) == 1


def test_oversized_paragraph_is_split_on_sentence_boundaries() -> None:
    paragraph = " ".join(f"Sentence number {i} explains a detail." for i in range(80))

    chunks = chunk_text(paragraph, max_tokens=40, overlap_tokens=5)

    assert len(chunks) > 1
    # Sentence-aware splitting should leave most pieces ending in a period.
    assert sum(chunk.content.rstrip().endswith(".") for chunk in chunks) >= len(chunks) - 1


def test_a_single_sentence_longer_than_the_budget_still_splits() -> None:
    chunks = chunk_text("word " * 500, max_tokens=20, overlap_tokens=2)

    assert len(chunks) > 1


def test_short_single_section_is_not_dropped() -> None:
    chunks = chunk_text("Too short.", max_tokens=512)

    assert len(chunks) == 1
    assert chunks[0].content == "Too short."


def test_tables_become_whole_chunks_with_their_caption() -> None:
    chunk = chunk_asset(
        kind="table",
        content="| Model | F1 |\n| --- | --- |\n| BERT | 0.89 |",
        caption="Table 1: Benchmark results.",
        page_number=4,
        index=7,
    )

    assert chunk.kind == "table"
    assert chunk.page_number == 4
    assert chunk.index == 7
    assert "Table 1" in chunk.content
    assert "| BERT | 0.89 |" in chunk.content


def test_figure_chunk_uses_its_caption_as_content() -> None:
    chunk = chunk_asset(
        kind="figure",
        content="",
        caption="Figure 2: System architecture.",
        page_number=2,
        index=0,
    )

    assert chunk.content == "Figure 2: System architecture."
    assert chunk.section_path == "Figure 2"


@pytest.mark.parametrize(
    ("text", "expected"),
    [("", 1), ("abcd", 1), ("a" * 400, 100)],
)
def test_token_estimate(text: str, expected: int) -> None:
    assert estimate_tokens(text) == expected
