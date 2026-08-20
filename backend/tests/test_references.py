import pytest

from app.ingestion.references import find_references_section, parse_references

NUMBERED = """
Body text discussing prior work and references to other systems.

References

[1] A. Vaswani, N. Shazeer. Attention is all you need. In NeurIPS, 2017. arXiv:1706.03762
[2] J. Devlin, M. Chang. BERT: Pre-training of deep bidirectional transformers. 2019.
    doi:10.18653/v1/N19-1423.
[3] T. Brown et al. Language models are few-shot learners. NeurIPS, 2020.
"""


def test_finds_references_section() -> None:
    section = find_references_section(NUMBERED)

    assert section is not None
    assert "Attention is all you need" in section
    assert "Body text discussing" not in section


def test_returns_none_without_a_references_heading() -> None:
    assert find_references_section("Just body text, no bibliography here.") is None


def test_appendix_is_excluded_from_the_bibliography() -> None:
    text = NUMBERED + "\n\nAppendix A\n\nSupplementary derivations follow here at length.\n"

    section = find_references_section(text)

    assert "Supplementary derivations" not in section


def test_parses_numbered_entries() -> None:
    references = parse_references(NUMBERED)

    assert len(references) == 3
    assert [r.order for r in references] == [1, 2, 3]


def test_extracts_arxiv_id() -> None:
    reference = parse_references(NUMBERED)[0]

    assert reference.arxiv_id == "1706.03762"
    assert reference.year == 2017


def test_extracts_doi_without_trailing_punctuation() -> None:
    reference = parse_references(NUMBERED)[1]

    assert reference.doi == "10.18653/v1/N19-1423"


def test_extracts_titles_rather_than_authors_or_venue() -> None:
    titles = [r.title for r in parse_references(NUMBERED)]

    assert titles[0] == "Attention is all you need"
    assert titles[2] == "Language models are few-shot learners"


def test_handles_unnumbered_author_year_bibliographies() -> None:
    text = """
References

Vaswani, A., & Shazeer, N. (2017). Attention is all you need. NeurIPS.

Devlin, J. (2019). BERT: Pre-training of deep bidirectional transformers. NAACL.
"""

    references = parse_references(text)

    assert len(references) == 2
    assert references[0].year == 2017


def test_the_last_references_heading_wins() -> None:
    text = """
References

This early mention should be ignored because a real bibliography follows.

References

[1] A. Author. A genuine cited work with enough length. 2020.
[2] B. Author. Another genuine cited work with length. 2021.
"""

    references = parse_references(text)

    assert all("early mention" not in r.raw_text for r in references)


def test_short_fragments_are_skipped() -> None:
    text = "References\n\n[1] 12\n[2] A properly formed reference entry, 2020.\n"

    references = parse_references(text)

    assert len(references) == 1


@pytest.mark.parametrize("heading", ["References", "REFERENCES", "Bibliography", "Works Cited"])
def test_common_heading_variants_are_recognized(heading: str) -> None:
    text = f"{heading}\n\n[1] A. Author. Some cited work title here. 2020.\n"

    assert len(parse_references(text)) == 1


def test_no_references_returns_empty_list() -> None:
    assert parse_references("A paper with no bibliography at all.") == []
