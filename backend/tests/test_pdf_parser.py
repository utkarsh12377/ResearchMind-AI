import io

import pytest

from app.ingestion.pdf_parser import PdfParseError, parse_pdf
from tests.factories import build_pdf, build_scanned_pdf


def test_parses_pages_and_text() -> None:
    parsed = parse_pdf(build_pdf(body="Retrieval augmented generation works.", pages=3))

    assert parsed.page_count == 3
    assert len(parsed.pages) == 3
    assert "Retrieval augmented generation works." in parsed.full_text
    assert parsed.pages[0].number == 1


def test_recovers_title_from_layout_when_metadata_is_missing() -> None:
    parsed = parse_pdf(build_pdf(title="Graph Retrieval for Scientific Discovery"))

    assert parsed.title == "Graph Retrieval for Scientific Discovery"


def test_prefers_usable_metadata_title_over_layout() -> None:
    parsed = parse_pdf(
        build_pdf(title="Layout Rendered Title", metadata_title="Canonical Metadata Title")
    )

    assert parsed.title == "Canonical Metadata Title"


@pytest.mark.parametrize("junk", ["untitled", "Microsoft Word", "paper", "thesis.dvi", "12"])
def test_junk_metadata_titles_fall_back_to_layout(junk: str) -> None:
    parsed = parse_pdf(build_pdf(title="A Real Descriptive Paper Title", metadata_title=junk))

    assert parsed.title == "A Real Descriptive Paper Title"


def test_extracts_abstract_and_stops_at_introduction() -> None:
    parsed = parse_pdf(
        build_pdf(
            abstract="We study hybrid retrieval over scientific corpora.",
            body="Introduction body that must not be captured.",
        )
    )

    assert parsed.abstract is not None
    assert "We study hybrid retrieval over scientific corpora." in parsed.abstract
    assert "must not be captured" not in parsed.abstract


def test_extracts_authors_from_metadata() -> None:
    parsed = parse_pdf(build_pdf(metadata_author="Ada Lovelace, Alan Turing"))

    assert parsed.authors == "Ada Lovelace, Alan Turing"


def test_flags_documents_with_no_text_layer_as_scanned() -> None:
    parsed = parse_pdf(build_scanned_pdf(pages=2))

    assert parsed.is_probably_scanned is True


def test_text_pdf_is_not_flagged_as_scanned() -> None:
    parsed = parse_pdf(build_pdf(body="A" * 500))

    assert parsed.is_probably_scanned is False


def test_non_pdf_input_raises_parse_error() -> None:
    with pytest.raises(PdfParseError):
        parse_pdf(io.BytesIO(b"this is definitely not a pdf"))
