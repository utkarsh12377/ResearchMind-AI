import pymupdf
import pytest

from app.ingestion.assets import _cells_to_markdown, extract_figures, extract_tables
from tests.factories import build_pdf, build_pdf_with_figure, build_pdf_with_table


def _first_page(pdf) -> pymupdf.Page:  # noqa: ANN001
    return pymupdf.open(stream=pdf.read(), filetype="pdf")[0]


def test_extracts_table_as_markdown_with_caption() -> None:
    tables = extract_tables(_first_page(build_pdf_with_table()), 1)

    assert len(tables) == 1
    table = tables[0]
    assert table.caption == "Table 1: Benchmark results on the evaluation split."
    assert table.page_number == 1
    assert table.row_count == 3
    assert table.column_count == 3
    assert "| Model | Accuracy | F1 |" in table.markdown
    assert "| GPT-4 | 0.95 | 0.94 |" in table.markdown


def test_pages_without_tables_yield_nothing() -> None:
    assert extract_tables(_first_page(build_pdf()), 1) == []


def test_extracts_figure_with_caption_and_image() -> None:
    figures = extract_figures(_first_page(build_pdf_with_figure()), 1)

    assert len(figures) == 1
    figure = figures[0]
    assert figure.caption == "Figure 1: Architecture of the proposed retrieval system."
    assert figure.image_png is not None
    assert figure.image_png.startswith(b"\x89PNG")


def test_figure_images_can_be_skipped() -> None:
    figures = extract_figures(_first_page(build_pdf_with_figure()), 1, include_images=False)

    assert len(figures) == 1
    assert figures[0].image_png is None
    # The bounding box is still recorded so the region stays locatable.
    assert figures[0].bbox[2] > figures[0].bbox[0]


def test_text_only_pages_yield_no_figures() -> None:
    assert extract_figures(_first_page(build_pdf()), 1) == []


def test_markdown_renderer_pads_ragged_rows() -> None:
    markdown = _cells_to_markdown([["A", "B", "C"], ["1"]])

    lines = markdown.splitlines()
    assert lines[0] == "| A | B | C |"
    assert lines[2] == "| 1 |  |  |"


def test_markdown_renderer_escapes_pipes_in_cells() -> None:
    markdown = _cells_to_markdown([["col"], ["a|b"]])

    assert r"a\|b" in markdown


def test_markdown_renderer_handles_none_cells() -> None:
    markdown = _cells_to_markdown([["A", None], [None, "B"]])

    assert "| A |  |" in markdown


@pytest.mark.parametrize("rows", [[], [[]]])
def test_markdown_renderer_handles_empty_input(rows: list) -> None:
    assert _cells_to_markdown(rows) in ("", "|  |\n| --- |")
