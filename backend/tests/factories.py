"""Helpers for building test fixtures (real PDFs, not mocks)."""

from __future__ import annotations

import io
import uuid

import pymupdf

from app.models import Paper, PaperStatus


def make_paper(**overrides) -> Paper:  # noqa: ANN003
    """Build a Paper with the required source-file columns already filled in."""
    checksum = overrides.pop("checksum", uuid.uuid4().hex * 2)
    defaults = {
        "original_filename": "paper.pdf",
        "content_type": "application/pdf",
        "checksum": checksum,
        "size_bytes": 1024,
        "storage_key": f"papers/{checksum[:2]}/{checksum[2:4]}/{checksum}.pdf",
        "status": PaperStatus.PENDING,
    }
    return Paper(**{**defaults, **overrides})


def build_pdf(
    *,
    title: str = "Attention Mechanisms for Scientific Retrieval",
    abstract: str = "We present a hybrid retrieval system for scientific literature.",
    body: str = "This is the introduction body text.",
    metadata_title: str | None = None,
    metadata_author: str | None = None,
    pages: int = 1,
) -> io.BytesIO:
    """Render a small paper-shaped PDF with a large title and an abstract."""
    document = pymupdf.open()

    first = document.new_page()
    # Title in a noticeably larger font so the layout heuristic can find it.
    first.insert_text((72, 90), title, fontsize=22)
    first.insert_text((72, 140), "Abstract", fontsize=11)
    first.insert_text((72, 160), abstract, fontsize=10)
    first.insert_text((72, 200), "1 Introduction", fontsize=11)
    first.insert_text((72, 220), body, fontsize=10)

    for index in range(1, pages):
        page = document.new_page()
        page.insert_text((72, 90), f"Page {index + 1} content.", fontsize=10)

    document.set_metadata(
        {
            "title": metadata_title or "",
            "author": metadata_author or "",
        }
    )

    buffer = io.BytesIO(document.tobytes())
    document.close()
    buffer.seek(0)
    return buffer


def build_pdf_with_table(
    *,
    headers: tuple[str, ...] = ("Model", "Accuracy", "F1"),
    rows: tuple[tuple[str, ...], ...] = (("BERT", "0.91", "0.89"), ("GPT-4", "0.95", "0.94")),
    caption: str = "Table 1: Benchmark results on the evaluation split.",
) -> io.BytesIO:
    """A PDF containing a ruled table plus a caption beneath it.

    PyMuPDF's table finder keys off ruling lines, so the grid is drawn
    explicitly rather than relying on whitespace alignment.
    """
    document = pymupdf.open()
    page = document.new_page()

    left, top = 72.0, 120.0
    col_width, row_height = 120.0, 24.0
    all_rows = (headers, *rows)

    for row_index, row in enumerate(all_rows):
        for col_index, cell in enumerate(row):
            x = left + col_index * col_width
            y = top + row_index * row_height
            page.draw_rect(pymupdf.Rect(x, y, x + col_width, y + row_height), width=0.8)
            page.insert_text((x + 5, y + 16), cell, fontsize=10)

    table_bottom = top + len(all_rows) * row_height
    page.insert_text((left, table_bottom + 18), caption, fontsize=9)

    buffer = io.BytesIO(document.tobytes())
    document.close()
    buffer.seek(0)
    return buffer


def build_pdf_with_figure(
    caption: str = "Figure 1: Architecture of the proposed retrieval system.",
) -> io.BytesIO:
    """A PDF containing an embedded raster image plus a caption."""
    document = pymupdf.open()
    page = document.new_page()

    # A solid 200x150 image, comfortably above the decorative-element threshold.
    pixmap = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 200, 150), False)
    pixmap.set_rect(pixmap.irect, (40, 90, 200))
    image_rect = pymupdf.Rect(72, 100, 272, 250)
    page.insert_image(image_rect, pixmap=pixmap)
    page.insert_text((72, 268), caption, fontsize=9)

    buffer = io.BytesIO(document.tobytes())
    document.close()
    buffer.seek(0)
    return buffer


def build_scanned_pdf(pages: int = 2) -> io.BytesIO:
    """A PDF with no text layer, standing in for a scanned document."""
    document = pymupdf.open()
    for _ in range(pages):
        document.new_page()

    buffer = io.BytesIO(document.tobytes())
    document.close()
    buffer.seek(0)
    return buffer
