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


def build_scanned_pdf(pages: int = 2) -> io.BytesIO:
    """A PDF with no text layer, standing in for a scanned document."""
    document = pymupdf.open()
    for _ in range(pages):
        document.new_page()

    buffer = io.BytesIO(document.tobytes())
    document.close()
    buffer.seek(0)
    return buffer
