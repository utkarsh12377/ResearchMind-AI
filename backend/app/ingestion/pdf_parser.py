"""Layout-aware PDF parsing built on PyMuPDF.

Extracts per-page text plus best-effort bibliographic metadata (title,
authors, abstract). Papers rarely carry trustworthy PDF metadata, so the title
and authors are recovered from the rendered layout when the embedded metadata
is missing or junk ("untitled", a LaTeX temp name, and so on).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import BinaryIO

import pymupdf

# Embedded PDF titles are frequently placeholders left by the authoring tool.
_JUNK_TITLE_PATTERN = re.compile(
    r"^(untitled|microsoft word|document\d*|paper|manuscript|\d+|.*\.(dvi|tex|doc|docx|pdf|indd))$",
    re.IGNORECASE,
)

_ABSTRACT_PATTERN = re.compile(
    r"\bA\s?B\s?S\s?T\s?R\s?A\s?C\s?T\b|\bAbstract\b\s*[-—:.]?",
    re.IGNORECASE,
)

# Where the abstract stops: the next major section heading.
_ABSTRACT_END_PATTERN = re.compile(
    r"\n\s*(?:(?:[IVX]+|\d+)[.)]?\s*)?"
    r"(?:INTRODUCTION|Introduction|KEYWORDS|Keywords|Index Terms|CCS CONCEPTS)\b",
)

MIN_TITLE_LENGTH = 6
MAX_TITLE_LENGTH = 400
MAX_ABSTRACT_LENGTH = 5000


@dataclass
class ParsedPage:
    number: int
    text: str
    char_count: int


@dataclass
class ParsedDocument:
    page_count: int
    pages: list[ParsedPage] = field(default_factory=list)
    title: str | None = None
    authors: str | None = None
    abstract: str | None = None
    is_probably_scanned: bool = False

    @property
    def full_text(self) -> str:
        return "\n\n".join(page.text for page in self.pages)


class PdfParseError(Exception):
    """Raised when a file cannot be opened or read as a PDF."""


def _clean(value: str | None) -> str | None:
    if not value:
        return None
    collapsed = re.sub(r"\s+", " ", value).strip()
    return collapsed or None


def _is_usable_title(title: str | None) -> bool:
    if not title:
        return False
    if len(title) < MIN_TITLE_LENGTH or len(title) > MAX_TITLE_LENGTH:
        return False
    return not _JUNK_TITLE_PATTERN.match(title.strip())


def _extract_title_from_layout(page: pymupdf.Page) -> str | None:
    """Recover the title as the largest-font text block near the top of page 1.

    Academic PDFs almost always typeset the title in the largest font on the
    first page, which is far more reliable than the embedded metadata.
    """
    try:
        blocks = page.get_text("dict")["blocks"]
    except Exception:  # noqa: BLE001 - malformed page structure, fall back to None
        return None

    candidates: list[tuple[float, float, str]] = []
    for block in blocks:
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            if not spans:
                continue
            text = _clean("".join(span.get("text", "") for span in spans))
            if not text:
                continue
            max_size = max(span.get("size", 0.0) for span in spans)
            top = line.get("bbox", [0, 0, 0, 0])[1]
            candidates.append((max_size, top, text))

    if not candidates:
        return None

    largest_size = max(size for size, _, _ in candidates)
    # Group the consecutive lines sharing the largest font: multi-line titles
    # are common and would otherwise be truncated to their first line.
    title_lines = [
        text
        for size, top, text in sorted(candidates, key=lambda item: item[1])
        if size >= largest_size - 0.5 and top < page.rect.height * 0.5
    ]

    title = _clean(" ".join(title_lines))
    return title if _is_usable_title(title) else None


def _extract_abstract(first_pages_text: str) -> str | None:
    match = _ABSTRACT_PATTERN.search(first_pages_text)
    if not match:
        return None

    body = first_pages_text[match.end() :]
    end_match = _ABSTRACT_END_PATTERN.search(body)
    if end_match:
        body = body[: end_match.start()]

    return _clean(body[:MAX_ABSTRACT_LENGTH])


def _extract_authors(metadata_author: str | None) -> str | None:
    """Normalize the PDF's author metadata field.

    Layout-based author recovery is deliberately deferred: the line between
    authors and affiliations is ambiguous without a trained model, and
    Milestone 9 resolves authors properly via DOI/arXiv lookup.
    """
    author = _clean(metadata_author)
    if not author or len(author) > 1000:
        return None
    return author


def parse_pdf(source: BinaryIO) -> ParsedDocument:
    """Parse a PDF into pages plus bibliographic metadata."""
    source.seek(0)
    data = source.read()

    try:
        document = pymupdf.open(stream=data, filetype="pdf")
    except Exception as exc:  # noqa: BLE001 - PyMuPDF raises varied exception types
        raise PdfParseError(f"Could not open file as a PDF: {exc}") from exc

    try:
        pages = [
            ParsedPage(number=index + 1, text=text, char_count=len(text.strip()))
            for index, text in enumerate(page.get_text("text") for page in document)
        ]

        metadata = document.metadata or {}
        title = _clean(metadata.get("title"))
        if not _is_usable_title(title) and document.page_count:
            title = _extract_title_from_layout(document[0])
        elif not _is_usable_title(title):
            title = None

        header_text = "\n\n".join(page.text for page in pages[:3])

        # A text layer that is essentially empty means the pages are images,
        # so the document needs OCR (Milestone 7) rather than text extraction.
        total_chars = sum(page.char_count for page in pages)
        is_probably_scanned = bool(pages) and total_chars < 100 * len(pages)

        return ParsedDocument(
            page_count=document.page_count,
            pages=pages,
            title=title,
            authors=_extract_authors(metadata.get("author")),
            abstract=_extract_abstract(header_text),
            is_probably_scanned=is_probably_scanned,
        )
    finally:
        document.close()
