"""Table and figure extraction with caption association.

Tables come from PyMuPDF's built-in table finder (no external dependency) and
are serialized to Markdown, which keeps row/column structure intact for both
LLM prompting and embedding. Figures are located by their image blocks and
paired with the nearest caption line.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import pymupdf

from app.core.logging import get_logger

logger = get_logger(__name__)

_CAPTION_PATTERN = re.compile(
    r"^\s*(?:Table|TABLE|Figure|FIGURE|Fig\.?|FIG\.?)\s*(\d+|[IVXLC]+)\s*[.:—-]?\s*(.*)$"
)

# A caption sits directly above or below its element; beyond this many points
# the nearest text is unrelated body copy.
MAX_CAPTION_DISTANCE = 60.0
MIN_FIGURE_AREA = 5000.0
MAX_CAPTION_LENGTH = 1000


@dataclass
class ExtractedTable:
    page_number: int
    caption: str | None
    markdown: str
    row_count: int
    column_count: int


@dataclass
class ExtractedFigure:
    page_number: int
    caption: str | None
    bbox: tuple[float, float, float, float]
    image_png: bytes | None


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _cells_to_markdown(rows: list[list[str | None]]) -> str:
    """Render extracted cells as a Markdown table.

    Markdown (rather than CSV or raw text) because it survives chunking legibly
    and LLMs reliably read column alignment from it.
    """
    if not rows:
        return ""

    width = max(len(row) for row in rows)
    # A zero-column table would render an empty separator row ("|  |"), which
    # isn't valid Markdown and carries no information.
    if width == 0:
        return ""
    normalized = [
        [_clean(cell or "").replace("|", "\\|") for cell in row] + [""] * (width - len(row))
        for row in rows
    ]

    header, *body = normalized
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * width) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in body)
    return "\n".join(lines)


def _find_caption(
    page: pymupdf.Page, bbox: tuple[float, float, float, float], keyword: str
) -> str | None:
    """Find the caption line nearest an element's bounding box."""
    try:
        blocks = page.get_text("blocks")
    except Exception:  # noqa: BLE001 - malformed page
        return None

    _, top, _, bottom = bbox
    best: tuple[float, str] | None = None

    for block in blocks:
        block_top, block_bottom, text = block[1], block[3], _clean(str(block[4]))
        if not text:
            continue

        match = _CAPTION_PATTERN.match(text)
        if not match or keyword.lower() not in text[:12].lower():
            continue

        # Distance to the element, whether the caption sits above or below.
        distance = min(abs(block_top - bottom), abs(top - block_bottom))
        if distance > MAX_CAPTION_DISTANCE:
            continue
        if best is None or distance < best[0]:
            best = (distance, text[:MAX_CAPTION_LENGTH])

    return best[1] if best else None


def extract_tables(page: pymupdf.Page, page_number: int) -> list[ExtractedTable]:
    try:
        found = page.find_tables()
    except Exception as exc:  # noqa: BLE001 - table finder is best-effort
        logger.warning("table_extraction_failed", page=page_number, error=str(exc))
        return []

    tables: list[ExtractedTable] = []
    for table in found.tables:
        try:
            rows = table.extract()
        except Exception:  # noqa: BLE001 - skip a single malformed table
            continue

        markdown = _cells_to_markdown(rows)
        if not markdown:
            continue

        tables.append(
            ExtractedTable(
                page_number=page_number,
                caption=_find_caption(page, tuple(table.bbox), "table"),
                markdown=markdown,
                row_count=len(rows),
                column_count=max((len(row) for row in rows), default=0),
            )
        )
    return tables


def extract_figures(
    page: pymupdf.Page, page_number: int, *, include_images: bool = True
) -> list[ExtractedFigure]:
    figures: list[ExtractedFigure] = []

    for block in page.get_text("dict").get("blocks", []):
        # Type 1 blocks are images; type 0 is text.
        if block.get("type") != 1:
            continue

        bbox = tuple(block.get("bbox", (0, 0, 0, 0)))
        width, height = bbox[2] - bbox[0], bbox[3] - bbox[1]
        # Skip decorative rules, logos, and separator glyphs.
        if width * height < MIN_FIGURE_AREA:
            continue

        image_png: bytes | None = None
        if include_images:
            raw = block.get("image")
            if raw:
                try:
                    pixmap = pymupdf.Pixmap(raw)
                    image_png = pixmap.tobytes("png")
                except Exception:  # noqa: BLE001 - unsupported colorspace, keep the bbox
                    image_png = None

        figures.append(
            ExtractedFigure(
                page_number=page_number,
                caption=_find_caption(page, bbox, "fig"),
                bbox=bbox,
                image_png=image_png,
            )
        )

    return figures
