"""OCR for scanned papers, behind a swappable engine interface.

Only pages whose text layer is effectively empty are rendered and OCR'd, since
OCR is orders of magnitude slower than reading an existing text layer and its
output is noisier.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pymupdf

from app.core.logging import get_logger

logger = get_logger(__name__)

# Below this many characters a page is treated as having no usable text layer.
MIN_CHARS_FOR_TEXT_LAYER = 100


class OcrUnavailableError(RuntimeError):
    """Raised when OCR is requested but no engine is usable on this host."""


class OcrEngine(ABC):
    @abstractmethod
    def is_available(self) -> bool:
        """Whether this engine can actually run on the current host."""

    @abstractmethod
    def image_to_text(self, image_png: bytes) -> str:
        """Recognize text in a PNG-encoded page image."""


class TesseractOcrEngine(OcrEngine):
    """OCR via Tesseract, using PyMuPDF's bundled binding.

    Requires the Tesseract binary and a TESSDATA_PREFIX pointing at language
    data; `is_available()` reports whether both are present so callers can
    degrade gracefully rather than crash mid-ingestion.
    """

    def __init__(self, language: str = "eng") -> None:
        self.language = language

    def is_available(self) -> bool:
        try:
            pymupdf.get_tessdata()
        except Exception:  # noqa: BLE001 - any failure means unusable
            return False
        return True

    def image_to_text(self, image_png: bytes) -> str:
        if not self.is_available():
            raise OcrUnavailableError(
                "Tesseract is not installed or TESSDATA_PREFIX is unset. "
                "Install Tesseract, or set OCR_ENABLED=false to skip scanned pages."
            )

        pixmap = pymupdf.Pixmap(image_png)
        try:
            pdf_bytes = pixmap.pdfocr_tobytes(language=self.language)
        finally:
            pixmap = None  # noqa: F841 - release the pixmap's buffer promptly

        # pdfocr_tobytes returns a single-page PDF carrying an invisible text
        # layer; reading that back is how we get the recognized characters.
        with pymupdf.open(stream=pdf_bytes, filetype="pdf") as document:
            return "\n".join(page.get_text("text") for page in document)


class NullOcrEngine(OcrEngine):
    """Engine used when OCR is disabled: never available, never called."""

    def is_available(self) -> bool:
        return False

    def image_to_text(self, image_png: bytes) -> str:
        raise OcrUnavailableError("OCR is disabled (OCR_ENABLED=false)")


def get_ocr_engine() -> OcrEngine:
    from app.core.config import get_settings

    settings = get_settings()
    if not settings.ocr_enabled:
        return NullOcrEngine()
    return TesseractOcrEngine(language=settings.ocr_language)


def render_page_png(page: pymupdf.Page, dpi: int) -> bytes:
    """Rasterize a page for OCR at the configured resolution."""
    return page.get_pixmap(dpi=dpi).tobytes("png")


def ocr_document(
    document: pymupdf.Document,
    page_texts: list[str],
    *,
    engine: OcrEngine | None = None,
    dpi: int = 300,
) -> tuple[list[str], int]:
    """Fill in text for pages that lack a usable text layer.

    Returns the (possibly updated) page texts and how many pages were OCR'd.
    Pages that already have text are left untouched, so a mixed document —
    digital body with scanned appendix scans — only pays for the scanned part.
    """
    engine = engine or get_ocr_engine()

    needs_ocr = [
        index
        for index, text in enumerate(page_texts)
        if len(text.strip()) < MIN_CHARS_FOR_TEXT_LAYER
    ]
    if not needs_ocr:
        return page_texts, 0

    if not engine.is_available():
        logger.warning("ocr_skipped_engine_unavailable", pages=len(needs_ocr))
        return page_texts, 0

    updated = list(page_texts)
    recognized = 0
    for index in needs_ocr:
        try:
            image = render_page_png(document[index], dpi)
            text = engine.image_to_text(image)
        except (OcrUnavailableError, RuntimeError, ValueError) as exc:
            # One unreadable page shouldn't abort ingestion of the rest.
            logger.warning("ocr_page_failed", page=index + 1, error=str(exc))
            continue

        if text.strip():
            updated[index] = text
            recognized += 1

    logger.info("ocr_completed", pages_attempted=len(needs_ocr), pages_recognized=recognized)
    return updated, recognized
