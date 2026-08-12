"""OCR tests.

Tesseract is an external binary that may not be installed, so these drive the
engine interface with a fake implementation. That is the point of the
abstraction: the orchestration logic (which pages to OCR, how failures degrade)
is what carries the risk, and it is fully exercised here.
"""

import pymupdf
import pytest

from app.ingestion.ocr import (
    NullOcrEngine,
    OcrEngine,
    OcrUnavailableError,
    TesseractOcrEngine,
    ocr_document,
)
from tests.factories import build_pdf, build_scanned_pdf


class FakeOcrEngine(OcrEngine):
    def __init__(self, text: str = "Recognized text from the scanned page.") -> None:
        self.text = text
        self.calls = 0

    def is_available(self) -> bool:
        return True

    def image_to_text(self, image_png: bytes) -> str:
        assert image_png.startswith(b"\x89PNG"), "engine should receive a PNG"
        self.calls += 1
        return self.text


class FlakyOcrEngine(FakeOcrEngine):
    def image_to_text(self, image_png: bytes) -> str:
        self.calls += 1
        raise RuntimeError("recognition failed")


def _open(pdf) -> pymupdf.Document:  # noqa: ANN001
    return pymupdf.open(stream=pdf.read(), filetype="pdf")


def test_ocr_fills_in_text_for_pages_without_a_text_layer() -> None:
    document = _open(build_scanned_pdf(pages=2))
    engine = FakeOcrEngine()

    texts, recognized = ocr_document(document, ["", ""], engine=engine)

    assert recognized == 2
    assert engine.calls == 2
    assert all("Recognized text" in text for text in texts)


def test_pages_that_already_have_text_are_not_ocred() -> None:
    document = _open(build_pdf(body="A" * 400))
    engine = FakeOcrEngine()
    existing = ["Existing text layer. " * 20]

    texts, recognized = ocr_document(document, existing, engine=engine)

    assert recognized == 0
    assert engine.calls == 0
    assert texts == existing


def test_mixed_document_only_ocrs_the_scanned_pages() -> None:
    document = _open(build_scanned_pdf(pages=2))
    engine = FakeOcrEngine()

    texts, recognized = ocr_document(document, ["Real text layer. " * 20, ""], engine=engine)

    assert recognized == 1
    assert engine.calls == 1
    assert texts[0].startswith("Real text layer.")
    assert "Recognized text" in texts[1]


def test_unavailable_engine_degrades_instead_of_failing() -> None:
    document = _open(build_scanned_pdf())

    texts, recognized = ocr_document(document, [""], engine=NullOcrEngine())

    assert recognized == 0
    assert texts == [""]


def test_a_failing_page_does_not_abort_the_rest() -> None:
    document = _open(build_scanned_pdf(pages=2))

    texts, recognized = ocr_document(document, ["", ""], engine=FlakyOcrEngine())

    assert recognized == 0
    assert texts == ["", ""]


def test_null_engine_raises_when_called_directly() -> None:
    with pytest.raises(OcrUnavailableError):
        NullOcrEngine().image_to_text(b"\x89PNG")


def test_tesseract_engine_reports_availability_without_crashing() -> None:
    # Must return a bool on any host, present or not — callers rely on this to
    # decide whether to attempt OCR at all.
    assert isinstance(TesseractOcrEngine().is_available(), bool)
