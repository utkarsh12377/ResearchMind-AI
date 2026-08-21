import io

import pytest
from httpx import AsyncClient

from app.core.validation import (
    UnsupportedFileError,
    assert_pdf,
    clamp_text,
    sanitize_filename,
)


@pytest.fixture(autouse=True)
def _no_celery_dispatch(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setattr(
        "app.api.v1.endpoints.papers.process_paper.delay", lambda paper_id: None
    )


# --- Magic bytes -----------------------------------------------------------


def test_a_real_pdf_passes() -> None:
    assert_pdf(io.BytesIO(b"%PDF-1.7\nrest of the file"))


def test_a_renamed_file_is_rejected() -> None:
    """Content-Type is a claim the client makes; the header is the file."""
    with pytest.raises(UnsupportedFileError):
        assert_pdf(io.BytesIO(b"MZ\x90\x00 this is an executable"))


def test_an_empty_stream_is_rejected() -> None:
    with pytest.raises(UnsupportedFileError):
        assert_pdf(io.BytesIO(b""))


def test_the_stream_position_is_restored() -> None:
    """The caller checksums the same stream afterwards, so it must not move."""
    stream = io.BytesIO(b"%PDF-1.7\nbody")
    stream.seek(0)

    assert_pdf(stream)

    assert stream.tell() == 0
    assert stream.read().startswith(b"%PDF")


def test_a_non_zero_start_position_is_preserved() -> None:
    stream = io.BytesIO(b"XX%PDF-1.7")
    stream.seek(2)

    assert_pdf(stream)

    assert stream.tell() == 2


# --- Filenames -------------------------------------------------------------


def test_a_plain_filename_is_untouched() -> None:
    assert sanitize_filename("attention-is-all-you-need.pdf") == "attention-is-all-you-need.pdf"


def test_directory_traversal_is_stripped() -> None:
    assert sanitize_filename("../../../etc/passwd") == "passwd"
    assert sanitize_filename("..\\..\\windows\\system32\\config") == "config"


def test_a_fullwidth_separator_cannot_smuggle_a_path() -> None:
    """NFKC runs first, so a lookalike solidus normalizes before filtering."""
    assert "/" not in sanitize_filename("a／b／c.pdf")


def test_control_characters_are_replaced() -> None:
    cleaned = sanitize_filename("report\n\r\x00.pdf")

    assert "\n" not in cleaned
    assert "\x00" not in cleaned


def test_windows_reserved_names_are_defused() -> None:
    """A file named CON is not merely awkward on Windows, it is undeletable."""
    assert sanitize_filename("CON.pdf").startswith("file_")
    assert sanitize_filename("lpt1.pdf").startswith("file_")


def test_an_empty_name_falls_back() -> None:
    assert sanitize_filename("") == "upload.pdf"
    assert sanitize_filename("   ") == "upload.pdf"
    assert sanitize_filename("...") == "upload.pdf"


def test_a_long_name_keeps_its_extension() -> None:
    cleaned = sanitize_filename("a" * 400 + ".pdf")

    assert len(cleaned) <= 255
    assert cleaned.endswith(".pdf")


def test_repeated_dots_are_collapsed() -> None:
    assert sanitize_filename("weird....name.pdf") == "weird.name.pdf"


# --- Text clamping ---------------------------------------------------------


def test_clamp_text_bounds_and_trims() -> None:
    assert clamp_text("  hello  ", 10) == "hello"
    assert clamp_text("x" * 50, 10) == "x" * 10


def test_clamp_text_maps_blank_to_none() -> None:
    assert clamp_text("   ", 10) is None
    assert clamp_text(None, 10) is None


# --- Upload endpoint -------------------------------------------------------


async def _auth_headers(client: AsyncClient, email: str = "upload@example.com") -> dict:
    await client.post(
        "/api/v1/auth/register", json={"email": email, "password": "supersecret123"}
    )
    login = await client.post(
        "/api/v1/auth/login", data={"username": email, "password": "supersecret123"}
    )
    return {"Authorization": f"Bearer {login.json()['access_token']}"}


@pytest.mark.asyncio
async def test_a_disguised_file_is_refused_by_the_endpoint(client: AsyncClient) -> None:
    headers = await _auth_headers(client)

    response = await client.post(
        "/api/v1/papers",
        files={"file": ("notes.pdf", b"MZ\x90\x00 not a pdf", "application/pdf")},
        headers=headers,
    )

    assert response.status_code == 422
    assert "not a PDF" in response.json()["error"]["message"]


@pytest.mark.asyncio
async def test_a_traversing_filename_is_stored_sanitized(client: AsyncClient) -> None:
    from tests.factories import build_pdf

    headers = await _auth_headers(client)

    response = await client.post(
        "/api/v1/papers",
        files={
            "file": ("../../etc/passwd.pdf", build_pdf().getvalue(), "application/pdf")
        },
        headers=headers,
    )

    assert response.status_code == 201
    assert response.json()["original_filename"] == "passwd.pdf"
