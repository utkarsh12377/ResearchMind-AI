"""Input validation for values that arrive from outside the system.

Everything here treats the client as untrusted, including the parts of an upload
that look like metadata. A declared Content-Type is a claim, not a fact, and a
filename is attacker-controlled text that ends up in a storage path, a log line,
and eventually a UI.
"""

from __future__ import annotations

import re
import unicodedata
from typing import BinaryIO

from app.core.logging import get_logger

logger = get_logger(__name__)

#: Every PDF starts with this. Checking it costs five bytes and closes the gap
#: between "the client said it was a PDF" and "it is a PDF".
PDF_MAGIC = b"%PDF-"
MAGIC_READ_BYTES = 8

MAX_FILENAME_LENGTH = 255
UNSAFE_FILENAME_CHARS = re.compile(r'[\x00-\x1f\x7f<>:"/\\|?*]')
COLLAPSE_DOTS = re.compile(r"\.{2,}")


class UnsupportedFileError(ValueError):
    """Raised when an upload is not the file type it claims to be."""


def assert_pdf(stream: BinaryIO) -> None:
    """Verify the magic bytes, leaving the stream where it was found."""
    position = stream.tell()
    try:
        header = stream.read(MAGIC_READ_BYTES)
    finally:
        stream.seek(position)

    if not header.startswith(PDF_MAGIC):
        logger.warning("upload_magic_mismatch", header=header[:8].hex())
        raise UnsupportedFileError(
            "That file is not a PDF. The content type can be set by the client, "
            "so the file itself is checked."
        )


def sanitize_filename(name: str, *, fallback: str = "upload.pdf") -> str:
    """Reduce a client-supplied filename to something safe to store and display.

    Handles the three things that actually go wrong: directory traversal via
    separators or "..", control characters that corrupt log output, and Windows
    reserved device names, which make a file undeletable rather than merely
    awkward.
    """
    if not name or not name.strip():
        return fallback

    # NFKC first, so a fullwidth solidus cannot smuggle a separator past the
    # character filter below.
    normalized = unicodedata.normalize("NFKC", name)
    basename = normalized.replace("\\", "/").rsplit("/", 1)[-1]
    cleaned = UNSAFE_FILENAME_CHARS.sub("_", basename)
    cleaned = COLLAPSE_DOTS.sub(".", cleaned).strip(" .")

    if not cleaned:
        return fallback

    stem, _, extension = cleaned.rpartition(".")
    if stem and _is_reserved(stem):
        cleaned = f"file_{cleaned}"

    if len(cleaned) > MAX_FILENAME_LENGTH:
        # Truncate the stem rather than the extension: the extension is what
        # decides how the file is handled later.
        keep = MAX_FILENAME_LENGTH - len(extension) - 1
        cleaned = f"{cleaned[:keep]}.{extension}" if extension else cleaned[:MAX_FILENAME_LENGTH]

    return cleaned


WINDOWS_RESERVED = frozenset(
    {"con", "prn", "aux", "nul"}
    | {f"com{i}" for i in range(1, 10)}
    | {f"lpt{i}" for i in range(1, 10)}
)


def _is_reserved(stem: str) -> bool:
    return stem.lower() in WINDOWS_RESERVED


def clamp_text(value: str | None, limit: int) -> str | None:
    """Bound free text before it reaches a column or a prompt."""
    if value is None:
        return None
    stripped = value.strip()
    return stripped[:limit] if stripped else None
