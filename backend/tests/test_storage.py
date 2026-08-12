import io

import pytest

from app.core.storage import (
    LocalStorageBackend,
    build_storage_key,
    compute_checksum,
)


def test_compute_checksum_is_stable_and_rewinds_stream() -> None:
    source = io.BytesIO(b"hello world")

    checksum, size = compute_checksum(source)

    assert size == 11
    assert checksum == compute_checksum(io.BytesIO(b"hello world"))[0]
    # The caller must be able to read the stream afterwards.
    assert source.read() == b"hello world"


def test_build_storage_key_shards_by_checksum_and_keeps_extension() -> None:
    key = build_storage_key("abcdef1234567890", "Paper Draft.PDF")

    assert key == "papers/ab/cd/abcdef1234567890.pdf"


def test_save_open_delete_round_trip(tmp_path) -> None:  # noqa: ANN001
    storage = LocalStorageBackend(tmp_path)

    written = storage.save("papers/ab/cd/file.pdf", io.BytesIO(b"content"))

    assert written == 7
    assert storage.exists("papers/ab/cd/file.pdf")
    with storage.open("papers/ab/cd/file.pdf") as handle:
        assert handle.read() == b"content"

    storage.delete("papers/ab/cd/file.pdf")
    assert not storage.exists("papers/ab/cd/file.pdf")


def test_deleting_missing_key_is_not_an_error(tmp_path) -> None:  # noqa: ANN001
    LocalStorageBackend(tmp_path).delete("papers/does/not/exist.pdf")


def test_keys_cannot_escape_the_storage_root(tmp_path) -> None:  # noqa: ANN001
    storage = LocalStorageBackend(tmp_path / "root")

    with pytest.raises(ValueError, match="escapes the storage root"):
        storage.save("../../escaped.pdf", io.BytesIO(b"malicious"))

    assert not (tmp_path / "escaped.pdf").exists()
