"""Blob storage abstraction for uploaded papers and extracted assets.

Only the interface is consumed by services, so the local-filesystem backend
used in development can be swapped for S3/MinIO in production without touching
ingestion code.
"""

from __future__ import annotations

import hashlib
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import BinaryIO

CHUNK_SIZE = 1024 * 1024


class StorageBackend(ABC):
    """Content-addressed blob storage."""

    @abstractmethod
    def save(self, key: str, source: BinaryIO) -> int:
        """Persist `source` under `key`, returning the number of bytes written."""

    @abstractmethod
    def open(self, key: str) -> BinaryIO:
        """Open a stored blob for reading."""

    @abstractmethod
    def delete(self, key: str) -> None:
        """Remove a stored blob. Missing keys are not an error."""

    @abstractmethod
    def exists(self, key: str) -> bool: ...


class LocalStorageBackend(StorageBackend):
    """Stores blobs on the local filesystem, rooted at `base_path`."""

    def __init__(self, base_path: str | Path) -> None:
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _resolve(self, key: str) -> Path:
        # Resolve and confirm containment so a crafted key ("../../etc/passwd")
        # can't escape the storage root.
        target = (self.base_path / key).resolve()
        root = self.base_path.resolve()
        if not target.is_relative_to(root):
            raise ValueError(f"Storage key escapes the storage root: {key!r}")
        return target

    def save(self, key: str, source: BinaryIO) -> int:
        target = self._resolve(key)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as destination:
            shutil.copyfileobj(source, destination, CHUNK_SIZE)
        return target.stat().st_size

    def open(self, key: str) -> BinaryIO:
        return self._resolve(key).open("rb")

    def delete(self, key: str) -> None:
        self._resolve(key).unlink(missing_ok=True)

    def exists(self, key: str) -> bool:
        return self._resolve(key).is_file()


def compute_checksum(source: BinaryIO) -> tuple[str, int]:
    """Return (sha256_hex, byte_size) for `source`, leaving it rewound."""
    digest = hashlib.sha256()
    size = 0
    source.seek(0)
    while chunk := source.read(CHUNK_SIZE):
        digest.update(chunk)
        size += len(chunk)
    source.seek(0)
    return digest.hexdigest(), size


def build_storage_key(checksum: str, filename: str) -> str:
    """Content-addressed key, sharded by checksum prefix.

    Sharding keeps any single directory from accumulating tens of thousands of
    entries, which degrades listing performance on most filesystems.
    """
    suffix = Path(filename).suffix.lower() or ".bin"
    return f"papers/{checksum[:2]}/{checksum[2:4]}/{checksum}{suffix}"


def get_storage() -> StorageBackend:
    from app.core.config import get_settings

    return LocalStorageBackend(get_settings().storage_local_path)
