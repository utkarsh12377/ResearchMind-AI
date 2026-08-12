"""Paper upload, listing, and ingestion-state transitions."""

from __future__ import annotations

import uuid
from typing import BinaryIO

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.exceptions import ConflictError, NotFoundError, ValidationError
from app.core.storage import build_storage_key, compute_checksum, get_storage
from app.models import Paper, PaperStatus, User, Workspace

ALLOWED_CONTENT_TYPES = {"application/pdf"}


async def get_default_workspace(db: AsyncSession, user: User) -> Workspace:
    workspace = await db.scalar(
        select(Workspace).where(Workspace.owner_id == user.id).order_by(Workspace.created_at)
    )
    if workspace is None:
        raise NotFoundError("No workspace found for this user")
    return workspace


async def upload_paper(
    db: AsyncSession,
    user: User,
    *,
    file: BinaryIO,
    filename: str,
    content_type: str,
    workspace: Workspace | None = None,
) -> Paper:
    if content_type not in ALLOWED_CONTENT_TYPES:
        raise ValidationError(f"Unsupported content type {content_type!r}; expected a PDF")

    checksum, size_bytes = compute_checksum(file)

    if size_bytes == 0:
        raise ValidationError("Uploaded file is empty")

    max_bytes = get_settings().max_upload_bytes
    if size_bytes > max_bytes:
        raise ValidationError(f"File exceeds the maximum upload size of {max_bytes} bytes")

    workspace = workspace or await get_default_workspace(db, user)

    storage_key = build_storage_key(checksum, filename)
    # Content-addressed, so an identical blob already present is the same
    # bytes; re-writing it would be wasted I/O.
    storage = get_storage()
    if not storage.exists(storage_key):
        storage.save(storage_key, file)

    paper = Paper(
        workspace_id=workspace.id,
        uploaded_by_id=user.id,
        original_filename=filename,
        content_type=content_type,
        checksum=checksum,
        size_bytes=size_bytes,
        storage_key=storage_key,
        status=PaperStatus.PENDING,
    )
    db.add(paper)

    try:
        await db.commit()
    except IntegrityError as exc:
        await db.rollback()
        raise ConflictError("This paper has already been uploaded to the workspace") from exc

    await db.refresh(paper)
    return paper


async def list_papers(
    db: AsyncSession, user: User, *, limit: int = 50, offset: int = 0
) -> tuple[list[Paper], int]:
    owned_workspaces = select(Workspace.id).where(Workspace.owner_id == user.id)

    total = await db.scalar(
        select(func.count()).select_from(Paper).where(Paper.workspace_id.in_(owned_workspaces))
    )
    result = await db.scalars(
        select(Paper)
        .where(Paper.workspace_id.in_(owned_workspaces))
        .order_by(Paper.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    return list(result), total or 0


async def get_paper(db: AsyncSession, user: User, paper_id: uuid.UUID) -> Paper:
    owned_workspaces = select(Workspace.id).where(Workspace.owner_id == user.id)
    paper = await db.scalar(
        select(Paper).where(Paper.id == paper_id, Paper.workspace_id.in_(owned_workspaces))
    )
    if paper is None:
        # Deliberately 404 rather than 403 for papers owned by someone else, so
        # the API doesn't confirm the existence of other users' documents.
        raise NotFoundError("Paper not found")
    return paper


async def delete_paper(db: AsyncSession, user: User, paper_id: uuid.UUID) -> None:
    paper = await get_paper(db, user, paper_id)
    storage_key = paper.checksum

    await db.delete(paper)
    await db.commit()

    # Blobs are content-addressed and shared across workspaces, so only remove
    # the file once no paper anywhere still references it.
    still_referenced = await db.scalar(
        select(func.count()).select_from(Paper).where(Paper.checksum == storage_key)
    )
    if not still_referenced:
        get_storage().delete(paper.storage_key)
