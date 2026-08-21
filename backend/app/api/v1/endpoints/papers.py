import uuid

from fastapi import APIRouter, Depends, File, Query, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.v1.deps import get_current_active_user
from app.core.exceptions import ValidationError
from app.db.session import get_db
from app.models import AssetKind, Paper, PaperAsset, User
from app.schemas.paper import PaperAssetRead, PaperList, PaperRead
from app.services.paper_service import (
    delete_paper,
    get_paper,
    list_paper_assets,
    list_papers,
    upload_paper,
)
from app.worker.tasks import process_paper

router = APIRouter(prefix="/papers", tags=["papers"])


@router.post("", response_model=PaperRead, status_code=status.HTTP_201_CREATED)
async def upload(
    file: UploadFile = File(...),
    user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> Paper:
    if not file.filename:
        raise ValidationError("Upload is missing a filename")

    paper = await upload_paper(
        db,
        user,
        file=file.file,
        filename=file.filename,
        content_type=file.content_type or "application/octet-stream",
    )

    # Parsing is slow and IO-bound, so the request returns as soon as the file
    # is durably stored; clients poll `status` for progress.
    process_paper.delay(str(paper.id))

    # Re-read before serializing. With a real broker the row is still pending
    # and this is a no-op, but in eager mode the task has already finished and
    # the in-memory object would report a status that is no longer true.
    await db.refresh(paper)
    return paper


@router.get("", response_model=PaperList)
async def list_all(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> PaperList:
    items, total = await list_papers(db, user, limit=limit, offset=offset)
    return PaperList(items=[PaperRead.model_validate(item) for item in items], total=total)


@router.get("/{paper_id}", response_model=PaperRead)
async def get_one(
    paper_id: uuid.UUID,
    user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> Paper:
    return await get_paper(db, user, paper_id)


@router.delete("/{paper_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_one(
    paper_id: uuid.UUID,
    user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> None:
    await delete_paper(db, user, paper_id)


@router.get("/{paper_id}/assets", response_model=list[PaperAssetRead])
async def list_assets(
    paper_id: uuid.UUID,
    kind: AssetKind | None = Query(default=None),
    user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> list[PaperAsset]:
    """Tables, figures, and equations extracted from a paper."""
    return await list_paper_assets(db, user, paper_id, kind=kind)
