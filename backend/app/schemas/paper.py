import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict

from app.models.paper import PaperStatus
from app.models.paper_asset import AssetKind


class PaperRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    workspace_id: uuid.UUID
    title: str | None
    status: PaperStatus
    original_filename: str
    content_type: str
    size_bytes: int
    checksum: str
    page_count: int | None
    abstract: str | None
    authors: str | None
    error_message: str | None
    ocr_page_count: int
    created_at: datetime
    updated_at: datetime


class PaperList(BaseModel):
    items: list[PaperRead]
    total: int


class PaperAssetRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    kind: AssetKind
    page_number: int
    caption: str | None
    content: str | None
    storage_key: str | None
