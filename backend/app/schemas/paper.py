import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict

from app.models.paper import PaperStatus


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
    created_at: datetime
    updated_at: datetime


class PaperList(BaseModel):
    items: list[PaperRead]
    total: int
