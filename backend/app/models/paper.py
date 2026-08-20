import enum
import uuid
from typing import TYPE_CHECKING

from sqlalchemy import BigInteger, Enum, ForeignKey, Index, Integer, String, Text, Uuid
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base
from app.models.mixins import IdMixin, TimestampMixin

if TYPE_CHECKING:
    from app.models.document_chunk import DocumentChunk
    from app.models.experiment_result import ExperimentResult
    from app.models.extracted_entity import ExtractedEntity
    from app.models.paper_asset import PaperAsset
    from app.models.paper_reference import PaperReference
    from app.models.user import User
    from app.models.workspace import Workspace


class PaperStatus(str, enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


class Paper(IdMixin, TimestampMixin, Base):
    __tablename__ = "papers"
    __table_args__ = (
        # Uploading the same file twice into one workspace should be rejected
        # rather than silently duplicated; enforced in the DB so concurrent
        # uploads can't race past a service-layer check.
        Index("uq_papers_workspace_checksum", "workspace_id", "checksum", unique=True),
    )

    workspace_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("workspaces.id"), nullable=False
    )
    uploaded_by_id: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("users.id"), nullable=False)
    title: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    status: Mapped[PaperStatus] = mapped_column(
        Enum(PaperStatus, name="paper_status"), default=PaperStatus.PENDING, nullable=False
    )

    # --- Source file ---
    original_filename: Mapped[str] = mapped_column(String(512), nullable=False)
    content_type: Mapped[str] = mapped_column(String(128), nullable=False)
    checksum: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    storage_key: Mapped[str] = mapped_column(String(512), nullable=False)

    # --- Extraction results (populated by the ingestion pipeline) ---
    page_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    abstract: Mapped[str | None] = mapped_column(Text, nullable=True)
    authors: Mapped[str | None] = mapped_column(Text, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    ocr_page_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    published_year: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    venue: Mapped[str | None] = mapped_column(String(255), nullable=True)

    workspace: Mapped["Workspace"] = relationship(back_populates="papers")
    uploaded_by: Mapped["User"] = relationship(back_populates="papers")
    assets: Mapped[list["PaperAsset"]] = relationship(
        back_populates="paper", cascade="all, delete-orphan"
    )
    references: Mapped[list["PaperReference"]] = relationship(
        back_populates="paper", cascade="all, delete-orphan"
    )
    chunks: Mapped[list["DocumentChunk"]] = relationship(
        back_populates="paper", cascade="all, delete-orphan"
    )
    entities: Mapped[list["ExtractedEntity"]] = relationship(
        back_populates="paper", cascade="all, delete-orphan"
    )
    experiments: Mapped[list["ExperimentResult"]] = relationship(
        back_populates="paper", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"Paper(id={self.id!r}, title={self.title!r}, status={self.status!r})"
