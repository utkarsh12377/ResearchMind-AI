import uuid
from typing import TYPE_CHECKING

from sqlalchemy import Boolean, ForeignKey, Index, Integer, String, Text, Uuid
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base
from app.models.mixins import IdMixin, TimestampMixin

if TYPE_CHECKING:
    from app.models.paper import Paper


class DocumentChunk(IdMixin, TimestampMixin, Base):
    """A retrievable passage of a paper.

    Chunks are the unit of both dense and sparse retrieval. The row stays the
    system of record even after embedding: the vector store holds only the
    embedding plus this chunk's id, so text, section, and page can be resolved
    for citation without duplicating content into the index.
    """

    __tablename__ = "document_chunks"
    __table_args__ = (
        Index("ix_document_chunks_paper_index", "paper_id", "chunk_index", unique=True),
        # Re-embedding after a model change scans for stale chunks.
        Index("ix_document_chunks_embedded", "is_embedded"),
    )

    paper_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("papers.id", ondelete="CASCADE"), nullable=False
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    section_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    page_number: Mapped[int | None] = mapped_column(Integer, nullable=True)
    kind: Mapped[str] = mapped_column(String(32), default="text", nullable=False)
    token_estimate: Mapped[int] = mapped_column(Integer, nullable=False)
    is_embedded: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    paper: Mapped["Paper"] = relationship(back_populates="chunks")

    def __repr__(self) -> str:
        return f"DocumentChunk(paper_id={self.paper_id!r}, index={self.chunk_index!r})"
