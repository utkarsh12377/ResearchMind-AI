import uuid
from typing import TYPE_CHECKING

from sqlalchemy import Float, ForeignKey, Index, String, Text, Uuid
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base
from app.models.mixins import IdMixin, TimestampMixin

if TYPE_CHECKING:
    from app.models.paper import Paper


class ExtractedEntity(IdMixin, TimestampMixin, Base):
    """A research entity found in a paper, mirrored out of the knowledge graph.

    The graph owns the relationships; this table owns the per-paper provenance.
    Keeping a relational copy means "which papers mention this dataset" stays a
    plain indexed query, and entity filters compose with the existing retrieval
    filters without a second database being reachable.
    """

    __tablename__ = "extracted_entities"
    __table_args__ = (
        Index("uq_entity_paper_label_key", "paper_id", "label", "key", unique=True),
        Index("ix_entity_label_key", "label", "key"),
    )

    paper_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("papers.id", ondelete="CASCADE"), nullable=False
    )
    label: Mapped[str] = mapped_column(String(32), nullable=False)
    key: Mapped[str] = mapped_column(String(255), nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, default=0.5, nullable=False)
    evidence: Mapped[str | None] = mapped_column(Text, nullable=True)

    paper: Mapped["Paper"] = relationship(back_populates="entities")

    def __repr__(self) -> str:
        return f"ExtractedEntity(label={self.label!r}, key={self.key!r})"
