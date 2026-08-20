import uuid
from typing import TYPE_CHECKING

from sqlalchemy import ForeignKey, Index, Integer, String, Text, Uuid
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base
from app.models.mixins import IdMixin, TimestampMixin

if TYPE_CHECKING:
    from app.models.paper import Paper


class PaperReference(IdMixin, TimestampMixin, Base):
    """One entry from a paper's bibliography."""

    __tablename__ = "paper_references"
    __table_args__ = (
        Index("ix_paper_references_paper_order", "paper_id", "order"),
        # DOI/arXiv lookups drive citation-graph linking in Milestone 26.
        Index("ix_paper_references_doi", "doi"),
        Index("ix_paper_references_arxiv_id", "arxiv_id"),
    )

    paper_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("papers.id", ondelete="CASCADE"), nullable=False
    )
    order: Mapped[int] = mapped_column(Integer, nullable=False)
    raw_text: Mapped[str] = mapped_column(Text, nullable=False)
    title: Mapped[str | None] = mapped_column(Text, nullable=True)
    doi: Mapped[str | None] = mapped_column(String(255), nullable=True)
    arxiv_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    year: Mapped[int | None] = mapped_column(Integer, nullable=True)

    paper: Mapped["Paper"] = relationship(back_populates="references")

    def __repr__(self) -> str:
        return f"PaperReference(order={self.order!r}, title={self.title!r})"
