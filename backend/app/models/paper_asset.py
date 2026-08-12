import enum
import uuid
from typing import TYPE_CHECKING

from sqlalchemy import Enum, ForeignKey, Index, Integer, String, Text, Uuid
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base
from app.models.mixins import IdMixin, TimestampMixin

if TYPE_CHECKING:
    from app.models.paper import Paper


class AssetKind(str, enum.Enum):
    TABLE = "table"
    FIGURE = "figure"
    EQUATION = "equation"


class PaperAsset(IdMixin, TimestampMixin, Base):
    """A non-prose element extracted from a paper: a table, figure, or equation."""

    __tablename__ = "paper_assets"
    __table_args__ = (Index("ix_paper_assets_paper_kind", "paper_id", "kind"),)

    paper_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("papers.id", ondelete="CASCADE"), nullable=False
    )
    kind: Mapped[AssetKind] = mapped_column(Enum(AssetKind, name="asset_kind"), nullable=False)
    page_number: Mapped[int] = mapped_column(Integer, nullable=False)
    caption: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Markdown for tables, LaTeX for equations, null for figures (which carry
    # only an image plus caption).
    content: Mapped[str | None] = mapped_column(Text, nullable=True)
    storage_key: Mapped[str | None] = mapped_column(String(512), nullable=True)

    paper: Mapped["Paper"] = relationship(back_populates="assets")

    def __repr__(self) -> str:
        return f"PaperAsset(kind={self.kind!r}, page={self.page_number!r})"
