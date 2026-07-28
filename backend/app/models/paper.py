import enum
import uuid
from typing import TYPE_CHECKING

from sqlalchemy import Enum, ForeignKey, String, Uuid
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base
from app.models.mixins import IdMixin, TimestampMixin

if TYPE_CHECKING:
    from app.models.user import User
    from app.models.workspace import Workspace


class PaperStatus(str, enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


class Paper(IdMixin, TimestampMixin, Base):
    __tablename__ = "papers"

    workspace_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("workspaces.id"), nullable=False
    )
    uploaded_by_id: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("users.id"), nullable=False)
    title: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    status: Mapped[PaperStatus] = mapped_column(
        Enum(PaperStatus, name="paper_status"), default=PaperStatus.PENDING, nullable=False
    )

    workspace: Mapped["Workspace"] = relationship(back_populates="papers")
    uploaded_by: Mapped["User"] = relationship(back_populates="papers")

    def __repr__(self) -> str:
        return f"Paper(id={self.id!r}, title={self.title!r}, status={self.status!r})"
