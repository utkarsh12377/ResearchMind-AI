import uuid
from typing import TYPE_CHECKING

from sqlalchemy import Float, ForeignKey, Index, String, Text, Uuid
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base
from app.models.mixins import IdMixin, TimestampMixin

if TYPE_CHECKING:
    from app.models.paper import Paper


class ExperimentResult(IdMixin, TimestampMixin, Base):
    """One reported number: a model, evaluated on a dataset, under a metric.

    This is the row that makes cross-paper comparison possible. Storing the
    quadruple explicitly — rather than leaving the numbers inside prose — is
    what lets the comparison and trend features work without re-reading every
    paper each time a question is asked.
    """

    __tablename__ = "experiment_results"
    __table_args__ = (
        Index("ix_experiment_paper", "paper_id"),
        Index("ix_experiment_lookup", "dataset_key", "metric_key"),
    )

    paper_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("papers.id", ondelete="CASCADE"), nullable=False
    )
    model_name: Mapped[str] = mapped_column(String(255), nullable=False)
    model_key: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    dataset_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    dataset_key: Mapped[str | None] = mapped_column(String(255), nullable=True)
    task_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    metric_name: Mapped[str] = mapped_column(String(128), nullable=False)
    metric_key: Mapped[str] = mapped_column(String(128), nullable=False)
    value: Mapped[float] = mapped_column(Float, nullable=False)
    unit: Mapped[str | None] = mapped_column(String(32), nullable=True)
    split: Mapped[str | None] = mapped_column(String(64), nullable=True)
    confidence: Mapped[float] = mapped_column(Float, default=0.5, nullable=False)
    evidence: Mapped[str | None] = mapped_column(Text, nullable=True)

    paper: Mapped["Paper"] = relationship(back_populates="experiments")

    def __repr__(self) -> str:
        return (
            f"ExperimentResult(model={self.model_name!r}, dataset={self.dataset_name!r}, "
            f"{self.metric_name}={self.value})"
        )
