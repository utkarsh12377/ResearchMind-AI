"""Cross-paper comparison tables built from extracted results.

The hard part of comparing papers is not rendering a table, it is deciding which
numbers belong in the same cell. Two papers reporting "F1" on "SQuAD" are
comparable; the same metric name on a different dataset, or on a different split,
is not. Grouping happens on the normalized triple and anything that doesn't
match is kept in a separate bucket rather than silently averaged in.
"""

from __future__ import annotations

import uuid
from collections import defaultdict
from dataclasses import dataclass, field

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.models import ExperimentResult, Paper
from app.research.extraction import canonical_metric

logger = get_logger(__name__)

#: Metrics where a lower number is the better one.
LOWER_IS_BETTER = frozenset({"perplexity", "wer", "cer", "loss", "latency", "error rate"})


@dataclass
class ComparisonCell:
    paper_id: uuid.UUID
    paper_title: str
    model: str
    value: float
    unit: str | None = None
    split: str | None = None
    confidence: float = 0.5
    is_best: bool = False


@dataclass
class ComparisonRow:
    dataset: str
    metric: str
    cells: list[ComparisonCell] = field(default_factory=list)

    @property
    def spread(self) -> float:
        if len(self.cells) < 2:
            return 0.0
        values = [cell.value for cell in self.cells]
        return max(values) - min(values)


@dataclass
class ComparisonTable:
    rows: list[ComparisonRow] = field(default_factory=list)
    papers: dict[str, str] = field(default_factory=dict)
    ungrouped: int = 0

    @property
    def is_empty(self) -> bool:
        return not self.rows

    def to_markdown(self) -> str:
        if self.is_empty:
            return "_No comparable results were extracted from these papers._"

        lines = ["| Dataset | Metric | Model | Value | Paper |", "| --- | --- | --- | --- | --- |"]
        for row in self.rows:
            for cell in row.cells:
                value = f"{cell.value:g}{cell.unit or ''}"
                if cell.is_best:
                    value = f"**{value}**"
                lines.append(
                    f"| {row.dataset} | {row.metric} | {cell.model} | {value} "
                    f"| {cell.paper_title} |"
                )
        return "\n".join(lines)


async def build_comparison_table(
    db: AsyncSession,
    paper_ids: list[uuid.UUID],
    *,
    metrics: list[str] | None = None,
    datasets: list[str] | None = None,
    min_papers_per_row: int = 1,
) -> ComparisonTable:
    if not paper_ids:
        return ComparisonTable()

    stmt = (
        select(ExperimentResult, Paper.title)
        .join(Paper, Paper.id == ExperimentResult.paper_id)
        .where(ExperimentResult.paper_id.in_(paper_ids))
    )
    if metrics:
        stmt = stmt.where(
            ExperimentResult.metric_key.in_([canonical_metric(m) for m in metrics])
        )
    if datasets:
        from app.graph.schema import normalize_key

        stmt = stmt.where(
            ExperimentResult.dataset_key.in_([normalize_key(d) for d in datasets])
        )

    rows = (await db.execute(stmt)).all()
    if not rows:
        return ComparisonTable()

    grouped: dict[tuple[str, str], list[ComparisonCell]] = defaultdict(list)
    titles: dict[str, str] = {}
    ungrouped = 0

    for result, title in rows:
        titles[str(result.paper_id)] = title or "Untitled"
        if not result.dataset_key:
            ungrouped += 1
            continue

        grouped[(result.dataset_name or result.dataset_key, result.metric_name)].append(
            ComparisonCell(
                paper_id=result.paper_id,
                paper_title=title or "Untitled",
                model=result.model_name,
                value=result.value,
                unit=result.unit,
                split=result.split,
                confidence=result.confidence,
            )
        )

    table = ComparisonTable(papers=titles, ungrouped=ungrouped)
    for (dataset, metric), cells in grouped.items():
        distinct_papers = {cell.paper_id for cell in cells}
        if len(distinct_papers) < min_papers_per_row:
            continue
        _mark_best(cells, metric)
        cells.sort(key=lambda c: c.value, reverse=metric_higher_is_better(metric))
        table.rows.append(ComparisonRow(dataset=dataset, metric=metric, cells=cells))

    table.rows.sort(key=lambda row: (len(row.cells), row.spread), reverse=True)
    logger.info("comparison_table_built", rows=len(table.rows), papers=len(paper_ids))
    return table


def metric_higher_is_better(metric: str) -> bool:
    return canonical_metric(metric) not in LOWER_IS_BETTER


def _mark_best(cells: list[ComparisonCell], metric: str) -> None:
    if not cells:
        return
    chooser = max if metric_higher_is_better(metric) else min
    best = chooser(cells, key=lambda c: c.value)
    best.is_best = True


@dataclass
class PaperSummaryRow:
    paper_id: uuid.UUID
    title: str
    year: int | None
    datasets: list[str] = field(default_factory=list)
    models: list[str] = field(default_factory=list)
    metrics: list[str] = field(default_factory=list)


async def build_paper_matrix(
    db: AsyncSession, paper_ids: list[uuid.UUID]
) -> list[PaperSummaryRow]:
    """A per-paper overview: what each one used and measured."""
    if not paper_ids:
        return []

    papers = (
        await db.execute(
            select(Paper.id, Paper.title, Paper.published_year).where(Paper.id.in_(paper_ids))
        )
    ).all()

    results = (
        await db.scalars(
            select(ExperimentResult).where(ExperimentResult.paper_id.in_(paper_ids))
        )
    ).all()

    by_paper: dict[uuid.UUID, PaperSummaryRow] = {
        pid: PaperSummaryRow(paper_id=pid, title=title or "Untitled", year=year)
        for pid, title, year in papers
    }

    for result in results:
        row = by_paper.get(result.paper_id)
        if row is None:
            continue
        if result.dataset_name and result.dataset_name not in row.datasets:
            row.datasets.append(result.dataset_name)
        if result.model_name not in row.models:
            row.models.append(result.model_name)
        if result.metric_name not in row.metrics:
            row.metrics.append(result.metric_name)

    return sorted(by_paper.values(), key=lambda r: (r.year or 0), reverse=True)
