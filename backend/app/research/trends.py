"""Trend analysis and timelines over a corpus.

All of this is computed from extracted rows rather than from text, so a trend is
a query rather than a summarisation. The model only gets involved at the last
step, to describe a series that has already been measured — which keeps it from
inventing a narrative the data doesn't support.
"""

from __future__ import annotations

import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.llm.gateway import LLMGateway, Message, Role
from app.llm.prompts import SYNTHESIZE_TREND
from app.models import ExperimentResult, ExtractedEntity, Paper

logger = get_logger(__name__)

MIN_YEARS_FOR_TREND = 2
EMERGING_WINDOW = 2


@dataclass
class YearBucket:
    year: int
    paper_count: int
    titles: list[str] = field(default_factory=list)


@dataclass
class Timeline:
    buckets: list[YearBucket] = field(default_factory=list)
    undated_papers: int = 0

    @property
    def span(self) -> tuple[int, int] | None:
        if not self.buckets:
            return None
        return self.buckets[0].year, self.buckets[-1].year

    @property
    def total_papers(self) -> int:
        return sum(bucket.paper_count for bucket in self.buckets) + self.undated_papers


@dataclass
class EntityTrend:
    label: str
    name: str
    by_year: dict[int, int]
    total: int

    @property
    def first_seen(self) -> int | None:
        return min(self.by_year) if self.by_year else None

    @property
    def direction(self) -> str:
        """Whether mentions are climbing, fading, or flat.

        Compares the most recent window against everything before it. A single
        year of data is reported as "new" rather than "rising", because one
        point is not a direction.
        """
        if len(self.by_year) < MIN_YEARS_FOR_TREND:
            return "new"

        years = sorted(self.by_year)
        split = years[-EMERGING_WINDOW:]
        recent = sum(self.by_year[y] for y in split)
        earlier = sum(self.by_year[y] for y in years if y not in split)

        if earlier == 0:
            return "emerging"
        ratio = recent / earlier
        if ratio >= 1.5:
            return "rising"
        if ratio <= 0.5:
            return "declining"
        return "steady"


@dataclass
class MetricProgression:
    dataset: str
    metric: str
    points: list[tuple[int, float, str]] = field(default_factory=list)

    @property
    def improvement(self) -> float | None:
        if len(self.points) < 2:
            return None
        return self.points[-1][1] - self.points[0][1]


@dataclass
class TrendReport:
    timeline: Timeline = field(default_factory=Timeline)
    entity_trends: list[EntityTrend] = field(default_factory=list)
    progressions: list[MetricProgression] = field(default_factory=list)
    narrative: str = ""

    @property
    def rising(self) -> list[EntityTrend]:
        return [t for t in self.entity_trends if t.direction in {"rising", "emerging"}]

    @property
    def declining(self) -> list[EntityTrend]:
        return [t for t in self.entity_trends if t.direction == "declining"]


async def build_timeline(db: AsyncSession, paper_ids: list[uuid.UUID]) -> Timeline:
    if not paper_ids:
        return Timeline()

    rows = (
        await db.execute(
            select(Paper.published_year, Paper.title).where(Paper.id.in_(paper_ids))
        )
    ).all()

    grouped: dict[int, list[str]] = defaultdict(list)
    undated = 0
    for year, title in rows:
        if year is None:
            undated += 1
            continue
        grouped[int(year)].append(title or "Untitled")

    buckets = [
        YearBucket(year=year, paper_count=len(titles), titles=sorted(titles)[:20])
        for year, titles in sorted(grouped.items())
    ]
    return Timeline(buckets=buckets, undated_papers=undated)


async def entity_trends(
    db: AsyncSession,
    paper_ids: list[uuid.UUID],
    *,
    labels: list[str] | None = None,
    top_n: int = 15,
) -> list[EntityTrend]:
    if not paper_ids:
        return []

    stmt = (
        select(
            ExtractedEntity.label,
            ExtractedEntity.name,
            Paper.published_year,
            func.count(func.distinct(ExtractedEntity.paper_id)).label("papers"),
        )
        .join(Paper, Paper.id == ExtractedEntity.paper_id)
        .where(
            ExtractedEntity.paper_id.in_(paper_ids),
            Paper.published_year.is_not(None),
        )
        .group_by(ExtractedEntity.label, ExtractedEntity.name, Paper.published_year)
    )
    if labels:
        stmt = stmt.where(ExtractedEntity.label.in_(labels))

    rows = (await db.execute(stmt)).all()

    series: dict[tuple[str, str], dict[int, int]] = defaultdict(dict)
    totals: Counter[tuple[str, str]] = Counter()
    for label, name, year, papers in rows:
        series[(label, name)][int(year)] = int(papers)
        totals[(label, name)] += int(papers)

    trends = [
        EntityTrend(
            label=label,
            name=name,
            by_year=dict(sorted(series[(label, name)].items())),
            total=total,
        )
        for (label, name), total in totals.most_common(top_n)
    ]
    logger.info("entity_trends_computed", count=len(trends))
    return trends


async def metric_progressions(
    db: AsyncSession, paper_ids: list[uuid.UUID], *, top_n: int = 6
) -> list[MetricProgression]:
    """Best reported score per year, per dataset and metric."""
    if not paper_ids:
        return []

    rows = (
        await db.execute(
            select(
                ExperimentResult.dataset_name,
                ExperimentResult.metric_name,
                ExperimentResult.value,
                ExperimentResult.model_name,
                Paper.published_year,
            )
            .join(Paper, Paper.id == ExperimentResult.paper_id)
            .where(
                ExperimentResult.paper_id.in_(paper_ids),
                ExperimentResult.dataset_key.is_not(None),
                Paper.published_year.is_not(None),
            )
        )
    ).all()

    from app.research.comparison import metric_higher_is_better

    best: dict[tuple[str, str], dict[int, tuple[float, str]]] = defaultdict(dict)
    for dataset, metric, value, model, year in rows:
        key = (dataset, metric)
        year = int(year)
        current = best[key].get(year)
        if current is None:
            best[key][year] = (value, model)
            continue
        higher_better = metric_higher_is_better(metric)
        if (value > current[0]) if higher_better else (value < current[0]):
            best[key][year] = (value, model)

    progressions = [
        MetricProgression(
            dataset=dataset,
            metric=metric,
            points=[(year, value, model) for year, (value, model) in sorted(points.items())],
        )
        for (dataset, metric), points in best.items()
        if len(points) >= MIN_YEARS_FOR_TREND
    ]
    progressions.sort(key=lambda p: len(p.points), reverse=True)
    return progressions[:top_n]


def _format_evidence(report: TrendReport) -> str:
    lines = []
    for bucket in report.timeline.buckets:
        lines.append(f"{bucket.year}: {bucket.paper_count} paper(s)")

    for trend in report.entity_trends[:10]:
        years = ", ".join(f"{y}={c}" for y, c in trend.by_year.items())
        lines.append(f"{trend.name} ({trend.label}, {trend.direction}): {years}")

    for progression in report.progressions[:5]:
        points = ", ".join(f"{y}: {v:g} ({m})" for y, v, m in progression.points)
        lines.append(f"{progression.dataset} / {progression.metric}: {points}")

    return "\n".join(lines)


async def analyze_trends(
    db: AsyncSession,
    paper_ids: list[uuid.UUID],
    *,
    topic: str = "this corpus",
    gateway: LLMGateway | None = None,
    labels: list[str] | None = None,
) -> TrendReport:
    report = TrendReport(
        timeline=await build_timeline(db, paper_ids),
        entity_trends=await entity_trends(db, paper_ids, labels=labels),
        progressions=await metric_progressions(db, paper_ids),
    )

    if gateway is not None and report.timeline.buckets:
        evidence = _format_evidence(report)
        try:
            completion = await gateway.complete(
                [
                    Message(Role.SYSTEM, SYNTHESIZE_TREND.system),
                    Message(Role.USER, SYNTHESIZE_TREND.render(topic=topic, evidence=evidence)),
                ],
                temperature=0.2,
                max_tokens=350,
            )
            report.narrative = completion.text.strip()
        except Exception as exc:  # noqa: BLE001
            logger.warning("trend_narrative_failed", error=str(exc))

    return report
