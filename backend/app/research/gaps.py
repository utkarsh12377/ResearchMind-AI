"""Research gap discovery.

Two sources of evidence feed the same analysis. Structural gaps come from the
extracted rows — a dataset every paper but one evaluates on, a model never tested
on a benchmark its peers all use — and are found by set arithmetic, not by asking
a model to imagine what is missing. Stated gaps come from the papers' own future
work sections. The LLM's job is to turn that evidence into readable directions,
not to supply the evidence.
"""

from __future__ import annotations

import uuid
from collections import defaultdict
from dataclasses import dataclass, field

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.llm.gateway import LLMGateway, Message, Role
from app.llm.json_output import clamp_confidence, load_json_object
from app.llm.prompts import IDENTIFY_GAPS
from app.models import DocumentChunk, ExperimentResult, ExtractedEntity, Paper
from app.research.contradictions import find_numeric_conflicts

logger = get_logger(__name__)

MIN_CORPUS_COVERAGE = 0.4
MAX_STRUCTURAL_GAPS = 12
FUTURE_WORK_MARKERS = (
    "future work", "future research", "we leave", "left for future",
    "remains an open", "open question", "an interesting direction",
    "further work", "we plan to", "has yet to be", "remains unexplored",
)


@dataclass
class StructuralGap:
    kind: str
    description: str
    evidence: list[str] = field(default_factory=list)
    strength: float = 0.5


@dataclass
class StatedGap:
    paper_title: str
    text: str


@dataclass
class SuggestedDirection:
    gap: str
    rationale: str
    suggested_direction: str
    confidence: float
    evidence_indices: list[int] = field(default_factory=list)


@dataclass
class GapReport:
    topic: str
    structural: list[StructuralGap] = field(default_factory=list)
    stated: list[StatedGap] = field(default_factory=list)
    directions: list[SuggestedDirection] = field(default_factory=list)
    papers_examined: int = 0

    @property
    def has_findings(self) -> bool:
        return bool(self.structural or self.stated or self.directions)

    def summary(self) -> str:
        if not self.has_findings:
            return f"No gaps identified across {self.papers_examined} papers."
        return (
            f"{len(self.structural)} structural gap(s), {len(self.stated)} stated "
            f"by the authors, {len(self.directions)} suggested direction(s)."
        )


async def find_coverage_gaps(
    db: AsyncSession, paper_ids: list[uuid.UUID]
) -> list[StructuralGap]:
    """Entities that are near-universal in the corpus but absent from some papers."""
    if len(paper_ids) < 3:
        return []

    rows = (
        await db.execute(
            select(ExtractedEntity.label, ExtractedEntity.name, ExtractedEntity.paper_id)
            .where(ExtractedEntity.paper_id.in_(paper_ids))
        )
    ).all()

    titles = dict(
        (await db.execute(select(Paper.id, Paper.title).where(Paper.id.in_(paper_ids)))).all()
    )

    coverage: dict[tuple[str, str], set[uuid.UUID]] = defaultdict(set)
    for label, name, paper_id in rows:
        coverage[(label, name)].add(paper_id)

    total = len(paper_ids)
    gaps: list[StructuralGap] = []

    for (label, name), covered in coverage.items():
        ratio = len(covered) / total
        if ratio < MIN_CORPUS_COVERAGE or ratio >= 1.0:
            continue

        missing = [pid for pid in paper_ids if pid not in covered]
        if not missing:
            continue

        gaps.append(
            StructuralGap(
                kind=f"unused_{label.lower()}",
                description=(
                    f"{name} appears in {len(covered)} of {total} papers but is absent "
                    f"from {len(missing)}"
                ),
                evidence=[titles.get(pid) or "Untitled" for pid in missing[:5]],
                strength=ratio,
            )
        )

    gaps.sort(key=lambda g: g.strength, reverse=True)
    return gaps[:MAX_STRUCTURAL_GAPS]


async def find_evaluation_gaps(
    db: AsyncSession, paper_ids: list[uuid.UUID]
) -> list[StructuralGap]:
    """Model and dataset pairs the corpus never actually evaluates together."""
    if len(paper_ids) < 2:
        return []

    results = (
        await db.scalars(
            select(ExperimentResult).where(
                ExperimentResult.paper_id.in_(paper_ids),
                ExperimentResult.dataset_key.is_not(None),
            )
        )
    ).all()
    if not results:
        return []

    tested: set[tuple[str, str]] = set()
    models: dict[str, str] = {}
    datasets: dict[str, str] = {}

    for result in results:
        tested.add((result.model_key, result.dataset_key))
        models[result.model_key] = result.model_name
        datasets[result.dataset_key] = result.dataset_name or result.dataset_key

    if len(models) < 2 or len(datasets) < 2:
        return []

    gaps = []
    for model_key, model_name in models.items():
        untested = [
            datasets[dataset_key]
            for dataset_key in datasets
            if (model_key, dataset_key) not in tested
        ]
        if not untested or len(untested) == len(datasets):
            continue
        gaps.append(
            StructuralGap(
                kind="untested_pairing",
                description=(
                    f"{model_name} is never evaluated on "
                    f"{', '.join(untested[:3])}"
                ),
                evidence=untested[:5],
                strength=1 - (len(untested) / len(datasets)),
            )
        )

    gaps.sort(key=lambda g: g.strength, reverse=True)
    return gaps[:MAX_STRUCTURAL_GAPS]


async def find_stated_gaps(
    db: AsyncSession, paper_ids: list[uuid.UUID], *, per_paper: int = 2
) -> list[StatedGap]:
    """Passages where the authors name their own open problems."""
    if not paper_ids:
        return []

    stated: list[StatedGap] = []
    for paper_id in paper_ids:
        paper = await db.get(Paper, paper_id)
        if paper is None:
            continue

        chunks = (
            await db.scalars(
                select(DocumentChunk)
                .where(DocumentChunk.paper_id == paper_id)
                .order_by(DocumentChunk.chunk_index.desc())
                .limit(12)
            )
        ).all()

        found = 0
        for chunk in chunks:
            lowered = chunk.content.lower()
            marker = next((m for m in FUTURE_WORK_MARKERS if m in lowered), None)
            if marker is None:
                continue

            position = lowered.index(marker)
            excerpt = chunk.content[max(0, position - 80) : position + 320]
            stated.append(
                StatedGap(
                    paper_title=paper.title or paper.original_filename,
                    text=" ".join(excerpt.split()),
                )
            )
            found += 1
            if found >= per_paper:
                break

    return stated


def _format_evidence(report: GapReport) -> tuple[str, list[str]]:
    lines: list[str] = []
    labels: list[str] = []

    for gap in report.structural:
        labels.append(gap.description)
        lines.append(f"[{len(labels)}] Structural: {gap.description}")

    for gap in report.stated:
        labels.append(f"{gap.paper_title}: {gap.text[:120]}")
        lines.append(f"[{len(labels)}] Stated in '{gap.paper_title}': {gap.text}")

    return "\n".join(lines), labels


def parse_gaps_payload(raw: str) -> list[SuggestedDirection]:
    payload = load_json_object(raw)
    if payload is None:
        return []

    directions = []
    for item in payload.get("gaps", []) or []:
        if not isinstance(item, dict) or not item.get("gap"):
            continue
        indices = [
            int(i)
            for i in (item.get("evidence_indices") or [])
            if isinstance(i, int | float) and int(i) > 0
        ]
        directions.append(
            SuggestedDirection(
                gap=str(item["gap"])[:500],
                rationale=str(item.get("rationale", ""))[:800],
                suggested_direction=str(item.get("suggested_direction", ""))[:800],
                confidence=clamp_confidence(item.get("confidence")),
                evidence_indices=indices[:8],
            )
        )
    directions.sort(key=lambda d: d.confidence, reverse=True)
    return directions


async def discover_gaps(
    db: AsyncSession,
    paper_ids: list[uuid.UUID],
    *,
    topic: str = "this research area",
    gateway: LLMGateway | None = None,
) -> GapReport:
    report = GapReport(topic=topic, papers_examined=len(paper_ids))
    report.structural = await find_coverage_gaps(db, paper_ids)
    report.structural.extend(await find_evaluation_gaps(db, paper_ids))
    report.stated = await find_stated_gaps(db, paper_ids)

    conflicts = await find_numeric_conflicts(db, paper_ids)
    for conflict in conflicts[:5]:
        report.structural.append(
            StructuralGap(
                kind="unresolved_conflict",
                description=f"Unreconciled result: {conflict.describe()}",
                evidence=[conflict.left_paper, conflict.right_paper],
                strength=min(1.0, conflict.relative_gap),
            )
        )

    if gateway is not None and (report.structural or report.stated):
        evidence, _ = _format_evidence(report)
        try:
            completion = await gateway.complete(
                [
                    Message(Role.SYSTEM, IDENTIFY_GAPS.system),
                    Message(Role.USER, IDENTIFY_GAPS.render(topic=topic, evidence=evidence)),
                ],
                temperature=0.4,
                max_tokens=1200,
            )
            report.directions = parse_gaps_payload(completion.text)
        except Exception as exc:  # noqa: BLE001
            logger.warning("gap_synthesis_failed", error=str(exc))

    logger.info(
        "gaps_discovered",
        papers=len(paper_ids),
        structural=len(report.structural),
        stated=len(report.stated),
        directions=len(report.directions),
    )
    return report
