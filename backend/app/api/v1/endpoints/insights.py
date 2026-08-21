"""Cross-paper analysis endpoints: comparison, trends, consistency, gaps, reviews.

Every route takes an optional list of paper ids and falls back to the caller's
whole library. Requested ids are always intersected with what the caller can
actually read, so scoping is a convenience rather than an authorization
boundary.
"""

import uuid

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.v1.deps import get_current_active_user
from app.api.v1.endpoints.graph import accessible_paper_ids
from app.db.session import get_db
from app.llm.gateway import get_gateway
from app.models import User
from app.research.comparison import build_comparison_table, build_paper_matrix
from app.research.contradictions import analyze_consistency
from app.research.gaps import discover_gaps
from app.research.review import generate_literature_review
from app.research.trends import analyze_trends
from app.schemas.insights import (
    ClaimPairPayload,
    ComparisonCellPayload,
    ComparisonRequest,
    ComparisonResponse,
    ComparisonRowPayload,
    ConsistencyRequest,
    ConsistencyResponse,
    DirectionPayload,
    EntityTrendPayload,
    GapRequest,
    GapResponse,
    NumericConflictPayload,
    PaperMatrixResponse,
    PaperMatrixRow,
    PaperScopeRequest,
    ProgressionPayload,
    ProgressionPoint,
    ReviewRequest,
    ReviewResponse,
    ReviewSectionPayload,
    ReviewSourcePayload,
    StatedGapPayload,
    StructuralGapPayload,
    TrendRequest,
    TrendResponse,
    YearBucketPayload,
)

router = APIRouter(prefix="/insights", tags=["insights"])


async def _scope(
    db: AsyncSession, user: User, requested: list[uuid.UUID]
) -> list[uuid.UUID]:
    accessible = await accessible_paper_ids(db, user)
    if not requested:
        return accessible
    allowed = set(accessible)
    return [paper_id for paper_id in requested if paper_id in allowed]


@router.post("/comparison", response_model=ComparisonResponse)
async def comparison(
    request: ComparisonRequest,
    user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> ComparisonResponse:
    """Compare reported results across papers, grouped by dataset and metric."""
    paper_ids = await _scope(db, user, request.paper_ids)
    table = await build_comparison_table(
        db,
        paper_ids,
        metrics=request.metrics or None,
        datasets=request.datasets or None,
        min_papers_per_row=request.min_papers_per_row,
    )

    return ComparisonResponse(
        rows=[
            ComparisonRowPayload(
                dataset=row.dataset,
                metric=row.metric,
                spread=row.spread,
                cells=[
                    ComparisonCellPayload(
                        paper_id=cell.paper_id,
                        paper_title=cell.paper_title,
                        model=cell.model,
                        value=cell.value,
                        unit=cell.unit,
                        split=cell.split,
                        is_best=cell.is_best,
                    )
                    for cell in row.cells
                ],
            )
            for row in table.rows
        ],
        ungrouped_results=table.ungrouped,
        markdown=table.to_markdown(),
    )


@router.post("/matrix", response_model=PaperMatrixResponse)
async def matrix(
    request: PaperScopeRequest,
    user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> PaperMatrixResponse:
    """One row per paper: what it used and what it measured."""
    paper_ids = await _scope(db, user, request.paper_ids)
    rows = await build_paper_matrix(db, paper_ids)

    return PaperMatrixResponse(
        rows=[
            PaperMatrixRow(
                paper_id=row.paper_id,
                title=row.title,
                year=row.year,
                datasets=row.datasets,
                models=row.models,
                metrics=row.metrics,
            )
            for row in rows
        ]
    )


@router.post("/trends", response_model=TrendResponse)
async def trends(
    request: TrendRequest,
    user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> TrendResponse:
    """How the corpus changed over time, measured before it is narrated."""
    paper_ids = await _scope(db, user, request.paper_ids)
    report = await analyze_trends(
        db,
        paper_ids,
        topic=request.topic,
        gateway=get_gateway(),
        labels=request.labels or None,
    )

    return TrendResponse(
        narrative=report.narrative,
        timeline=[
            YearBucketPayload(year=b.year, paper_count=b.paper_count, titles=b.titles)
            for b in report.timeline.buckets
        ],
        undated_papers=report.timeline.undated_papers,
        entity_trends=[
            EntityTrendPayload(
                label=trend.label,
                name=trend.name,
                by_year=trend.by_year,
                total=trend.total,
                direction=trend.direction,
                first_seen=trend.first_seen,
            )
            for trend in report.entity_trends
        ],
        progressions=[
            ProgressionPayload(
                dataset=p.dataset,
                metric=p.metric,
                points=[
                    ProgressionPoint(year=year, value=value, model=model)
                    for year, value, model in p.points
                ],
                improvement=p.improvement,
            )
            for p in report.progressions
        ],
    )


@router.post("/consistency", response_model=ConsistencyResponse)
async def consistency(
    request: ConsistencyRequest,
    user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> ConsistencyResponse:
    """Where the papers agree and where they conflict."""
    paper_ids = await _scope(db, user, request.paper_ids)
    report = await analyze_consistency(
        db,
        paper_ids,
        gateway=get_gateway() if request.include_claims else None,
        include_claims=request.include_claims,
    )

    return ConsistencyResponse(
        summary=report.summary(),
        is_consistent=report.is_consistent,
        papers_examined=report.papers_examined,
        numeric_conflicts=[
            NumericConflictPayload(
                model=c.model,
                dataset=c.dataset,
                metric=c.metric,
                left_paper=c.left_paper,
                left_value=c.left_value,
                right_paper=c.right_paper,
                right_value=c.right_value,
                relative_gap=c.relative_gap,
            )
            for c in report.numeric_conflicts
        ],
        contradictions=[_claim_pair(c) for c in report.contradictions],
        agreements=[_claim_pair(c) for c in report.agreements],
    )


def _claim_pair(pair) -> ClaimPairPayload:  # noqa: ANN001
    return ClaimPairPayload(
        relation=pair.relation,
        confidence=pair.confidence,
        reason=pair.reason,
        left_paper=pair.left_paper,
        left_claim=pair.left_claim,
        right_paper=pair.right_paper,
        right_claim=pair.right_claim,
    )


@router.post("/gaps", response_model=GapResponse)
async def gaps(
    request: GapRequest,
    user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> GapResponse:
    """Open problems the corpus points at, grounded in evidence rather than guessed."""
    paper_ids = await _scope(db, user, request.paper_ids)
    report = await discover_gaps(db, paper_ids, topic=request.topic, gateway=get_gateway())

    return GapResponse(
        topic=report.topic,
        summary=report.summary(),
        papers_examined=report.papers_examined,
        structural=[
            StructuralGapPayload(
                kind=gap.kind,
                description=gap.description,
                evidence=gap.evidence,
                strength=gap.strength,
            )
            for gap in report.structural
        ],
        stated=[
            StatedGapPayload(paper_title=gap.paper_title, text=gap.text) for gap in report.stated
        ],
        directions=[
            DirectionPayload(
                gap=d.gap,
                rationale=d.rationale,
                suggested_direction=d.suggested_direction,
                confidence=d.confidence,
            )
            for d in report.directions
        ],
    )


@router.post("/review", response_model=ReviewResponse)
async def review(
    request: ReviewRequest,
    user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> ReviewResponse:
    """Generate a cited literature review over the selected papers.

    Markdown and BibTeX are returned alongside the structured sections so the
    output is usable as a document without the client having to reassemble it.
    """
    paper_ids = await _scope(db, user, request.paper_ids)
    generated = await generate_literature_review(
        db,
        user,
        request.topic,
        paper_ids,
        gateway=get_gateway(),
        max_sections=request.max_sections,
    )

    return ReviewResponse(
        topic=generated.topic,
        title=generated.title,
        sections=[
            ReviewSectionPayload(
                heading=section.heading,
                focus=section.focus,
                text=section.text,
                source_indices=section.source_indices,
            )
            for section in generated.sections
        ],
        sources=[
            ReviewSourcePayload(
                index=source.index,
                paper_id=source.paper_id,
                title=source.title,
                authors=source.authors,
                year=source.year,
                citation=source.citation,
            )
            for source in generated.sources
        ],
        word_count=generated.word_count,
        markdown=generated.to_markdown(),
        bibtex=generated.to_bibtex(),
    )
