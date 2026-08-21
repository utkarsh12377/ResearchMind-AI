import uuid

from pydantic import BaseModel, Field

MAX_PAPERS_PER_REQUEST = 50


class PaperScopeRequest(BaseModel):
    """Papers to analyse. Empty means the caller's whole library."""

    paper_ids: list[uuid.UUID] = Field(default_factory=list, max_length=MAX_PAPERS_PER_REQUEST)


class ComparisonRequest(PaperScopeRequest):
    metrics: list[str] = Field(default_factory=list, max_length=20)
    datasets: list[str] = Field(default_factory=list, max_length=20)
    min_papers_per_row: int = Field(default=1, ge=1, le=20)


class ComparisonCellPayload(BaseModel):
    paper_id: uuid.UUID
    paper_title: str
    model: str
    value: float
    unit: str | None = None
    split: str | None = None
    is_best: bool = False


class ComparisonRowPayload(BaseModel):
    dataset: str
    metric: str
    spread: float
    cells: list[ComparisonCellPayload]


class ComparisonResponse(BaseModel):
    rows: list[ComparisonRowPayload]
    ungrouped_results: int
    markdown: str


class PaperMatrixRow(BaseModel):
    paper_id: uuid.UUID
    title: str
    year: int | None
    datasets: list[str]
    models: list[str]
    metrics: list[str]


class PaperMatrixResponse(BaseModel):
    rows: list[PaperMatrixRow]


class TrendRequest(PaperScopeRequest):
    topic: str = Field(default="this corpus", max_length=300)
    labels: list[str] = Field(default_factory=list, max_length=10)


class YearBucketPayload(BaseModel):
    year: int
    paper_count: int
    titles: list[str]


class EntityTrendPayload(BaseModel):
    label: str
    name: str
    by_year: dict[int, int]
    total: int
    direction: str
    first_seen: int | None


class ProgressionPoint(BaseModel):
    year: int
    value: float
    model: str


class ProgressionPayload(BaseModel):
    dataset: str
    metric: str
    points: list[ProgressionPoint]
    improvement: float | None


class TrendResponse(BaseModel):
    narrative: str
    timeline: list[YearBucketPayload]
    undated_papers: int
    entity_trends: list[EntityTrendPayload]
    progressions: list[ProgressionPayload]


class ConsistencyRequest(PaperScopeRequest):
    include_claims: bool = True


class NumericConflictPayload(BaseModel):
    model: str
    dataset: str
    metric: str
    left_paper: str
    left_value: float
    right_paper: str
    right_value: float
    relative_gap: float


class ClaimPairPayload(BaseModel):
    relation: str
    confidence: float
    reason: str
    left_paper: str
    left_claim: str
    right_paper: str
    right_claim: str


class ConsistencyResponse(BaseModel):
    summary: str
    is_consistent: bool
    papers_examined: int
    numeric_conflicts: list[NumericConflictPayload]
    contradictions: list[ClaimPairPayload]
    agreements: list[ClaimPairPayload]


class GapRequest(PaperScopeRequest):
    topic: str = Field(default="this research area", max_length=300)


class StructuralGapPayload(BaseModel):
    kind: str
    description: str
    evidence: list[str]
    strength: float


class StatedGapPayload(BaseModel):
    paper_title: str
    text: str


class DirectionPayload(BaseModel):
    gap: str
    rationale: str
    suggested_direction: str
    confidence: float


class GapResponse(BaseModel):
    topic: str
    summary: str
    papers_examined: int
    structural: list[StructuralGapPayload]
    stated: list[StatedGapPayload]
    directions: list[DirectionPayload]


class ReviewRequest(PaperScopeRequest):
    topic: str = Field(min_length=1, max_length=300)
    max_sections: int = Field(default=6, ge=1, le=8)


class ReviewSectionPayload(BaseModel):
    heading: str
    focus: str
    text: str
    source_indices: list[int]


class ReviewSourcePayload(BaseModel):
    index: int
    paper_id: uuid.UUID
    title: str
    authors: str | None
    year: int | None
    citation: str


class ReviewResponse(BaseModel):
    topic: str
    title: str
    sections: list[ReviewSectionPayload]
    sources: list[ReviewSourcePayload]
    word_count: int
    markdown: str
    bibtex: str
