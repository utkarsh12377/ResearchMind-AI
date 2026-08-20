import json

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.graph import run_research, stream_research
from app.api.v1.deps import get_current_active_user
from app.db.session import get_db
from app.models import User
from app.schemas.retrieval import SearchFilters

router = APIRouter(prefix="/research", tags=["research"])


class ResearchRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    limit: int = Field(default=8, ge=1, le=30)
    filters: SearchFilters = Field(default_factory=SearchFilters)


class AgentStep(BaseModel):
    agent: str
    summary: str
    detail: str
    duration_ms: int


class ResearchSource(BaseModel):
    index: int
    chunk_id: str
    paper_id: str
    paper_title: str | None
    citation: str
    page_number: int | None
    content: str


class PlanInfo(BaseModel):
    intent: str
    sub_questions: list[str]
    reasoning: str
    needs_web_search: bool


class ResearchResponse(BaseModel):
    question: str
    answer: str
    plan: PlanInfo | None
    sources: list[ResearchSource]
    citations: list[int]
    confidence: float
    verification: dict | None
    revision_count: int
    usage_tokens: int
    steps: list[AgentStep]
    # Surfaced rather than swallowed: an agent can fail without failing the
    # run, and the caller should know the answer is partial.
    errors: list[str]


def _filters_dict(filters: SearchFilters) -> dict:
    return {
        "paper_ids": filters.paper_ids,
        "kinds": filters.kinds,
        "year_from": filters.year_from,
        "year_to": filters.year_to,
        "sections": filters.sections,
    }


@router.post("", response_model=ResearchResponse)
async def research(
    request: ResearchRequest,
    user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Run the full multi-agent research graph.

    Heavier than /chat: it plans and decomposes the question, retrieves per
    sub-question, drafts, critiques and revises, then verifies. Use /chat for
    single-shot questions and this for comparisons and literature reviews.
    """
    return await run_research(
        db,
        user,
        request.question,
        limit=request.limit,
        filters=_filters_dict(request.filters),
    )


@router.post("/stream")
async def research_streaming(
    request: ResearchRequest,
    user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> StreamingResponse:
    """Stream each agent's progress, then the final result.

    A multi-agent run is slow enough that a silent wait reads as a hang;
    streaming each completed step turns the latency into visible progress.
    """

    async def event_stream():  # noqa: ANN202
        async for event in stream_research(
            db,
            user,
            request.question,
            limit=request.limit,
            filters=_filters_dict(request.filters),
        ):
            yield f"data: {json.dumps(event, default=str)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
