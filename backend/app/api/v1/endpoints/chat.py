import json

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.v1.deps import get_current_active_user
from app.db.session import get_db
from app.llm.rag import answer_question, stream_answer
from app.models import User
from app.retrieval.service import RetrievalFilters
from app.schemas.retrieval import SearchFilters

router = APIRouter(prefix="/chat", tags=["chat"])


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    limit: int = Field(default=8, ge=1, le=30)
    filters: SearchFilters = Field(default_factory=SearchFilters)
    use_corrective_retrieval: bool = True
    use_verification: bool = True


class SourceRef(BaseModel):
    index: int
    chunk_id: str
    paper_id: str
    paper_title: str | None
    citation: str
    page_number: int | None
    content: str


class GradeInfo(BaseModel):
    sufficient: bool
    reason: str
    missing: str


class VerificationInfo(BaseModel):
    supported: bool
    confidence: float
    unsupported_claims: list[str]
    reason: str


class ChatResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceRef]
    cited_indices: list[int]
    confidence: float
    grade: GradeInfo | None
    verification: VerificationInfo | None
    rewritten_query: str | None
    prompt_tokens: int
    completion_tokens: int


def _to_filters(filters: SearchFilters) -> RetrievalFilters:
    return RetrievalFilters(
        paper_ids=filters.paper_ids,
        kinds=filters.kinds,
        year_from=filters.year_from,
        year_to=filters.year_to,
        sections=filters.sections,
    )


@router.post("", response_model=ChatResponse)
async def ask(
    request: ChatRequest,
    user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> ChatResponse:
    """Answer a question from the caller's papers, with citations.

    Runs the corrective-RAG loop: retrieval is graded, a weak result triggers a
    rewritten query, and the final answer is verified against its sources. The
    grade and verification are returned rather than hidden so the caller can
    see how much to trust the answer.
    """
    result = await answer_question(
        db,
        user,
        request.question,
        limit=request.limit,
        filters=_to_filters(request.filters),
        use_corrective_retrieval=request.use_corrective_retrieval,
        use_verification=request.use_verification,
    )

    return ChatResponse(
        question=result.question,
        answer=result.answer,
        cited_indices=result.cited_indices,
        confidence=result.confidence,
        rewritten_query=result.rewritten_query,
        prompt_tokens=result.usage.prompt_tokens,
        completion_tokens=result.usage.completion_tokens,
        grade=(
            GradeInfo(
                sufficient=result.grade.sufficient,
                reason=result.grade.reason,
                missing=result.grade.missing,
            )
            if result.grade
            else None
        ),
        verification=(
            VerificationInfo(
                supported=result.verification.supported,
                confidence=result.verification.confidence,
                unsupported_claims=result.verification.unsupported_claims,
                reason=result.verification.reason,
            )
            if result.verification
            else None
        ),
        sources=[
            SourceRef(
                index=index,
                chunk_id=str(chunk.chunk_id),
                paper_id=str(chunk.paper_id),
                paper_title=chunk.paper_title,
                citation=chunk.citation,
                page_number=chunk.page_number,
                content=chunk.content,
            )
            for index, chunk in enumerate(result.sources, start=1)
        ],
    )


@router.post("/stream")
async def ask_streaming(
    request: ChatRequest,
    user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> StreamingResponse:
    """Stream an answer as server-sent events.

    Sources arrive as the first event so the UI can render citations before the
    first token. Verification is omitted here because it can only run against a
    finished answer.
    """

    async def event_stream():  # noqa: ANN202
        async for event in stream_answer(
            db,
            user,
            request.question,
            limit=request.limit,
            filters=_to_filters(request.filters),
        ):
            yield f"data: {json.dumps(event)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            # Proxies that buffer would defeat streaming entirely.
            "X-Accel-Buffering": "no",
        },
    )
