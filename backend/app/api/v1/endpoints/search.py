from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.v1.deps import get_current_active_user
from app.db.session import get_db
from app.models import User
from app.retrieval.service import RetrievalFilters, retrieve
from app.schemas.retrieval import SearchRequest, SearchResponse, SearchResult

router = APIRouter(prefix="/search", tags=["search"])


@router.post("", response_model=SearchResponse)
async def search(
    request: SearchRequest,
    user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> SearchResponse:
    """Hybrid search over the caller's indexed papers.

    Runs dense and BM25 retrieval concurrently, fuses by reciprocal rank, then
    reranks the shortlist. Results carry their per-retriever ranks so callers
    can see why a passage surfaced.
    """
    results = await retrieve(
        db,
        user,
        request.query,
        limit=request.limit,
        filters=RetrievalFilters(
            paper_ids=request.filters.paper_ids,
            kinds=request.filters.kinds,
            year_from=request.filters.year_from,
            year_to=request.filters.year_to,
            sections=request.filters.sections,
        ),
        use_reranker=request.use_reranker,
    )

    return SearchResponse(
        query=request.query,
        total=len(results),
        results=[
            SearchResult(
                chunk_id=result.chunk_id,
                paper_id=result.paper_id,
                paper_title=result.paper_title,
                content=result.content,
                section_path=result.section_path,
                page_number=result.page_number,
                kind=result.kind,
                citation=result.citation,
                score=result.score,
                dense_rank=result.dense_rank,
                sparse_rank=result.sparse_rank,
                rerank_score=result.rerank_score,
                supporting_sentences=result.supporting_sentences,
            )
            for result in results
        ],
    )
