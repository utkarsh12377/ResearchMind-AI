"""Operator endpoints: system state, corpus statistics, index maintenance.

Superuser only. These read across every workspace, which is exactly why they
cannot sit behind the ordinary active-user dependency.
"""

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.v1.deps import get_current_superuser
from app.core.config import get_settings
from app.core.metrics import PROMETHEUS_AVAILABLE, set_indexed_chunks
from app.db.session import get_db
from app.graph.store import get_graph_store
from app.llm.gateway import get_gateway
from app.models import (
    DocumentChunk,
    ExperimentResult,
    ExtractedEntity,
    Paper,
    PaperStatus,
    User,
    Workspace,
)
from app.retrieval.service import rebuild_sparse_index
from app.retrieval.sparse import get_bm25_index
from app.retrieval.vector_store import get_vector_store

router = APIRouter(prefix="/admin", tags=["admin"])


class ComponentStatus(BaseModel):
    embedding_provider: str
    vector_store: str
    reranker: str
    graph_store: str
    llm_provider: str
    llm_fallback: str | None
    metrics_enabled: bool
    web_search_enabled: bool


class CorpusStats(BaseModel):
    users: int
    workspaces: int
    papers: int
    papers_by_status: dict[str, int]
    chunks: int
    embedded_chunks: int
    entities: int
    experiment_results: int


class IndexStats(BaseModel):
    sparse_documents: int
    dense_vectors: int
    graph_counts: dict[str, int]


class SystemStatusResponse(BaseModel):
    components: ComponentStatus
    corpus: CorpusStats
    indexes: IndexStats


class ReindexResponse(BaseModel):
    sparse_documents: int


@router.get("/status", response_model=SystemStatusResponse)
async def status(
    _: User = Depends(get_current_superuser),
    db: AsyncSession = Depends(get_db),
) -> SystemStatusResponse:
    """Which backends are live and what the corpus currently holds."""
    settings = get_settings()
    gateway = get_gateway()
    store = get_vector_store()
    bm25 = get_bm25_index()

    status_rows = (
        await db.execute(select(Paper.status, func.count()).group_by(Paper.status))
    ).all()

    corpus = CorpusStats(
        users=await _count(db, User),
        workspaces=await _count(db, Workspace),
        papers=await _count(db, Paper),
        papers_by_status={
            _status_name(value): int(count) for value, count in status_rows
        },
        chunks=await _count(db, DocumentChunk),
        embedded_chunks=int(
            await db.scalar(
                select(func.count())
                .select_from(DocumentChunk)
                .where(DocumentChunk.is_embedded.is_(True))
            )
            or 0
        ),
        entities=await _count(db, ExtractedEntity),
        experiment_results=await _count(db, ExperimentResult),
    )

    set_indexed_chunks(bm25.size)

    return SystemStatusResponse(
        components=ComponentStatus(
            embedding_provider=settings.embedding_provider,
            vector_store=settings.vector_store_backend,
            reranker=settings.reranker_backend,
            graph_store=settings.graph_store_backend,
            llm_provider=gateway.primary.name,
            llm_fallback=gateway.fallback.name if gateway.fallback else None,
            metrics_enabled=settings.metrics_enabled and PROMETHEUS_AVAILABLE,
            web_search_enabled=settings.web_search_enabled,
        ),
        corpus=corpus,
        indexes=IndexStats(
            sparse_documents=bm25.size,
            dense_vectors=await store.count(),
            graph_counts=await get_graph_store().counts(),
        ),
    )


def _status_name(value: object) -> str:
    return value.value if isinstance(value, PaperStatus) else str(value)


async def _count(db: AsyncSession, model) -> int:  # noqa: ANN001
    return int(await db.scalar(select(func.count()).select_from(model)) or 0)


@router.post("/reindex-sparse", response_model=ReindexResponse)
async def reindex_sparse(
    _: User = Depends(get_current_superuser),
    db: AsyncSession = Depends(get_db),
) -> ReindexResponse:
    """Rebuild BM25 from Postgres.

    Term statistics live in process memory and are lost on restart. Startup
    warms them, but this makes the recovery explicit for when the index drifts
    without a restart to fix it.
    """
    count = await rebuild_sparse_index(db)
    set_indexed_chunks(count)
    return ReindexResponse(sparse_documents=count)
