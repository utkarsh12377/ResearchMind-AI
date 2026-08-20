import uuid

from pydantic import BaseModel, Field


class SearchFilters(BaseModel):
    paper_ids: list[uuid.UUID] = Field(default_factory=list)
    kinds: list[str] = Field(default_factory=list, description="text, table, or figure")
    year_from: int | None = Field(default=None, ge=1800, le=2200)
    year_to: int | None = Field(default=None, ge=1800, le=2200)
    sections: list[str] = Field(
        default_factory=list, description="Match chunks whose section path contains these"
    )


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    limit: int = Field(default=10, ge=1, le=50)
    filters: SearchFilters = Field(default_factory=SearchFilters)
    use_reranker: bool = True


class SearchResult(BaseModel):
    chunk_id: uuid.UUID
    paper_id: uuid.UUID
    paper_title: str | None
    content: str
    section_path: str | None
    page_number: int | None
    kind: str
    citation: str
    score: float
    # Exposed for explainability: which retriever surfaced this, and where the
    # reranker moved it. The UI shows this so results are auditable.
    dense_rank: int | None
    sparse_rank: int | None
    rerank_score: float | None
    supporting_sentences: list[str]


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]
    total: int
