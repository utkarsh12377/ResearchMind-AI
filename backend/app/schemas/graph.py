import uuid

from pydantic import BaseModel, Field


class GraphNodePayload(BaseModel):
    id: str
    label: str
    kind: str
    weight: int = 1
    metadata: dict = Field(default_factory=dict)


class GraphEdgePayload(BaseModel):
    source: str
    target: str
    weight: int = 1
    kind: str = "related"


class NetworkResponse(BaseModel):
    view: str
    nodes: list[GraphNodePayload]
    edges: list[GraphEdgePayload]
    node_count: int
    edge_count: int


class EntityCount(BaseModel):
    label: str
    name: str
    paper_count: int


class GraphOverviewResponse(BaseModel):
    papers: int
    counts: dict[str, int]
    top_entities: list[EntityCount]
    most_connected: list["ConnectedPaper"]


class ConnectedPaper(BaseModel):
    paper_id: uuid.UUID
    title: str
    entity_count: int


class GraphQueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)


class GraphQueryResponse(BaseModel):
    question: str
    cypher: str
    rows: list[dict]
    error: str | None = None


class NeighborPayload(BaseModel):
    label: str
    key: str
    name: str
    relationship: str
    distance: int


class NeighborsResponse(BaseModel):
    label: str
    key: str
    neighbors: list[NeighborPayload]


GraphOverviewResponse.model_rebuild()
