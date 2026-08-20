"""Knowledge graph endpoints, shaped for the frontend explorer.

The network responses return flat node and edge lists rather than nested paths,
because that is what a force-directed layout consumes. Building the adjacency
client-side from nested data would mean shipping the same node several times.
"""

import uuid

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.v1.deps import get_current_active_user
from app.core.exceptions import ValidationError
from app.db.session import get_db
from app.graph.analytics import (
    NetworkView,
    citation_network,
    coauthorship_network,
    entity_network,
    most_connected_papers,
)
from app.graph.query import answer_graph_question
from app.graph.retrieval import entity_cooccurrence
from app.graph.schema import NodeLabel
from app.graph.store import get_graph_store
from app.models import Paper, User, Workspace
from app.schemas.graph import (
    ConnectedPaper,
    EntityCount,
    GraphEdgePayload,
    GraphNodePayload,
    GraphOverviewResponse,
    GraphQueryRequest,
    GraphQueryResponse,
    NeighborPayload,
    NeighborsResponse,
    NetworkResponse,
)

router = APIRouter(prefix="/graph", tags=["graph"])

VIEWS = ("entities", "citations", "authors")


async def accessible_paper_ids(db: AsyncSession, user: User) -> list[uuid.UUID]:
    workspace_ids = (
        await db.scalars(select(Workspace.id).where(Workspace.owner_id == user.id))
    ).all()
    if not workspace_ids:
        return []
    return list(
        (await db.scalars(select(Paper.id).where(Paper.workspace_id.in_(workspace_ids)))).all()
    )


def _to_response(view: str, network: NetworkView) -> NetworkResponse:
    return NetworkResponse(
        view=view,
        nodes=[
            GraphNodePayload(
                id=node.id,
                label=node.label,
                kind=node.kind,
                weight=node.weight,
                metadata=node.metadata,
            )
            for node in network.nodes
        ],
        edges=[
            GraphEdgePayload(
                source=edge.source, target=edge.target, weight=edge.weight, kind=edge.kind
            )
            for edge in network.edges
        ],
        node_count=len(network.nodes),
        edge_count=len(network.edges),
    )


@router.get("/overview", response_model=GraphOverviewResponse)
async def overview(
    user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> GraphOverviewResponse:
    """Counts and headline entities for the caller's corpus."""
    paper_ids = await accessible_paper_ids(db, user)
    store = get_graph_store()

    entities = await entity_cooccurrence(db, paper_ids, limit=20)
    connected = await most_connected_papers(db, paper_ids)

    return GraphOverviewResponse(
        papers=len(paper_ids),
        counts=await store.counts(),
        top_entities=[
            EntityCount(label=label, name=name, paper_count=count)
            for label, name, count in entities
        ],
        most_connected=[
            ConnectedPaper(paper_id=pid, title=title, entity_count=degree)
            for pid, title, degree in connected
        ],
    )


@router.get("/network", response_model=NetworkResponse)
async def network(
    view: str = Query(default="entities"),
    labels: list[str] | None = Query(default=None),
    min_papers: int = Query(default=2, ge=1, le=50),
    user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> NetworkResponse:
    """A node/edge view of the corpus, ready for a force-directed layout."""
    if view not in VIEWS:
        raise ValidationError(f"Unknown view {view!r}. Choose one of: {', '.join(VIEWS)}")

    known = {label.value for label in NodeLabel}
    if labels and not set(labels) <= known:
        raise ValidationError(f"Unknown label(s): {', '.join(sorted(set(labels) - known))}")

    paper_ids = await accessible_paper_ids(db, user)

    if view == "citations":
        return _to_response(view, await citation_network(db, paper_ids))
    if view == "authors":
        return _to_response(view, await coauthorship_network(db, paper_ids))
    return _to_response(
        view, await entity_network(db, paper_ids, labels=labels, min_papers=min_papers)
    )


@router.get("/neighbors", response_model=NeighborsResponse)
async def neighbors(
    label: str = Query(...),
    key: str = Query(..., min_length=1, max_length=255),
    depth: int = Query(default=1, ge=1, le=3),
    limit: int = Query(default=50, ge=1, le=200),
    user: User = Depends(get_current_active_user),
) -> NeighborsResponse:
    """Walk out from one node."""
    if label not in {item.value for item in NodeLabel}:
        raise ValidationError(f"Unknown node label: {label!r}")

    hits = await get_graph_store().neighbors(label, key, depth=depth, limit=limit)

    return NeighborsResponse(
        label=label,
        key=key,
        neighbors=[
            NeighborPayload(
                label=hit.node.label,
                key=hit.node.key,
                name=str(hit.node.properties.get("name", hit.node.key)),
                relationship=hit.relationship,
                distance=hit.distance,
            )
            for hit in hits
        ],
    )


@router.post("/query", response_model=GraphQueryResponse)
async def query(
    request: GraphQueryRequest,
    user: User = Depends(get_current_active_user),
) -> GraphQueryResponse:
    """Answer a question by generating and running read-only Cypher.

    The generated query is returned alongside the rows whether it succeeded or
    not. A graph answer nobody can inspect is not one worth trusting, and the
    rejected query is the most useful thing to show when validation fails.
    """
    result = await answer_graph_question(request.question)

    return GraphQueryResponse(
        question=result.question,
        cypher=result.cypher,
        rows=result.rows,
        error=result.error,
    )
