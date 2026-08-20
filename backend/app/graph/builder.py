"""Projecting a parsed paper into the knowledge graph.

Runs after ingestion: the paper already has chunks, references, and metadata, so
this stage only has to decide what becomes a node, what becomes an edge, and
what gets mirrored back into Postgres for filtering.
"""

from __future__ import annotations

import uuid

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.graph.extraction import ExtractionResult, extract_entities
from app.graph.schema import GraphNode, GraphRelationship, NodeLabel, RelationType, normalize_key
from app.graph.store import GraphStore, get_graph_store
from app.llm.gateway import LLMGateway
from app.models import DocumentChunk, ExtractedEntity, Paper, PaperReference

logger = get_logger(__name__)

MAX_EXTRACTION_CHARS = 24000
AUTHOR_SEPARATORS = (";", ",", " and ", "&")


async def build_paper_graph(
    db: AsyncSession,
    paper_id: uuid.UUID,
    *,
    store: GraphStore | None = None,
    gateway: LLMGateway | None = None,
    use_llm: bool = True,
) -> ExtractionResult:
    store = store or get_graph_store()

    paper = await db.get(Paper, paper_id)
    if paper is None:
        raise ValueError(f"Paper {paper_id} does not exist")

    text = await _extraction_text(db, paper)
    result = await extract_entities(
        text,
        paper_title=paper.title or paper.original_filename,
        gateway=gateway,
        use_llm=use_llm,
    )

    paper_node = _paper_node(paper)
    nodes: list[GraphNode] = [paper_node]
    edges: list[GraphRelationship] = []

    for author in split_authors(paper.authors or ""):
        author_node = GraphNode(NodeLabel.AUTHOR.value, author, {"name": author})
        nodes.append(author_node)
        edges.append(
            GraphRelationship(RelationType.AUTHORED_BY.value, paper_node, author_node)
        )

    if paper.venue:
        venue_node = GraphNode(NodeLabel.VENUE.value, paper.venue, {"name": paper.venue})
        nodes.append(venue_node)
        edges.append(
            GraphRelationship(RelationType.PUBLISHED_IN.value, paper_node, venue_node)
        )

    entity_nodes: dict[str, GraphNode] = {}
    for mention in result.entities:
        node = GraphNode(
            mention.label,
            mention.name,
            {"name": mention.name, "confidence": mention.confidence},
        )
        entity_nodes[normalize_key(mention.name)] = node
        nodes.append(node)

    paper_title_key = normalize_key(paper.title or paper.original_filename)
    for relation in result.relations:
        source = _resolve_endpoint(relation.source, paper_node, paper_title_key, entity_nodes)
        target = _resolve_endpoint(relation.target, paper_node, paper_title_key, entity_nodes)
        if source is None or target is None:
            continue
        try:
            edges.append(
                GraphRelationship(
                    relation.type,
                    source,
                    target,
                    {"confidence": relation.confidence},
                )
            )
        except ValueError:
            logger.debug("relation_dropped_wrong_shape", type=relation.type)

    edges.extend(await _citation_edges(db, paper, paper_node, nodes))

    await store.upsert_nodes(nodes)
    await store.upsert_relationships(edges)
    await _mirror_entities(db, paper_id, result)

    logger.info(
        "paper_graph_built",
        paper_id=str(paper_id),
        nodes=len(nodes),
        edges=len(edges),
    )
    return result


def _paper_node(paper: Paper) -> GraphNode:
    return GraphNode(
        NodeLabel.PAPER.value,
        str(paper.id),
        {
            "name": paper.title or paper.original_filename,
            "paper_id": str(paper.id),
            "workspace_id": str(paper.workspace_id),
            "year": paper.published_year,
            "page_count": paper.page_count,
        },
    )


def _resolve_endpoint(
    surface: str,
    paper_node: GraphNode,
    paper_title_key: str,
    entity_nodes: dict[str, GraphNode],
) -> GraphNode | None:
    """Map a name in an extracted relation back to a node we actually created."""
    key = normalize_key(surface)
    if key == paper_title_key or key == paper_node.key:
        return paper_node
    return entity_nodes.get(key)


async def _citation_edges(
    db: AsyncSession,
    paper: Paper,
    paper_node: GraphNode,
    nodes: list[GraphNode],
) -> list[GraphRelationship]:
    """Link this paper to the works it cites.

    A reference only becomes an edge when the cited work is itself in the
    workspace. Creating placeholder Paper nodes for every bibliography entry
    would swamp the graph with thousands of stubs that nothing can be retrieved
    from.
    """
    references = (
        await db.scalars(select(PaperReference).where(PaperReference.paper_id == paper.id))
    ).all()
    if not references:
        return []

    candidates = (
        await db.scalars(
            select(Paper).where(
                Paper.workspace_id == paper.workspace_id, Paper.id != paper.id
            )
        )
    ).all()
    by_title = {normalize_key(p.title or ""): p for p in candidates if p.title}
    if not by_title:
        return []

    edges = []
    seen: set[uuid.UUID] = set()
    for reference in references:
        if not reference.title:
            continue
        cited = by_title.get(normalize_key(reference.title))
        if cited is None or cited.id in seen:
            continue
        seen.add(cited.id)
        cited_node = _paper_node(cited)
        nodes.append(cited_node)
        edges.append(
            GraphRelationship(
                RelationType.CITES.value,
                paper_node,
                cited_node,
                {"reference_id": str(reference.id)},
            )
        )
    return edges


async def _extraction_text(db: AsyncSession, paper: Paper) -> str:
    """Assemble the text extraction reads.

    Abstract first, then the earliest chunks. Entities that matter are almost
    always introduced early, and this keeps a 40-page paper inside one prompt.
    """
    parts = []
    if paper.abstract:
        parts.append(paper.abstract)

    chunks = (
        await db.scalars(
            select(DocumentChunk)
            .where(DocumentChunk.paper_id == paper.id)
            .order_by(DocumentChunk.chunk_index)
            .limit(30)
        )
    ).all()
    parts.extend(chunk.content for chunk in chunks)

    return "\n\n".join(parts)[:MAX_EXTRACTION_CHARS]


async def _mirror_entities(
    db: AsyncSession, paper_id: uuid.UUID, result: ExtractionResult
) -> None:
    await db.execute(delete(ExtractedEntity).where(ExtractedEntity.paper_id == paper_id))
    await db.flush()

    seen: set[tuple[str, str]] = set()
    for mention in result.entities:
        signature = (mention.label, mention.key)
        if signature in seen:
            continue
        seen.add(signature)
        db.add(
            ExtractedEntity(
                paper_id=paper_id,
                label=mention.label,
                key=mention.key[:255],
                name=mention.name[:255],
                confidence=mention.confidence,
                evidence=mention.evidence or None,
            )
        )
    await db.commit()


def split_authors(raw: str) -> list[str]:
    """Split an author string on the separators PDFs actually use."""
    if not raw.strip():
        return []

    working = raw
    for separator in AUTHOR_SEPARATORS:
        working = working.replace(separator, "|")

    authors = []
    for candidate in working.split("|"):
        name = candidate.strip(" .")
        if len(name) < 3 or len(name) > 80:
            continue
        if not any(ch.isalpha() for ch in name):
            continue
        authors.append(name)
    return authors[:50]


async def remove_paper_from_graph(
    db: AsyncSession, paper_id: uuid.UUID, *, store: GraphStore | None = None
) -> int:
    store = store or get_graph_store()
    removed = await store.delete_paper(str(paper_id))
    await db.execute(delete(ExtractedEntity).where(ExtractedEntity.paper_id == paper_id))
    await db.commit()
    return removed
