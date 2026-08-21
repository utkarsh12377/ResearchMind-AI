"""Structural questions about a corpus that only the graph can answer.

Everything here is aggregate rather than per-document: who works with whom,
which datasets a field converged on, which papers sit at the centre of the
citation network. These read from the relational mirror so they work on the
default stack, and stay cheap enough to serve a dashboard.
"""

from __future__ import annotations

import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.graph.builder import split_authors
from app.graph.schema import normalize_key
from app.models import ExtractedEntity, Paper, PaperReference

logger = get_logger(__name__)


@dataclass
class GraphEdgeView:
    source: str
    target: str
    weight: int = 1
    kind: str = "related"


@dataclass
class GraphNodeView:
    id: str
    label: str
    kind: str
    weight: int = 1
    metadata: dict = field(default_factory=dict)


@dataclass
class NetworkView:
    nodes: list[GraphNodeView] = field(default_factory=list)
    edges: list[GraphEdgeView] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.nodes


async def coauthorship_network(
    db: AsyncSession, paper_ids: list[uuid.UUID], *, limit: int = 60
) -> NetworkView:
    if not paper_ids:
        return NetworkView()

    rows = (
        await db.execute(select(Paper.id, Paper.authors).where(Paper.id.in_(paper_ids)))
    ).all()

    paper_counts: Counter[str] = Counter()
    pair_counts: Counter[tuple[str, str]] = Counter()

    for _, authors in rows:
        names = split_authors(authors or "")
        for name in names:
            paper_counts[name] += 1
        for i, first in enumerate(names):
            for second in names[i + 1 :]:
                pair_counts[tuple(sorted((first, second)))] += 1

    top_authors = [name for name, _ in paper_counts.most_common(limit)]
    keep = set(top_authors)

    return NetworkView(
        nodes=[
            GraphNodeView(id=name, label=name, kind="Author", weight=paper_counts[name])
            for name in top_authors
        ],
        edges=[
            GraphEdgeView(source=a, target=b, weight=count, kind="COAUTHORED")
            for (a, b), count in pair_counts.most_common()
            if a in keep and b in keep
        ],
    )


async def citation_network(
    db: AsyncSession, paper_ids: list[uuid.UUID], *, limit: int = 80
) -> NetworkView:
    """Citations resolved inside the corpus.

    References to work that isn't in the library are deliberately dropped: an
    edge to a node with no content behind it can't be explored, so it would add
    visual noise without adding a path anyone can follow.
    """
    if not paper_ids:
        return NetworkView()

    papers = (
        await db.execute(select(Paper.id, Paper.title).where(Paper.id.in_(paper_ids)))
    ).all()
    by_title = {normalize_key(title or ""): pid for pid, title in papers if title}
    titles = {str(pid): (title or "Untitled") for pid, title in papers}

    references = (
        await db.scalars(
            select(PaperReference).where(PaperReference.paper_id.in_(paper_ids)).limit(2000)
        )
    ).all()

    edges: list[GraphEdgeView] = []
    in_degree: Counter[str] = Counter()
    seen: set[tuple[str, str]] = set()

    for reference in references:
        if not reference.title:
            continue
        target = by_title.get(normalize_key(reference.title))
        if target is None or target == reference.paper_id:
            continue
        pair = (str(reference.paper_id), str(target))
        if pair in seen:
            continue
        seen.add(pair)
        edges.append(GraphEdgeView(source=pair[0], target=pair[1], kind="CITES"))
        in_degree[pair[1]] += 1

    nodes = [
        GraphNodeView(
            id=str(pid),
            label=titles[str(pid)],
            kind="Paper",
            weight=in_degree.get(str(pid), 0) + 1,
            metadata={"citations_within_corpus": in_degree.get(str(pid), 0)},
        )
        for pid, _ in papers
    ]
    nodes.sort(key=lambda n: n.weight, reverse=True)
    return NetworkView(nodes=nodes[:limit], edges=edges)


async def entity_network(
    db: AsyncSession,
    paper_ids: list[uuid.UUID],
    *,
    labels: list[str] | None = None,
    min_papers: int = 2,
    limit: int = 60,
) -> NetworkView:
    """Papers linked to the entities they share."""
    if not paper_ids:
        return NetworkView()

    stmt = select(ExtractedEntity, Paper.title).join(Paper, Paper.id == ExtractedEntity.paper_id)
    stmt = stmt.where(ExtractedEntity.paper_id.in_(paper_ids))
    if labels:
        stmt = stmt.where(ExtractedEntity.label.in_(labels))

    rows = (await db.execute(stmt.limit(5000))).all()

    entity_papers: dict[tuple[str, str], set[str]] = defaultdict(set)
    entity_names: dict[tuple[str, str], str] = {}
    paper_titles: dict[str, str] = {}

    for entity, title in rows:
        signature = (entity.label, entity.key)
        entity_papers[signature].add(str(entity.paper_id))
        entity_names[signature] = entity.name
        paper_titles[str(entity.paper_id)] = title or "Untitled"

    shared = {
        signature: papers
        for signature, papers in entity_papers.items()
        if len(papers) >= min_papers
    }
    ranked = sorted(shared.items(), key=lambda item: len(item[1]), reverse=True)[:limit]

    nodes: list[GraphNodeView] = []
    edges: list[GraphEdgeView] = []
    involved: set[str] = set()

    for (label, key), papers in ranked:
        entity_id = f"{label}:{key}"
        nodes.append(
            GraphNodeView(
                id=entity_id,
                label=entity_names[(label, key)],
                kind=label,
                weight=len(papers),
            )
        )
        for paper_id in papers:
            involved.add(paper_id)
            edges.append(GraphEdgeView(source=paper_id, target=entity_id, kind="MENTIONS"))

    nodes.extend(
        GraphNodeView(id=pid, label=paper_titles.get(pid, "Untitled"), kind="Paper")
        for pid in involved
    )
    return NetworkView(nodes=nodes, edges=edges)


async def entity_adoption_by_year(
    db: AsyncSession,
    paper_ids: list[uuid.UUID],
    *,
    label: str,
    top_n: int = 8,
) -> dict[str, dict[int, int]]:
    """How often each entity of a label appears per publication year."""
    if not paper_ids:
        return {}

    rows = (
        await db.execute(
            select(ExtractedEntity.name, Paper.published_year, func.count().label("total"))
            .join(Paper, Paper.id == ExtractedEntity.paper_id)
            .where(
                ExtractedEntity.paper_id.in_(paper_ids),
                ExtractedEntity.label == label,
                Paper.published_year.is_not(None),
            )
            .group_by(ExtractedEntity.name, Paper.published_year)
        )
    ).all()

    totals: Counter[str] = Counter()
    series: dict[str, dict[int, int]] = defaultdict(dict)
    for name, year, total in rows:
        series[name][int(year)] = int(total)
        totals[name] += int(total)

    return {name: dict(sorted(series[name].items())) for name, _ in totals.most_common(top_n)}


async def most_connected_papers(
    db: AsyncSession, paper_ids: list[uuid.UUID], *, limit: int = 10
) -> list[tuple[uuid.UUID, str, int]]:
    """Papers ranked by how many distinct entities they share with the corpus."""
    if not paper_ids:
        return []

    rows = (
        await db.execute(
            select(
                ExtractedEntity.paper_id,
                Paper.title,
                func.count(func.distinct(ExtractedEntity.key)).label("degree"),
            )
            .join(Paper, Paper.id == ExtractedEntity.paper_id)
            .where(ExtractedEntity.paper_id.in_(paper_ids))
            .group_by(ExtractedEntity.paper_id, Paper.title)
            .order_by(func.count(func.distinct(ExtractedEntity.key)).desc())
            .limit(limit)
        )
    ).all()

    return [(row[0], row[1] or "Untitled", int(row[2])) for row in rows]
