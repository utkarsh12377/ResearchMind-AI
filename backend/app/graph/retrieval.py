"""GraphRAG: retrieval that follows relationships instead of similarity.

Vector search answers "what text looks like this question". It cannot answer
"which other papers used this dataset" or "what connects these two models",
because the answer is a path, not a passage. This module walks the graph from
entities named in the question and returns the papers those paths reach, so the
agent gets neighbours that no amount of embedding similarity would surface.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.graph.extraction import extract_with_gazetteer
from app.graph.schema import normalize_key
from app.models import ExtractedEntity, Paper, Workspace

logger = get_logger(__name__)

MAX_SEED_ENTITIES = 8
MAX_RELATED_PAPERS = 12
MIN_TERM_LENGTH = 3


@dataclass
class GraphPath:
    """One reason a paper was pulled in: a shared entity."""

    paper_id: uuid.UUID
    paper_title: str
    entity_label: str
    entity_name: str
    shared_with: list[str] = field(default_factory=list)

    def describe(self) -> str:
        return f"{self.paper_title} — via {self.entity_label.lower()} '{self.entity_name}'"


@dataclass
class GraphContext:
    seed_entities: list[tuple[str, str]] = field(default_factory=list)
    paths: list[GraphPath] = field(default_factory=list)

    @property
    def paper_ids(self) -> list[uuid.UUID]:
        seen: list[uuid.UUID] = []
        for path in self.paths:
            if path.paper_id not in seen:
                seen.append(path.paper_id)
        return seen

    def summary(self) -> str:
        if not self.paths:
            return "No graph connections found."
        entities = ", ".join(f"{name} ({label})" for label, name in self.seed_entities)
        return f"{len(self.paper_ids)} related paper(s) via {entities or 'shared entities'}"


async def _accessible_paper_ids(db: AsyncSession, user) -> list[uuid.UUID]:  # noqa: ANN001
    workspace_ids = (
        await db.scalars(select(Workspace.id).where(Workspace.owner_id == user.id))
    ).all()
    if not workspace_ids:
        return []
    return list(
        (await db.scalars(select(Paper.id).where(Paper.workspace_id.in_(workspace_ids)))).all()
    )


async def find_seed_entities(
    db: AsyncSession, query: str, paper_ids: list[uuid.UUID]
) -> list[ExtractedEntity]:
    """Entities from the user's own corpus that the question mentions.

    Seeding from the corpus rather than from the raw query matters: an entity
    nothing was indexed under is a dead end, and walking from it wastes a hop.
    """
    if not paper_ids:
        return []

    candidate_keys = {normalize_key(e.name) for e in extract_with_gazetteer(query).entities}
    candidate_keys.update(
        normalize_key(term)
        for term in query.split()
        if len(term) >= MIN_TERM_LENGTH
    )
    candidate_keys.discard("")
    if not candidate_keys:
        return []

    rows = (
        await db.scalars(
            select(ExtractedEntity)
            .where(
                ExtractedEntity.paper_id.in_(paper_ids),
                ExtractedEntity.key.in_(candidate_keys),
            )
            .order_by(ExtractedEntity.confidence.desc())
        )
    ).all()

    seen: set[tuple[str, str]] = set()
    seeds = []
    for row in rows:
        signature = (row.label, row.key)
        if signature in seen:
            continue
        seen.add(signature)
        seeds.append(row)
        if len(seeds) >= MAX_SEED_ENTITIES:
            break
    return seeds


async def expand(
    db: AsyncSession,
    user,  # noqa: ANN001
    query: str,
    *,
    exclude_paper_ids: list[uuid.UUID] | None = None,
    limit: int = MAX_RELATED_PAPERS,
) -> GraphContext:
    accessible = await _accessible_paper_ids(db, user)
    seeds = await find_seed_entities(db, query, accessible)
    if not seeds:
        return GraphContext()

    excluded = set(exclude_paper_ids or [])
    context = GraphContext(seed_entities=[(s.label, s.name) for s in seeds])

    for seed in seeds:
        siblings = (
            await db.execute(
                select(ExtractedEntity.paper_id, Paper.title)
                .join(Paper, Paper.id == ExtractedEntity.paper_id)
                .where(
                    ExtractedEntity.key == seed.key,
                    ExtractedEntity.label == seed.label,
                    ExtractedEntity.paper_id.in_(accessible),
                )
                .limit(limit)
            )
        ).all()

        for paper_id, title in siblings:
            if paper_id in excluded:
                continue
            context.paths.append(
                GraphPath(
                    paper_id=paper_id,
                    paper_title=title or "Untitled",
                    entity_label=seed.label,
                    entity_name=seed.name,
                )
            )

    context.paths = _rank_paths(context.paths)[:limit]
    logger.info(
        "graph_expanded",
        seeds=len(seeds),
        related_papers=len(context.paper_ids),
    )
    return context


def _rank_paths(paths: list[GraphPath]) -> list[GraphPath]:
    """Papers connected through more shared entities come first.

    A single shared metric like "accuracy" is weak evidence of relatedness; the
    same paper turning up through three different entities is not.
    """
    grouped: dict[uuid.UUID, GraphPath] = {}
    strength: dict[uuid.UUID, int] = {}

    for path in paths:
        existing = grouped.get(path.paper_id)
        if existing is None:
            grouped[path.paper_id] = path
            strength[path.paper_id] = 1
        else:
            strength[path.paper_id] += 1
            existing.shared_with.append(f"{path.entity_name} ({path.entity_label.lower()})")

    return sorted(grouped.values(), key=lambda p: strength[p.paper_id], reverse=True)


async def entity_cooccurrence(
    db: AsyncSession,
    paper_ids: list[uuid.UUID],
    *,
    label: str | None = None,
    limit: int = 25,
) -> list[tuple[str, str, int]]:
    """Most-mentioned entities across a set of papers, for the graph overview."""
    if not paper_ids:
        return []

    stmt = (
        select(
            ExtractedEntity.label,
            ExtractedEntity.name,
            func.count(func.distinct(ExtractedEntity.paper_id)).label("papers"),
        )
        .where(ExtractedEntity.paper_id.in_(paper_ids))
        .group_by(ExtractedEntity.label, ExtractedEntity.name)
        .order_by(func.count(func.distinct(ExtractedEntity.paper_id)).desc())
        .limit(limit)
    )
    if label:
        stmt = stmt.where(ExtractedEntity.label == label)

    return [(row[0], row[1], int(row[2])) for row in (await db.execute(stmt)).all()]
