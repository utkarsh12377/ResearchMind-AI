"""Literature review generation: outline, draft, citation weaving, export.

A review is not one long generation. Written in a single pass the model loses
track of which paper said what by the third paragraph, and citations drift off
their sources. Splitting the work into an outline stage and per-section drafting
keeps each generation small enough that its sources fit in context and can be
checked against what it produced.
"""

from __future__ import annotations

import asyncio
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.llm.gateway import LLMGateway, Message, Role, Usage
from app.llm.json_output import load_json_object
from app.llm.prompts import REVIEW_OUTLINE, REVIEW_SECTION
from app.llm.rag import extract_citations
from app.models import Paper
from app.retrieval.service import RetrievalFilters, RetrievedChunk, retrieve

logger = get_logger(__name__)

MAX_PAPERS = 40
SOURCES_PER_SECTION = 8
MAX_SECTIONS = 8
DEFAULT_SECTIONS = (
    ("Overview", "What problem this area addresses and why it matters"),
    ("Approaches", "The main methods and how they differ"),
    ("Evaluation", "Datasets, benchmarks, and reported results"),
    ("Open problems", "Limitations the papers themselves identify"),
)


@dataclass
class ReviewSource:
    index: int
    paper_id: uuid.UUID
    title: str
    authors: str | None = None
    year: int | None = None

    @property
    def citation(self) -> str:
        parts = [self.title]
        if self.authors:
            parts.insert(0, self.authors)
        if self.year:
            parts.append(str(self.year))
        return ". ".join(parts)

    def bibtex_key(self) -> str:
        first_author = (self.authors or "unknown").split(",")[0].split()[-1:] or ["unknown"]
        slug = re.sub(r"[^a-z0-9]", "", first_author[0].lower())[:20] or "unknown"
        title_word = re.sub(r"[^a-z0-9]", "", self.title.split(" ")[0].lower())[:12]
        return f"{slug}{self.year or ''}{title_word}"


@dataclass
class ReviewSection:
    heading: str
    focus: str
    text: str = ""
    source_indices: list[int] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()


@dataclass
class LiteratureReview:
    topic: str
    title: str
    sections: list[ReviewSection] = field(default_factory=list)
    sources: list[ReviewSource] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def cited_sources(self) -> list[ReviewSource]:
        cited = {index for section in self.sections for index in section.source_indices}
        return [source for source in self.sources if source.index in cited]

    @property
    def word_count(self) -> int:
        return sum(len(section.text.split()) for section in self.sections)

    def to_markdown(self) -> str:
        lines = [f"# {self.title}", ""]
        for section in self.sections:
            lines.append(f"## {section.heading}")
            lines.append("")
            lines.append(section.text.strip() or "_No supporting sources were found._")
            lines.append("")

        cited = self.cited_sources
        if cited:
            lines.append("## References")
            lines.append("")
            for source in cited:
                lines.append(f"{source.index}. {source.citation}")
            lines.append("")

        lines.append(
            f"_Generated {self.generated_at.strftime('%Y-%m-%d')} from "
            f"{len(self.sources)} paper(s)._"
        )
        return "\n".join(lines)

    def to_bibtex(self) -> str:
        entries = []
        for source in self.cited_sources:
            fields = [f"  title = {{{source.title}}}"]
            if source.authors:
                fields.append(f"  author = {{{source.authors}}}")
            if source.year:
                fields.append(f"  year = {{{source.year}}}")
            body = ",\n".join(fields)
            entries.append(f"@article{{{source.bibtex_key()},\n{body}\n}}")
        return "\n\n".join(entries)


async def _load_sources(db: AsyncSession, paper_ids: list[uuid.UUID]) -> list[ReviewSource]:
    papers = (
        await db.scalars(select(Paper).where(Paper.id.in_(paper_ids)).limit(MAX_PAPERS))
    ).all()
    ordered = sorted(papers, key=lambda p: (p.published_year or 0), reverse=True)
    return [
        ReviewSource(
            index=index,
            paper_id=paper.id,
            title=paper.title or paper.original_filename,
            authors=paper.authors,
            year=paper.published_year,
        )
        for index, paper in enumerate(ordered, start=1)
    ]


def _format_paper_list(sources: list[ReviewSource]) -> str:
    return "\n".join(f"[{s.index}] {s.citation}" for s in sources)


async def plan_outline(
    topic: str, sources: list[ReviewSource], *, gateway: LLMGateway
) -> tuple[str, list[ReviewSection], Usage]:
    """Ask for a thematic outline, falling back to a standard structure."""
    usage = Usage()
    fallback_title = f"A Review of {topic}"

    try:
        completion = await gateway.complete(
            [
                Message(Role.SYSTEM, REVIEW_OUTLINE.system),
                Message(
                    Role.USER,
                    REVIEW_OUTLINE.render(topic=topic, papers=_format_paper_list(sources)),
                ),
            ],
            temperature=0.3,
            max_tokens=900,
        )
        usage = completion.usage
    except Exception as exc:  # noqa: BLE001
        logger.warning("review_outline_failed", error=str(exc))
        return fallback_title, _default_sections(), usage

    payload = load_json_object(completion.text)
    if not payload or not payload.get("sections"):
        return fallback_title, _default_sections(), usage

    sections = []
    for item in payload["sections"][:MAX_SECTIONS]:
        if not isinstance(item, dict) or not item.get("heading"):
            continue
        sections.append(
            ReviewSection(
                heading=str(item["heading"])[:200],
                focus=str(item.get("focus", ""))[:500],
            )
        )

    if not sections:
        sections = _default_sections()

    return str(payload.get("title") or fallback_title)[:300], sections, usage


def _default_sections() -> list[ReviewSection]:
    return [ReviewSection(heading=heading, focus=focus) for heading, focus in DEFAULT_SECTIONS]


def _format_sources_for_section(
    chunks: list[RetrievedChunk], sources: list[ReviewSource]
) -> tuple[str, list[int]]:
    """Render retrieved passages using the review's global source numbering.

    Section-local numbering would be simpler to generate but would make the
    citations meaningless once sections are concatenated, since [1] would refer
    to a different paper in every section.
    """
    by_paper = {source.paper_id: source.index for source in sources}
    blocks: list[str] = []
    used: list[int] = []

    for chunk in chunks:
        index = by_paper.get(chunk.paper_id)
        if index is None:
            continue
        if index not in used:
            used.append(index)
        section = chunk.section_path or "body"
        blocks.append(f"[{index}] {chunk.paper_title} ({section})\n{chunk.content.strip()}")

    return "\n\n".join(blocks), used


async def _draft_section(
    db: AsyncSession,
    user,  # noqa: ANN001
    topic: str,
    section: ReviewSection,
    sources: list[ReviewSource],
    *,
    gateway: LLMGateway,
) -> ReviewSection:
    query = f"{topic}. {section.heading}. {section.focus}".strip()
    chunks = await retrieve(
        db,
        user,
        query,
        filters=RetrievalFilters(paper_ids=[s.paper_id for s in sources]),
        limit=SOURCES_PER_SECTION,
    )

    if not chunks:
        section.text = "The available papers do not cover this aspect."
        return section

    rendered, available = _format_sources_for_section(chunks, sources)
    try:
        completion = await gateway.complete(
            [
                Message(Role.SYSTEM, REVIEW_SECTION.system),
                Message(
                    Role.USER,
                    REVIEW_SECTION.render(
                        topic=topic,
                        heading=section.heading,
                        focus=section.focus,
                        sources=rendered,
                    ),
                ),
            ],
            temperature=0.3,
            max_tokens=1200,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("review_section_failed", heading=section.heading, error=str(exc))
        section.text = "This section could not be generated."
        return section

    section.text = completion.text.strip()
    section.source_indices = _resolve_citations(section.text, available, sources)
    return section


def _resolve_citations(
    text: str, available: list[int], sources: list[ReviewSource]
) -> list[int]:
    """Keep only citation markers that point at a source this section was given."""
    cited = extract_citations(text, source_count=len(sources))
    allowed = set(available)
    return [index for index in cited if index in allowed]


async def generate_literature_review(
    db: AsyncSession,
    user,  # noqa: ANN001
    topic: str,
    paper_ids: list[uuid.UUID],
    *,
    gateway: LLMGateway,
    max_sections: int = MAX_SECTIONS,
) -> LiteratureReview:
    sources = await _load_sources(db, paper_ids)
    if not sources:
        return LiteratureReview(topic=topic, title=f"A Review of {topic}")

    title, sections, usage = await plan_outline(topic, sources, gateway=gateway)
    sections = sections[:max_sections]

    drafted = await asyncio.gather(
        *(
            _draft_section(db, user, topic, section, sources, gateway=gateway)
            for section in sections
        )
    )

    review = LiteratureReview(
        topic=topic,
        title=title,
        sections=list(drafted),
        sources=sources,
        usage=usage,
    )
    logger.info(
        "literature_review_generated",
        topic=topic[:80],
        sections=len(review.sections),
        words=review.word_count,
        cited=len(review.cited_sources),
    )
    return review
