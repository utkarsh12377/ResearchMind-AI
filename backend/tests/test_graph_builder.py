"""Graph construction from real ingested papers, not synthetic rows."""

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.graph.builder import build_paper_graph, remove_paper_from_graph, split_authors
from app.graph.schema import NodeLabel, RelationType
from app.graph.store import InMemoryGraphStore
from app.models import ExtractedEntity, Paper
from app.worker.tasks import _process_paper
from tests.factories import build_pdf
from tests.test_ingestion_task import _store_paper


@pytest.fixture
def store() -> InMemoryGraphStore:
    return InMemoryGraphStore()


async def _ingest(db: AsyncSession, *, filename: str = "graph.pdf", **pdf_kwargs) -> Paper:
    paper = await _store_paper(db, build_pdf(**pdf_kwargs), filename)
    await _process_paper(db, paper.id)
    return paper


def test_split_authors_handles_the_common_separators() -> None:
    assert split_authors("Ada Lovelace, Alan Turing; Grace Hopper") == [
        "Ada Lovelace",
        "Alan Turing",
        "Grace Hopper",
    ]


def test_split_authors_handles_the_and_conjunction() -> None:
    assert split_authors("Ada Lovelace and Alan Turing") == ["Ada Lovelace", "Alan Turing"]


def test_split_authors_drops_fragments_and_page_noise() -> None:
    assert split_authors("A, Ada Lovelace, 1") == ["Ada Lovelace"]


def test_split_authors_on_empty_input() -> None:
    assert split_authors("   ") == []


@pytest.mark.asyncio
async def test_building_creates_a_paper_node(db_session: AsyncSession, store) -> None:  # noqa: ANN001
    paper = await _ingest(db_session, body="We evaluate on SQuAD using BERT.")

    await build_paper_graph(db_session, paper.id, store=store, use_llm=False)

    counts = await store.counts()
    assert counts["Paper"] == 1


@pytest.mark.asyncio
async def test_building_links_the_paper_to_its_entities(
    db_session: AsyncSession, store  # noqa: ANN001
) -> None:
    paper = await _ingest(db_session, body="We fine-tune BERT on the SQuAD dataset.")

    await build_paper_graph(db_session, paper.id, store=store, use_llm=False)

    neighbors = await store.neighbors(NodeLabel.PAPER.value, str(paper.id))
    assert any(hit.node.key == "squad" for hit in neighbors)
    assert any(hit.relationship == RelationType.USES_DATASET.value for hit in neighbors)


@pytest.mark.asyncio
async def test_building_mirrors_entities_into_postgres(
    db_session: AsyncSession, store  # noqa: ANN001
) -> None:
    paper = await _ingest(db_session, body="Results on GLUE with accuracy reported.")

    await build_paper_graph(db_session, paper.id, store=store, use_llm=False)

    rows = (
        await db_session.scalars(
            select(ExtractedEntity).where(ExtractedEntity.paper_id == paper.id)
        )
    ).all()
    assert {row.key for row in rows} >= {"glue"}


@pytest.mark.asyncio
async def test_rebuilding_replaces_rather_than_duplicates(
    db_session: AsyncSession, store  # noqa: ANN001
) -> None:
    paper = await _ingest(db_session, body="Trained on ImageNet.")

    await build_paper_graph(db_session, paper.id, store=store, use_llm=False)
    await build_paper_graph(db_session, paper.id, store=store, use_llm=False)

    rows = (
        await db_session.scalars(
            select(ExtractedEntity).where(ExtractedEntity.paper_id == paper.id)
        )
    ).all()
    keys = [row.key for row in rows]
    assert len(keys) == len(set(keys))


@pytest.mark.asyncio
async def test_authors_become_nodes(db_session: AsyncSession, store) -> None:  # noqa: ANN001
    paper = await _ingest(db_session)
    paper.authors = "Ada Lovelace, Alan Turing"
    await db_session.commit()

    await build_paper_graph(db_session, paper.id, store=store, use_llm=False)

    authors = await store.find_nodes(NodeLabel.AUTHOR.value)
    assert {author.key for author in authors} == {"ada lovelace", "alan turing"}


@pytest.mark.asyncio
async def test_two_papers_sharing_a_dataset_are_connected(
    db_session: AsyncSession, store  # noqa: ANN001
) -> None:
    """The whole point of the graph: a path between papers that never cite each other."""
    first = await _ingest(db_session, filename="a.pdf", body="We train on ImageNet.")
    second = await _ingest(db_session, filename="b.pdf", body="Also evaluated on ImageNet.")

    await build_paper_graph(db_session, first.id, store=store, use_llm=False)
    await build_paper_graph(db_session, second.id, store=store, use_llm=False)

    reachable = await store.neighbors(NodeLabel.PAPER.value, str(first.id), depth=2)
    assert str(second.id) in {hit.node.key for hit in reachable}


@pytest.mark.asyncio
async def test_venue_is_linked_when_known(db_session: AsyncSession, store) -> None:  # noqa: ANN001
    paper = await _ingest(db_session)
    paper.venue = "NeurIPS"
    await db_session.commit()

    await build_paper_graph(db_session, paper.id, store=store, use_llm=False)

    venues = await store.find_nodes(NodeLabel.VENUE.value)
    assert [venue.key for venue in venues] == ["neurips"]


@pytest.mark.asyncio
async def test_building_an_unknown_paper_raises(db_session: AsyncSession, store) -> None:  # noqa: ANN001
    import uuid

    with pytest.raises(ValueError):
        await build_paper_graph(db_session, uuid.uuid4(), store=store, use_llm=False)


@pytest.mark.asyncio
async def test_removing_a_paper_clears_both_stores(
    db_session: AsyncSession, store  # noqa: ANN001
) -> None:
    paper = await _ingest(db_session, body="Evaluated on GLUE.")
    await build_paper_graph(db_session, paper.id, store=store, use_llm=False)

    await remove_paper_from_graph(db_session, paper.id, store=store)

    rows = (
        await db_session.scalars(
            select(ExtractedEntity).where(ExtractedEntity.paper_id == paper.id)
        )
    ).all()
    assert rows == []
    assert "Paper" not in await store.counts()
