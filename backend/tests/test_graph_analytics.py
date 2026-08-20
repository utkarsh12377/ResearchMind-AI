import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.graph.analytics import (
    citation_network,
    coauthorship_network,
    entity_adoption_by_year,
    entity_network,
    most_connected_papers,
)
from app.graph.builder import build_paper_graph
from app.graph.store import InMemoryGraphStore
from app.models import Paper, PaperReference
from app.worker.tasks import _process_paper
from tests.factories import build_pdf
from tests.test_ingestion_task import _store_paper


@pytest.fixture
def store() -> InMemoryGraphStore:
    return InMemoryGraphStore()


async def _ingest(
    db: AsyncSession,
    store,  # noqa: ANN001
    *,
    filename: str,
    authors: str | None = None,
    year: int | None = None,
    **pdf_kwargs,
) -> Paper:
    paper = await _store_paper(db, build_pdf(**pdf_kwargs), filename)
    await _process_paper(db, paper.id)
    if authors is not None:
        paper.authors = authors
    if year is not None:
        paper.published_year = year
    await db.commit()
    await build_paper_graph(db, paper.id, store=store, use_llm=False)
    return paper


async def _ids(db: AsyncSession) -> list:
    return list((await db.scalars(select(Paper.id))).all())


@pytest.mark.asyncio
async def test_coauthorship_links_authors_on_the_same_paper(
    db_session: AsyncSession, store  # noqa: ANN001
) -> None:
    await _ingest(db_session, store, filename="a.pdf", authors="Ada Lovelace, Alan Turing")

    network = await coauthorship_network(db_session, await _ids(db_session))

    assert {node.label for node in network.nodes} == {"Ada Lovelace", "Alan Turing"}
    assert len(network.edges) == 1


@pytest.mark.asyncio
async def test_coauthorship_weights_repeat_collaborations(
    db_session: AsyncSession, store  # noqa: ANN001
) -> None:
    await _ingest(db_session, store, filename="a.pdf", authors="Ada Lovelace, Alan Turing")
    await _ingest(db_session, store, filename="b.pdf", authors="Ada Lovelace, Alan Turing")

    network = await coauthorship_network(db_session, await _ids(db_session))

    assert network.edges[0].weight == 2


@pytest.mark.asyncio
async def test_a_sole_author_produces_no_edges(
    db_session: AsyncSession, store  # noqa: ANN001
) -> None:
    await _ingest(db_session, store, filename="a.pdf", authors="Ada Lovelace")

    network = await coauthorship_network(db_session, await _ids(db_session))

    assert len(network.nodes) == 1
    assert network.edges == []


@pytest.mark.asyncio
async def test_networks_are_empty_without_papers(db_session: AsyncSession) -> None:
    assert (await coauthorship_network(db_session, [])).is_empty
    assert (await citation_network(db_session, [])).is_empty
    assert (await entity_network(db_session, [])).is_empty


@pytest.mark.asyncio
async def test_citations_resolve_only_inside_the_corpus(
    db_session: AsyncSession, store  # noqa: ANN001
) -> None:
    """A reference to a paper nobody uploaded is an edge to nowhere."""
    cited = await _ingest(db_session, store, filename="cited.pdf", title="Deep Retrieval Methods")
    citing = await _ingest(db_session, store, filename="citing.pdf", title="A Follow Up Study")

    db_session.add(
        PaperReference(paper_id=citing.id, raw_text="x", title="Deep Retrieval Methods", order=1)
    )
    db_session.add(
        PaperReference(paper_id=citing.id, raw_text="y", title="Some Paper We Never Saw", order=2)
    )
    await db_session.commit()

    network = await citation_network(db_session, await _ids(db_session))

    assert len(network.edges) == 1
    assert network.edges[0].target == str(cited.id)


@pytest.mark.asyncio
async def test_citation_nodes_carry_their_in_corpus_citation_count(
    db_session: AsyncSession, store  # noqa: ANN001
) -> None:
    cited = await _ingest(db_session, store, filename="cited.pdf", title="Deep Retrieval Methods")
    citing = await _ingest(db_session, store, filename="citing.pdf", title="A Follow Up Study")
    db_session.add(
        PaperReference(paper_id=citing.id, raw_text="x", title="Deep Retrieval Methods", order=1)
    )
    await db_session.commit()

    network = await citation_network(db_session, await _ids(db_session))

    top = next(node for node in network.nodes if node.id == str(cited.id))
    assert top.metadata["citations_within_corpus"] == 1


@pytest.mark.asyncio
async def test_entity_network_only_keeps_shared_entities(
    db_session: AsyncSession, store  # noqa: ANN001
) -> None:
    await _ingest(db_session, store, filename="a.pdf", body="Evaluated on GLUE.")
    await _ingest(db_session, store, filename="b.pdf", body="Also evaluated on GLUE.")
    await _ingest(db_session, store, filename="c.pdf", body="Unrelated work on ImageNet.")

    network = await entity_network(db_session, await _ids(db_session), min_papers=2)

    kinds = {node.kind for node in network.nodes}
    labels = {node.label for node in network.nodes if node.kind != "Paper"}
    assert "Dataset" in kinds
    assert "imagenet" not in labels


@pytest.mark.asyncio
async def test_entity_network_can_be_filtered_by_label(
    db_session: AsyncSession, store  # noqa: ANN001
) -> None:
    await _ingest(db_session, store, filename="a.pdf", body="BERT on GLUE with accuracy.")
    await _ingest(db_session, store, filename="b.pdf", body="BERT on GLUE with accuracy.")

    network = await entity_network(db_session, await _ids(db_session), labels=["Dataset"])

    non_paper = {node.kind for node in network.nodes if node.kind != "Paper"}
    assert non_paper == {"Dataset"}


@pytest.mark.asyncio
async def test_adoption_is_bucketed_by_publication_year(
    db_session: AsyncSession, store  # noqa: ANN001
) -> None:
    await _ingest(db_session, store, filename="a.pdf", year=2019, body="Evaluated on GLUE.")
    await _ingest(db_session, store, filename="b.pdf", year=2023, body="Also on GLUE.")

    adoption = await entity_adoption_by_year(db_session, await _ids(db_session), label="Dataset")

    assert adoption["glue"] == {2019: 1, 2023: 1}


@pytest.mark.asyncio
async def test_adoption_ignores_undated_papers(
    db_session: AsyncSession, store  # noqa: ANN001
) -> None:
    await _ingest(db_session, store, filename="a.pdf", body="Evaluated on GLUE.")

    assert await entity_adoption_by_year(db_session, await _ids(db_session), label="Dataset") == {}


@pytest.mark.asyncio
async def test_most_connected_ranks_by_distinct_entities(
    db_session: AsyncSession, store  # noqa: ANN001
) -> None:
    rich = await _ingest(
        db_session, store, filename="rich.pdf", body="BERT on SQuAD and GLUE with accuracy and F1."
    )
    await _ingest(db_session, store, filename="thin.pdf", body="Nothing notable here.")

    ranked = await most_connected_papers(db_session, await _ids(db_session))

    assert ranked[0][0] == rich.id
    assert ranked[0][2] >= 2


@pytest.mark.asyncio
async def test_most_connected_on_an_empty_corpus(db_session: AsyncSession) -> None:
    assert await most_connected_papers(db_session, []) == []
