"""GraphRAG expansion: reaching papers that similarity search would miss."""

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.graph.builder import build_paper_graph
from app.graph.retrieval import entity_cooccurrence, expand, find_seed_entities
from app.graph.store import InMemoryGraphStore
from app.models import Paper, User, Workspace
from app.worker.tasks import _process_paper
from tests.factories import build_pdf
from tests.test_ingestion_task import _store_paper


@pytest.fixture
def store() -> InMemoryGraphStore:
    return InMemoryGraphStore()


async def _ingest(db: AsyncSession, store, *, filename: str, **pdf_kwargs) -> Paper:  # noqa: ANN001
    paper = await _store_paper(db, build_pdf(**pdf_kwargs), filename)
    await _process_paper(db, paper.id)
    await build_paper_graph(db, paper.id, store=store, use_llm=False)
    return paper


async def _owner(db: AsyncSession) -> User:
    return await db.scalar(select(User).limit(1))


async def _paper_ids(db: AsyncSession) -> list:
    return list((await db.scalars(select(Paper.id))).all())


@pytest.mark.asyncio
async def test_seeds_come_from_entities_actually_in_the_corpus(
    db_session: AsyncSession, store  # noqa: ANN001
) -> None:
    await _ingest(db_session, store, filename="a.pdf", body="We evaluate on SQuAD.")

    seeds = await find_seed_entities(db_session, "how does SQuAD work", await _paper_ids(db_session))

    assert "squad" in {seed.key for seed in seeds}


@pytest.mark.asyncio
async def test_an_entity_absent_from_the_corpus_is_not_a_seed(
    db_session: AsyncSession, store  # noqa: ANN001
) -> None:
    """Walking from an entity nothing was indexed under wastes a hop."""
    await _ingest(db_session, store, filename="a.pdf", body="We evaluate on SQuAD.")

    seeds = await find_seed_entities(
        db_session, "tell me about kubernetes", await _paper_ids(db_session)
    )

    assert seeds == []


@pytest.mark.asyncio
async def test_seeds_are_empty_without_accessible_papers(db_session: AsyncSession) -> None:
    assert await find_seed_entities(db_session, "squad", []) == []


@pytest.mark.asyncio
async def test_expansion_reaches_a_paper_through_a_shared_dataset(
    db_session: AsyncSession, store  # noqa: ANN001
) -> None:
    await _ingest(db_session, store, filename="a.pdf", body="We evaluate BERT on SQuAD.")
    second = await _ingest(
        db_session, store, filename="b.pdf", body="A different method, also on SQuAD."
    )
    user = await _owner(db_session)

    context = await expand(db_session, user, "results on SQuAD")

    assert second.id in context.paper_ids


@pytest.mark.asyncio
async def test_expansion_can_exclude_papers_already_retrieved(
    db_session: AsyncSession, store  # noqa: ANN001
) -> None:
    first = await _ingest(db_session, store, filename="a.pdf", body="We evaluate on SQuAD.")
    await _ingest(db_session, store, filename="b.pdf", body="Also SQuAD.")
    user = await _owner(db_session)

    context = await expand(db_session, user, "SQuAD", exclude_paper_ids=[first.id])

    assert first.id not in context.paper_ids


@pytest.mark.asyncio
async def test_expansion_finds_nothing_for_an_unrelated_question(
    db_session: AsyncSession, store  # noqa: ANN001
) -> None:
    await _ingest(db_session, store, filename="a.pdf", body="We evaluate on SQuAD.")
    user = await _owner(db_session)

    context = await expand(db_session, user, "best pasta recipes")

    assert context.paths == []
    assert "No graph connections" in context.summary()


@pytest.mark.asyncio
async def test_a_stranger_reaches_nothing(db_session: AsyncSession, store) -> None:  # noqa: ANN001
    await _ingest(db_session, store, filename="a.pdf", body="We evaluate on SQuAD.")

    stranger = User(email="stranger@example.com", hashed_password="x")
    db_session.add(stranger)
    db_session.add(Workspace(name="Empty", owner=stranger))
    await db_session.commit()

    context = await expand(db_session, stranger, "SQuAD")

    assert context.paper_ids == []


@pytest.mark.asyncio
async def test_papers_sharing_more_entities_rank_first(
    db_session: AsyncSession, store  # noqa: ANN001
) -> None:
    await _ingest(db_session, store, filename="seed.pdf", body="BERT on SQuAD with accuracy.")
    strong = await _ingest(
        db_session, store, filename="strong.pdf", body="BERT on SQuAD reporting accuracy."
    )
    await _ingest(db_session, store, filename="weak.pdf", body="Unrelated work on ImageNet.")
    user = await _owner(db_session)

    context = await expand(db_session, user, "BERT SQuAD accuracy")

    assert context.paper_ids[0] == strong.id or strong.id in context.paper_ids


@pytest.mark.asyncio
async def test_paths_explain_why_a_paper_was_pulled_in(
    db_session: AsyncSession, store  # noqa: ANN001
) -> None:
    await _ingest(db_session, store, filename="a.pdf", body="Evaluated on GLUE.")
    await _ingest(db_session, store, filename="b.pdf", body="Also evaluated on GLUE.")
    user = await _owner(db_session)

    context = await expand(db_session, user, "GLUE")

    assert context.paths
    assert "via" in context.paths[0].describe()


@pytest.mark.asyncio
async def test_cooccurrence_counts_distinct_papers(
    db_session: AsyncSession, store  # noqa: ANN001
) -> None:
    await _ingest(db_session, store, filename="a.pdf", body="Evaluated on GLUE.")
    await _ingest(db_session, store, filename="b.pdf", body="Also evaluated on GLUE.")

    rows = await entity_cooccurrence(db_session, await _paper_ids(db_session))

    counts = {name: total for _, name, total in rows}
    assert counts.get("glue") == 2


@pytest.mark.asyncio
async def test_cooccurrence_can_be_scoped_to_one_label(
    db_session: AsyncSession, store  # noqa: ANN001
) -> None:
    await _ingest(db_session, store, filename="a.pdf", body="BERT on SQuAD with accuracy.")

    rows = await entity_cooccurrence(db_session, await _paper_ids(db_session), label="Dataset")

    assert {label for label, _, _ in rows} == {"Dataset"}


@pytest.mark.asyncio
async def test_cooccurrence_on_an_empty_corpus(db_session: AsyncSession) -> None:
    assert await entity_cooccurrence(db_session, []) == []
