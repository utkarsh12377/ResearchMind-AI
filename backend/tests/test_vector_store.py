import uuid

import pytest

from app.retrieval.vector_store import (
    InMemoryVectorStore,
    VectorRecord,
    cosine_similarity,
)


def _record(vector: list[float], *, paper_id=None, workspace_id=None, **metadata):  # noqa: ANN001, ANN201
    paper_id = paper_id or uuid.uuid4()
    workspace_id = workspace_id or uuid.uuid4()
    chunk_id = uuid.uuid4()
    return VectorRecord(
        chunk_id=chunk_id,
        paper_id=paper_id,
        workspace_id=workspace_id,
        vector=vector,
        metadata={"paper_id": str(paper_id), "chunk_id": str(chunk_id), **metadata},
    )


@pytest.mark.asyncio
async def test_upsert_and_count() -> None:
    store = InMemoryVectorStore()

    await store.upsert([_record([1.0, 0.0]), _record([0.0, 1.0])])

    assert await store.count() == 2


@pytest.mark.asyncio
async def test_upserting_the_same_chunk_id_replaces_rather_than_duplicates() -> None:
    store = InMemoryVectorStore()
    record = _record([1.0, 0.0])

    await store.upsert([record])
    record.vector = [0.0, 1.0]
    await store.upsert([record])

    assert await store.count() == 1
    [hit] = await store.search([0.0, 1.0], limit=1)
    assert hit.score == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_search_ranks_by_similarity() -> None:
    store = InMemoryVectorStore()
    near = _record([1.0, 0.0])
    far = _record([0.0, 1.0])
    await store.upsert([far, near])

    hits = await store.search([1.0, 0.0], limit=2)

    assert hits[0].chunk_id == near.chunk_id
    assert hits[0].score > hits[1].score


@pytest.mark.asyncio
async def test_search_respects_the_limit() -> None:
    store = InMemoryVectorStore()
    await store.upsert([_record([1.0, 0.0]) for _ in range(5)])

    assert len(await store.search([1.0, 0.0], limit=3)) == 3


@pytest.mark.asyncio
async def test_equality_filter_restricts_results() -> None:
    store = InMemoryVectorStore()
    wanted = _record([1.0, 0.0], kind="table")
    await store.upsert([wanted, _record([1.0, 0.0], kind="text")])

    hits = await store.search([1.0, 0.0], limit=10, filters={"kind": "table"})

    assert [hit.chunk_id for hit in hits] == [wanted.chunk_id]


@pytest.mark.asyncio
async def test_list_filter_matches_any_value() -> None:
    store = InMemoryVectorStore()
    await store.upsert(
        [
            _record([1.0, 0.0], year=2023),
            _record([1.0, 0.0], year=2024),
            _record([1.0, 0.0], year=2019),
        ]
    )

    hits = await store.search([1.0, 0.0], limit=10, filters={"year": [2023, 2024]})

    assert len(hits) == 2


@pytest.mark.asyncio
async def test_filter_on_a_missing_key_matches_nothing() -> None:
    store = InMemoryVectorStore()
    await store.upsert([_record([1.0, 0.0])])

    assert await store.search([1.0, 0.0], filters={"nonexistent": "x"}) == []


@pytest.mark.asyncio
async def test_delete_by_paper_removes_only_that_paper() -> None:
    store = InMemoryVectorStore()
    target = uuid.uuid4()
    await store.upsert(
        [
            _record([1.0, 0.0], paper_id=target),
            _record([1.0, 0.0], paper_id=target),
            _record([0.0, 1.0]),
        ]
    )

    removed = await store.delete_by_paper(target)

    assert removed == 2
    assert await store.count() == 1


@pytest.mark.asyncio
async def test_deleting_an_unknown_paper_is_a_no_op() -> None:
    store = InMemoryVectorStore()
    await store.upsert([_record([1.0, 0.0])])

    assert await store.delete_by_paper(uuid.uuid4()) == 0
    assert await store.count() == 1


@pytest.mark.asyncio
async def test_searching_an_empty_store_returns_nothing() -> None:
    assert await InMemoryVectorStore().search([1.0, 0.0]) == []


@pytest.mark.asyncio
async def test_dimension_mismatch_fails_loudly(tmp_path) -> None:  # noqa: ANN001
    """A query from a different embedding model must not score silently wrong."""
    store = InMemoryVectorStore()
    await store.upsert([_record([1.0, 0.0, 0.0])])

    with pytest.raises(ValueError):
        await store.search([1.0, 0.0])


@pytest.mark.asyncio
async def test_index_survives_a_restart_when_persisted(tmp_path) -> None:  # noqa: ANN001
    path = tmp_path / "vectors" / "index.pkl"
    store = InMemoryVectorStore(persist_path=path)
    await store.upsert([_record([1.0, 0.0])])

    reopened = InMemoryVectorStore(persist_path=path)

    assert await reopened.count() == 1


@pytest.mark.asyncio
async def test_a_corrupt_index_file_does_not_block_startup(tmp_path) -> None:  # noqa: ANN001
    path = tmp_path / "index.pkl"
    path.write_bytes(b"not a pickle")

    store = InMemoryVectorStore(persist_path=path)

    assert await store.count() == 0


def test_cosine_similarity() -> None:
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0
