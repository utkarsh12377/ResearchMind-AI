import pytest

from app.graph.schema import GraphNode, GraphRelationship, NodeLabel, RelationType
from app.graph.store import InMemoryGraphStore


def _paper(key: str = "p1") -> GraphNode:
    return GraphNode(NodeLabel.PAPER.value, key, {"name": f"Paper {key}"})


def _dataset(key: str = "squad") -> GraphNode:
    return GraphNode(NodeLabel.DATASET.value, key, {"name": key})


@pytest.fixture
def store() -> InMemoryGraphStore:
    return InMemoryGraphStore()


@pytest.mark.asyncio
async def test_counts_are_grouped_by_label(store: InMemoryGraphStore) -> None:
    await store.upsert_nodes([_paper(), _dataset(), _dataset("glue")])

    counts = await store.counts()

    assert counts["Paper"] == 1
    assert counts["Dataset"] == 2


@pytest.mark.asyncio
async def test_upserting_the_same_node_merges_properties(store: InMemoryGraphStore) -> None:
    await store.upsert_nodes([GraphNode(NodeLabel.PAPER.value, "p1", {"year": 2020})])
    await store.upsert_nodes([GraphNode(NodeLabel.PAPER.value, "p1", {"venue": "NeurIPS"})])

    counts = await store.counts()
    hits = await store.find_nodes(NodeLabel.PAPER.value)

    assert counts["Paper"] == 1
    assert hits[0].properties == {"year": 2020, "venue": "NeurIPS"}


@pytest.mark.asyncio
async def test_relationships_pull_in_their_endpoints(store: InMemoryGraphStore) -> None:
    edge = GraphRelationship(RelationType.USES_DATASET.value, _paper(), _dataset())

    await store.upsert_relationships([edge])

    counts = await store.counts()
    assert counts["_relationships"] == 1
    assert counts["Paper"] == 1
    assert counts["Dataset"] == 1


@pytest.mark.asyncio
async def test_readding_the_same_edge_does_not_duplicate_it(store: InMemoryGraphStore) -> None:
    edge = GraphRelationship(RelationType.USES_DATASET.value, _paper(), _dataset())

    await store.upsert_relationships([edge, edge])

    assert (await store.counts())["_relationships"] == 1


@pytest.mark.asyncio
async def test_neighbors_traverse_in_both_directions(store: InMemoryGraphStore) -> None:
    await store.upsert_relationships(
        [GraphRelationship(RelationType.USES_DATASET.value, _paper(), _dataset())]
    )

    from_paper = await store.neighbors(NodeLabel.PAPER.value, "p1")
    from_dataset = await store.neighbors(NodeLabel.DATASET.value, "squad")

    assert [hit.node.key for hit in from_paper] == ["squad"]
    assert [hit.node.key for hit in from_dataset] == ["p1"]


@pytest.mark.asyncio
async def test_neighbors_reach_further_at_greater_depth(store: InMemoryGraphStore) -> None:
    """Two papers sharing a dataset are two hops apart, not one."""
    await store.upsert_relationships(
        [
            GraphRelationship(RelationType.USES_DATASET.value, _paper("p1"), _dataset()),
            GraphRelationship(RelationType.USES_DATASET.value, _paper("p2"), _dataset()),
        ]
    )

    shallow = await store.neighbors(NodeLabel.PAPER.value, "p1", depth=1)
    deep = await store.neighbors(NodeLabel.PAPER.value, "p1", depth=2)

    assert {hit.node.key for hit in shallow} == {"squad"}
    assert "p2" in {hit.node.key for hit in deep}


@pytest.mark.asyncio
async def test_neighbors_can_be_restricted_by_relationship(store: InMemoryGraphStore) -> None:
    author = GraphNode(NodeLabel.AUTHOR.value, "ada lovelace")
    await store.upsert_relationships(
        [
            GraphRelationship(RelationType.USES_DATASET.value, _paper(), _dataset()),
            GraphRelationship(RelationType.AUTHORED_BY.value, _paper(), author),
        ]
    )

    hits = await store.neighbors(
        NodeLabel.PAPER.value, "p1", relationship_types=[RelationType.AUTHORED_BY.value]
    )

    assert [hit.node.key for hit in hits] == ["ada lovelace"]


@pytest.mark.asyncio
async def test_neighbors_of_an_unknown_node_is_empty(store: InMemoryGraphStore) -> None:
    assert await store.neighbors(NodeLabel.PAPER.value, "nope") == []


@pytest.mark.asyncio
async def test_find_nodes_filters_by_substring(store: InMemoryGraphStore) -> None:
    await store.upsert_nodes([_dataset("squad"), _dataset("squad 2.0"), _dataset("glue")])

    hits = await store.find_nodes(NodeLabel.DATASET.value, contains="squad")

    assert {hit.key for hit in hits} == {"squad", "squad 2.0"}


@pytest.mark.asyncio
async def test_deleting_a_paper_detaches_its_edges(store: InMemoryGraphStore) -> None:
    await store.upsert_relationships(
        [GraphRelationship(RelationType.USES_DATASET.value, _paper(), _dataset())]
    )

    removed = await store.delete_paper("p1")
    counts = await store.counts()

    assert removed == 2
    assert "Paper" not in counts
    assert counts["_relationships"] == 0
    assert counts["Dataset"] == 1


@pytest.mark.asyncio
async def test_deleting_an_unknown_paper_is_a_no_op(store: InMemoryGraphStore) -> None:
    await store.upsert_nodes([_paper()])

    assert await store.delete_paper("missing") == 0
    assert (await store.counts())["Paper"] == 1


@pytest.mark.asyncio
async def test_the_graph_survives_a_restart_when_persisted(tmp_path) -> None:  # noqa: ANN001
    path = tmp_path / "graph" / "graph.json"
    store = InMemoryGraphStore(persist_path=path)
    await store.upsert_relationships(
        [GraphRelationship(RelationType.USES_DATASET.value, _paper(), _dataset())]
    )

    reopened = InMemoryGraphStore(persist_path=path)
    counts = await reopened.counts()

    assert counts["Paper"] == 1
    assert counts["_relationships"] == 1


@pytest.mark.asyncio
async def test_a_corrupt_graph_file_does_not_block_startup(tmp_path) -> None:  # noqa: ANN001
    path = tmp_path / "graph.json"
    path.write_text("{not json", encoding="utf-8")

    store = InMemoryGraphStore(persist_path=path)

    assert await store.counts() == {"_relationships": 0}


@pytest.mark.asyncio
async def test_cypher_is_refused_without_neo4j(store: InMemoryGraphStore) -> None:
    with pytest.raises(NotImplementedError):
        await store.run_read("MATCH (n) RETURN n")
