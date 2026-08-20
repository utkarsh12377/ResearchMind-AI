import pytest

from app.graph.schema import (
    ALLOWED_EDGES,
    GraphNode,
    GraphRelationship,
    NodeLabel,
    RelationType,
    normalize_key,
)


def test_normalize_key_collapses_surface_variants() -> None:
    assert normalize_key("BERT-Base") == normalize_key("bert-base")
    assert normalize_key("  BERT   Base  ") == "bert base"
    assert normalize_key("SQuAD.") == "squad"


def test_normalize_key_strips_wrapping_punctuation() -> None:
    assert normalize_key("(GLUE)") == "glue"
    assert normalize_key('"ImageNet",') == "imagenet"


def test_node_rejects_an_unknown_label() -> None:
    with pytest.raises(ValueError):
        GraphNode("Sandwich", "blt")


def test_node_rejects_an_empty_key() -> None:
    with pytest.raises(ValueError):
        GraphNode(NodeLabel.DATASET.value, "   ")


def test_node_uid_is_label_scoped() -> None:
    dataset = GraphNode(NodeLabel.DATASET.value, "GLUE")
    benchmark = GraphNode(NodeLabel.BENCHMARK.value, "GLUE")

    assert dataset.uid != benchmark.uid


def test_relationship_rejects_an_unknown_type() -> None:
    paper = GraphNode(NodeLabel.PAPER.value, "p1")
    author = GraphNode(NodeLabel.AUTHOR.value, "ada lovelace")

    with pytest.raises(ValueError):
        GraphRelationship("INVENTED_BY", paper, author)


def test_relationship_rejects_the_wrong_endpoint_types() -> None:
    """A hallucinated triple must not be writable just because both nodes exist."""
    author = GraphNode(NodeLabel.AUTHOR.value, "ada lovelace")
    dataset = GraphNode(NodeLabel.DATASET.value, "squad")

    with pytest.raises(ValueError):
        GraphRelationship(RelationType.USES_DATASET.value, author, dataset)


def test_relationship_accepts_the_declared_shape() -> None:
    paper = GraphNode(NodeLabel.PAPER.value, "p1")
    dataset = GraphNode(NodeLabel.DATASET.value, "squad")

    edge = GraphRelationship(RelationType.USES_DATASET.value, paper, dataset)

    assert edge.uid.endswith("Dataset:squad")


def test_every_relation_type_has_a_declared_shape() -> None:
    assert {relation.value for relation in RelationType} == set(ALLOWED_EDGES)
