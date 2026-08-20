"""Tests for the generated-Cypher validator.

This is the boundary where model output becomes a database query, so the tests
are written as attempts to get something dangerous past it rather than as
happy-path coverage.
"""

import pytest

from app.graph.query import (
    UnsafeCypherError,
    answer_graph_question,
    schema_description,
    validate_cypher,
)
from app.graph.store import InMemoryGraphStore
from app.llm.gateway import LLMGateway
from tests.test_rag import ScriptedProvider

SAFE = "MATCH (p:Paper)-[:USES_DATASET]->(d:Dataset) RETURN p.name LIMIT 10"


def test_a_well_formed_read_query_passes() -> None:
    assert validate_cypher(SAFE) == SAFE


@pytest.mark.parametrize(
    "query",
    [
        "MATCH (p:Paper) DELETE p RETURN 1 LIMIT 1",
        "MATCH (p:Paper) DETACH DELETE p RETURN 1 LIMIT 1",
        "CREATE (p:Paper {key: 'x'}) RETURN p LIMIT 1",
        "MERGE (p:Paper {key: 'x'}) RETURN p LIMIT 1",
        "MATCH (p:Paper) SET p.name = 'x' RETURN p LIMIT 1",
        "MATCH (p:Paper) REMOVE p.name RETURN p LIMIT 1",
        "MATCH (p:Paper) CALL apoc.export.csv.all() RETURN 1 LIMIT 1",
        "LOAD CSV FROM 'http://evil/x.csv' AS row RETURN row LIMIT 1",
        "MATCH (p:Paper) RETURN p LIMIT 1; DROP DATABASE neo4j",
    ],
)
def test_write_and_admin_operations_are_refused(query: str) -> None:
    with pytest.raises(UnsafeCypherError):
        validate_cypher(query)


def test_a_query_without_return_is_refused() -> None:
    with pytest.raises(UnsafeCypherError):
        validate_cypher("MATCH (p:Paper) LIMIT 10")


def test_a_query_that_does_not_start_with_a_read_clause_is_refused() -> None:
    with pytest.raises(UnsafeCypherError):
        validate_cypher("RETURN 1 LIMIT 1")


def test_an_empty_query_is_refused() -> None:
    with pytest.raises(UnsafeCypherError):
        validate_cypher("   ")


def test_an_unknown_label_is_refused() -> None:
    with pytest.raises(UnsafeCypherError, match="label"):
        validate_cypher("MATCH (u:User) RETURN u LIMIT 5")


def test_an_unknown_relationship_is_refused() -> None:
    with pytest.raises(UnsafeCypherError, match="relationship"):
        validate_cypher("MATCH (p:Paper)-[:STOLE_FROM]->(q:Paper) RETURN p LIMIT 5")


def test_a_missing_limit_is_added() -> None:
    result = validate_cypher("MATCH (p:Paper) RETURN p", max_limit=25)

    assert result.endswith("LIMIT 25")


def test_an_oversized_limit_is_clamped() -> None:
    result = validate_cypher("MATCH (p:Paper) RETURN p LIMIT 100000", max_limit=50)

    assert result.endswith("LIMIT 50")
    assert "100000" not in result


def test_a_limit_within_bounds_is_preserved() -> None:
    result = validate_cypher("MATCH (p:Paper) RETURN p LIMIT 5", max_limit=50)

    assert result.endswith("LIMIT 5")


def test_code_fences_are_stripped_before_validation() -> None:
    fenced = f"```cypher\n{SAFE}\n```"

    assert validate_cypher(fenced) == SAFE


def test_a_keyword_inside_a_string_literal_does_not_trip_the_check() -> None:
    """A paper titled 'Deleting Noisy Labels' is not a DELETE statement."""
    query = "MATCH (p:Paper) WHERE p.name = 'Deleting Noisy Labels' RETURN p LIMIT 5"

    assert validate_cypher(query) == query


def test_a_keyword_inside_a_string_cannot_smuggle_an_operation() -> None:
    """Quoted text is blanked, so it can neither trip nor bypass the checks."""
    query = "MATCH (p:Paper) WHERE p.name = 'x' DELETE p RETURN 1 LIMIT 1"

    with pytest.raises(UnsafeCypherError):
        validate_cypher(query)


def test_multi_relationship_alternation_is_validated_per_type() -> None:
    good = "MATCH (p:Paper)-[:CITES|USES_DATASET]->(x) RETURN x LIMIT 5"
    bad = "MATCH (p:Paper)-[:CITES|BRIBES]->(x) RETURN x LIMIT 5"

    assert validate_cypher(good) == good
    with pytest.raises(UnsafeCypherError):
        validate_cypher(bad)


def test_variable_length_paths_are_allowed() -> None:
    query = "MATCH (p:Paper)-[:CITES*1..3]->(q:Paper) RETURN q LIMIT 5"

    assert validate_cypher(query) == query


def test_the_schema_description_lists_only_allowed_vocabulary() -> None:
    schema = schema_description()

    assert "(:Paper" in schema
    assert "[:USES_DATASET]" in schema
    assert "User" not in schema


@pytest.mark.asyncio
async def test_a_rejected_query_is_reported_not_executed() -> None:
    gateway = LLMGateway(ScriptedProvider(["MATCH (p:Paper) DETACH DELETE p RETURN 1 LIMIT 1"]))

    result = await answer_graph_question("wipe everything", gateway=gateway)

    assert not result.succeeded
    assert result.rows == []
    assert result.cypher == ""


@pytest.mark.asyncio
async def test_an_accepted_query_reaches_the_store() -> None:
    gateway = LLMGateway(ScriptedProvider([SAFE]))

    result = await answer_graph_question(
        "which papers use which datasets", gateway=gateway, store=InMemoryGraphStore()
    )

    # The memory backend cannot run Cypher, but the query got that far, which
    # is what distinguishes acceptance from rejection.
    assert result.cypher == SAFE
    assert "Neo4j" in (result.error or "")
