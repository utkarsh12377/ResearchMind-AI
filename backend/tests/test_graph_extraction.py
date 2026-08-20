import json

import pytest

from app.graph.extraction import (
    extract_entities,
    extract_with_gazetteer,
    parse_extraction_payload,
)
from app.graph.schema import NodeLabel, RelationType
from app.llm.gateway import LLMGateway
from tests.test_rag import ScriptedProvider


def _labels(result) -> set[str]:  # noqa: ANN001
    return {entity.label for entity in result.entities}


def _names(result) -> set[str]:  # noqa: ANN001
    return {entity.name for entity in result.entities}


def test_gazetteer_finds_known_datasets_and_models() -> None:
    text = "We fine-tune BERT on SQuAD and report accuracy on the dev split."

    result = extract_with_gazetteer(text)

    assert "squad" in _names(result)
    assert "bert" in _names(result)
    assert NodeLabel.METRIC.value in _labels(result)


def test_gazetteer_requires_a_whole_word_match() -> None:
    """'arc' inside 'architecture' is not the ARC benchmark."""
    result = extract_with_gazetteer("We describe the architecture in detail.")

    assert "arc" not in _names(result)


def test_gazetteer_links_entities_to_the_paper() -> None:
    result = extract_with_gazetteer("Trained on ImageNet.", paper_title="A Vision Paper")

    relations = {relation.type for relation in result.relations}
    assert RelationType.USES_DATASET.value in relations


def test_gazetteer_emits_no_relations_without_a_paper_title() -> None:
    result = extract_with_gazetteer("Trained on ImageNet.")

    assert result.relations == []


def test_gazetteer_on_empty_text_returns_nothing() -> None:
    result = extract_with_gazetteer("   ")

    assert result.entities == []


def test_parser_drops_unknown_entity_types() -> None:
    payload = json.dumps(
        {
            "entities": [
                {"type": "Dataset", "name": "WikiSQL"},
                {"type": "Sandwich", "name": "BLT"},
            ]
        }
    )

    result = parse_extraction_payload(payload)

    assert _names(result) == {"WikiSQL"}


def test_parser_drops_relations_with_unknown_endpoints() -> None:
    """A relation naming an entity that was never extracted is unusable."""
    payload = json.dumps(
        {
            "entities": [{"type": "Dataset", "name": "WikiSQL"}],
            "relations": [
                {"type": "USES_DATASET", "source": "My Paper", "target": "WikiSQL"},
                {"type": "USES_DATASET", "source": "My Paper", "target": "Imaginary Set"},
            ],
        }
    )

    result = parse_extraction_payload(payload, paper_title="My Paper")

    assert len(result.relations) == 1
    assert result.relations[0].target == "WikiSQL"


def test_parser_drops_unknown_relation_types() -> None:
    payload = json.dumps(
        {
            "entities": [{"type": "Dataset", "name": "GLUE"}],
            "relations": [{"type": "EATS", "source": "P", "target": "GLUE"}],
        }
    )

    result = parse_extraction_payload(payload, paper_title="P")

    assert result.relations == []


def test_parser_reads_json_wrapped_in_a_code_fence() -> None:
    raw = '```json\n{"entities": [{"type": "Model", "name": "T5"}]}\n```'

    assert _names(parse_extraction_payload(raw)) == {"T5"}


def test_parser_recovers_json_buried_in_prose() -> None:
    raw = 'Sure! Here is the extraction:\n{"entities": [{"type": "Model", "name": "T5"}]}\nDone.'

    assert _names(parse_extraction_payload(raw)) == {"T5"}


def test_parser_returns_empty_on_unparseable_output() -> None:
    result = parse_extraction_payload("I could not complete that request.")

    assert result.entities == []
    assert result.relations == []


def test_parser_clamps_out_of_range_confidence() -> None:
    payload = json.dumps({"entities": [{"type": "Model", "name": "T5", "confidence": 9.5}]})

    assert parse_extraction_payload(payload).entities[0].confidence == 1.0


def test_merge_does_not_duplicate_the_same_entity() -> None:
    left = parse_extraction_payload(json.dumps({"entities": [{"type": "Model", "name": "T5"}]}))
    right = parse_extraction_payload(json.dumps({"entities": [{"type": "Model", "name": "t5"}]}))

    merged = left.merge(right)

    assert len(merged.entities) == 1


@pytest.mark.asyncio
async def test_extraction_combines_both_passes() -> None:
    payload = json.dumps(
        {"entities": [{"type": "Method", "name": "rotary position embedding"}]}
    )
    gateway = LLMGateway(ScriptedProvider([payload]))

    result = await extract_entities(
        "We fine-tune BERT on SQuAD.", paper_title="P", gateway=gateway
    )

    assert "bert" in _names(result)
    assert "rotary position embedding" in _names(result)


@pytest.mark.asyncio
async def test_extraction_survives_an_llm_failure() -> None:
    """The gazetteer result must still come back if the model call blows up."""

    class Broken(ScriptedProvider):
        async def complete(self, messages, **kwargs):  # noqa: ANN001, ANN003, ANN201
            raise RuntimeError("provider is down")

    result = await extract_entities(
        "Evaluated on GLUE.", paper_title="P", gateway=LLMGateway(Broken([]))
    )

    assert "glue" in _names(result)


@pytest.mark.asyncio
async def test_extraction_without_a_gateway_uses_rules_only() -> None:
    result = await extract_entities("Trained on ImageNet.", paper_title="P", use_llm=False)

    assert "imagenet" in _names(result)
