import json

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.llm.gateway import LLMGateway
from app.models import ExperimentResult
from app.research.extraction import (
    canonical_metric,
    extract_experiments,
    extract_methodology,
    extract_results_with_rules,
    nearby_dataset,
    parse_methodology_payload,
    parse_results_payload,
)
from app.worker.tasks import _process_paper
from tests.factories import build_pdf
from tests.test_ingestion_task import _store_paper
from tests.test_rag import ScriptedProvider


def test_metric_aliases_collapse_to_one_name() -> None:
    assert canonical_metric("F1 Score") == "f1"
    assert canonical_metric("f-1") == "f1"
    assert canonical_metric("EM") == "exact match"
    assert canonical_metric("AUROC") == "roc-auc"


def test_an_unknown_metric_keeps_its_normalized_form() -> None:
    assert canonical_metric("Brier Score") == "brier score"


def test_rules_find_a_metric_next_to_a_number() -> None:
    results = extract_results_with_rules("Our system reaches an F1 of 91.2 on the test set.")

    assert results
    assert results[0].metric == "f1"
    assert results[0].value == 91.2


def test_rules_capture_a_percent_unit() -> None:
    results = extract_results_with_rules("We report accuracy of 88.4%.")

    assert results[0].unit == "%"


def test_rules_ignore_prose_without_numbers() -> None:
    assert extract_results_with_rules("We evaluate accuracy carefully throughout.") == []


def test_rules_attribute_to_the_papers_own_model() -> None:
    results = extract_results_with_rules("BLEU of 34.1.", default_model="Our Paper")

    assert results[0].model == "Our Paper"


def test_rules_deduplicate_the_same_metric_and_value() -> None:
    text = "Accuracy of 90.0 in Table 1. We repeat that accuracy of 90.0 in the text."

    results = extract_results_with_rules(text)

    assert len([r for r in results if r.metric == "accuracy"]) == 1


def test_result_payload_requires_a_numeric_value() -> None:
    payload = json.dumps(
        {
            "results": [
                {"model": "T5", "metric": "F1", "value": "not a number"},
                {"model": "T5", "metric": "F1", "value": 88.1},
            ]
        }
    )

    results = parse_results_payload(payload)

    assert len(results) == 1
    assert results[0].value == 88.1


def test_result_payload_requires_a_model_and_metric() -> None:
    payload = json.dumps({"results": [{"model": "", "metric": "F1", "value": 1.0}]})

    assert parse_results_payload(payload) == []


def test_result_payload_normalizes_the_metric_name() -> None:
    payload = json.dumps({"results": [{"model": "T5", "metric": "F1 Score", "value": 88.1}]})

    assert parse_results_payload(payload)[0].metric == "f1"


def test_result_payload_on_junk_returns_nothing() -> None:
    assert parse_results_payload("I was unable to find results.") == []


def test_methodology_payload_is_read_into_fields() -> None:
    payload = json.dumps(
        {
            "approach": "Contrastive pretraining",
            "architecture": "Dual encoder",
            "training": "In-batch negatives",
            "hyperparameters": [{"name": "lr", "value": "1e-5"}],
            "limitations": ["English only"],
        }
    )

    methodology = parse_methodology_payload(payload)

    assert methodology.architecture == "Dual encoder"
    assert methodology.hyperparameters[0].name == "lr"
    assert methodology.limitations == ["English only"]
    assert not methodology.is_empty


def test_methodology_payload_on_junk_is_empty() -> None:
    assert parse_methodology_payload("no idea").is_empty


@pytest.mark.asyncio
async def test_extraction_persists_rows(db_session: AsyncSession) -> None:
    paper = await _store_paper(
        db_session, build_pdf(body="Our model reaches an F1 of 91.2 on the benchmark."), "r.pdf"
    )
    await _process_paper(db_session, paper.id)

    await extract_experiments(db_session, paper.id, use_llm=False)

    rows = (
        await db_session.scalars(
            select(ExperimentResult).where(ExperimentResult.paper_id == paper.id)
        )
    ).all()
    assert any(row.metric_key == "f1" for row in rows)


@pytest.mark.asyncio
async def test_re_extraction_replaces_previous_rows(db_session: AsyncSession) -> None:
    paper = await _store_paper(db_session, build_pdf(body="Accuracy of 90.0."), "r.pdf")
    await _process_paper(db_session, paper.id)

    await extract_experiments(db_session, paper.id, use_llm=False)
    await extract_experiments(db_session, paper.id, use_llm=False)

    rows = (
        await db_session.scalars(
            select(ExperimentResult).where(ExperimentResult.paper_id == paper.id)
        )
    ).all()
    assert len(rows) == len({(r.model_key, r.metric_key, r.value) for r in rows})


@pytest.mark.asyncio
async def test_llm_rows_take_precedence_over_rule_rows(db_session: AsyncSession) -> None:
    """The model knows attribution and dataset; the regex is only a safety net."""
    paper = await _store_paper(db_session, build_pdf(body="Accuracy of 90.0."), "r.pdf")
    await _process_paper(db_session, paper.id)

    payload = json.dumps(
        {
            "results": [
                {"model": "RoBERTa", "dataset": "SQuAD", "metric": "accuracy", "value": 90.0}
            ]
        }
    )
    gateway = LLMGateway(ScriptedProvider([payload]))

    results = await extract_experiments(db_session, paper.id, gateway=gateway)

    accuracy = [r for r in results if r.metric == "accuracy"]
    assert accuracy[0].model == "RoBERTa"
    assert accuracy[0].dataset == "SQuAD"


@pytest.mark.asyncio
async def test_extraction_falls_back_to_rules_when_the_model_fails(
    db_session: AsyncSession,
) -> None:
    paper = await _store_paper(db_session, build_pdf(body="Accuracy of 90.0."), "r.pdf")
    await _process_paper(db_session, paper.id)

    class Broken(ScriptedProvider):
        async def complete(self, messages, **kwargs):  # noqa: ANN001, ANN003, ANN201
            raise RuntimeError("down")

    results = await extract_experiments(db_session, paper.id, gateway=LLMGateway(Broken([])))

    assert any(r.metric == "accuracy" for r in results)


@pytest.mark.asyncio
async def test_extracting_an_unknown_paper_raises(db_session: AsyncSession) -> None:
    import uuid

    with pytest.raises(ValueError):
        await extract_experiments(db_session, uuid.uuid4(), use_llm=False)


@pytest.mark.asyncio
async def test_methodology_extraction_reads_the_model_output(db_session: AsyncSession) -> None:
    paper = await _store_paper(db_session, build_pdf(), "m.pdf")
    await _process_paper(db_session, paper.id)

    payload = json.dumps({"approach": "Dual encoder retrieval", "limitations": ["English only"]})
    gateway = LLMGateway(ScriptedProvider([payload]))

    methodology = await extract_methodology(db_session, paper.id, gateway=gateway)

    assert methodology.approach == "Dual encoder retrieval"
    assert methodology.limitations == ["English only"]


def test_a_dataset_named_beside_the_metric_is_attributed() -> None:
    results = extract_results_with_rules("Our model reaches an F1 of 88.4 on SQuAD.")

    assert results[0].dataset == "squad"


def test_the_nearest_dataset_wins() -> None:
    text = "Trained on ImageNet. Much later in the paper, an F1 of 71.0 on SQuAD."

    assert nearby_dataset(text, text.index("F1")) == "squad"


def test_a_distant_dataset_is_not_attributed() -> None:
    """Attributing to whichever benchmark appeared first is how this goes wrong."""
    text = "We use SQuAD." + (" filler" * 120) + " Accuracy of 90.0."

    assert nearby_dataset(text, text.index("Accuracy")) is None


def test_no_dataset_nearby_leaves_it_unattributed() -> None:
    results = extract_results_with_rules("The reported accuracy of 90.0 stands alone.")

    assert results[0].dataset is None


def test_attribution_raises_confidence_slightly() -> None:
    with_dataset = extract_results_with_rules("An F1 of 88.4 on SQuAD.")[0]
    without = extract_results_with_rules("An F1 of 88.4 in our setting.")[0]

    assert with_dataset.confidence > without.confidence
