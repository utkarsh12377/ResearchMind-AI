import json

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.llm.gateway import LLMGateway
from app.models import ExperimentResult, Paper
from app.research.contradictions import (
    analyze_consistency,
    compare_claims,
    extract_claims,
    find_numeric_conflicts,
)
from app.worker.tasks import _process_paper
from tests.factories import build_pdf
from tests.test_ingestion_task import _store_paper
from tests.test_rag import ScriptedProvider


async def _paper(db: AsyncSession, filename: str, *, title: str, body: str = "Body.") -> Paper:
    paper = await _store_paper(db, build_pdf(title=title, body=body), filename)
    await _process_paper(db, paper.id)
    paper.title = title
    await db.commit()
    return paper


def _result(paper_id, value: float, **overrides):  # noqa: ANN001, ANN003, ANN202
    defaults = {
        "paper_id": paper_id,
        "model_name": "BERT",
        "model_key": "bert",
        "dataset_name": "SQuAD",
        "dataset_key": "squad",
        "metric_name": "f1",
        "metric_key": "f1",
        "value": value,
        "confidence": 0.9,
    }
    return ExperimentResult(**{**defaults, **overrides})


def test_claims_are_sentences_that_assert_a_finding() -> None:
    text = (
        "We used a batch size of 32 throughout all of our reported experiments. "
        "We find that contrastive pretraining outperforms the masked baseline by a wide margin. "
    )

    claims = extract_claims(text)

    assert len(claims) == 1
    assert "outperforms" in claims[0]


def test_short_fragments_are_not_claims() -> None:
    assert extract_claims("We find x. Yes. No.") == []


def test_claim_extraction_respects_the_limit() -> None:
    sentence = "We find that this particular configuration improves results considerably here. "

    assert len(extract_claims(sentence * 10, limit=3)) == 3


@pytest.mark.asyncio
async def test_a_single_paper_has_no_conflicts(db_session: AsyncSession) -> None:
    paper = await _paper(db_session, "a.pdf", title="Solo")
    db_session.add(_result(paper.id, 88.0))
    await db_session.commit()

    assert await find_numeric_conflicts(db_session, [paper.id]) == []


@pytest.mark.asyncio
async def test_matching_numbers_are_not_a_conflict(db_session: AsyncSession) -> None:
    first = await _paper(db_session, "a.pdf", title="A")
    second = await _paper(db_session, "b.pdf", title="B")
    db_session.add(_result(first.id, 88.0))
    db_session.add(_result(second.id, 88.2))
    await db_session.commit()

    assert await find_numeric_conflicts(db_session, [first.id, second.id]) == []


@pytest.mark.asyncio
async def test_a_material_gap_is_a_conflict(db_session: AsyncSession) -> None:
    first = await _paper(db_session, "a.pdf", title="A")
    second = await _paper(db_session, "b.pdf", title="B")
    db_session.add(_result(first.id, 88.0))
    db_session.add(_result(second.id, 71.0))
    await db_session.commit()

    conflicts = await find_numeric_conflicts(db_session, [first.id, second.id])

    assert len(conflicts) == 1
    assert conflicts[0].relative_gap > 0.05
    assert "SQuAD" in conflicts[0].describe()


@pytest.mark.asyncio
async def test_different_datasets_are_never_a_conflict(db_session: AsyncSession) -> None:
    """Different results on different benchmarks are just different results."""
    first = await _paper(db_session, "a.pdf", title="A")
    second = await _paper(db_session, "b.pdf", title="B")
    db_session.add(_result(first.id, 88.0))
    db_session.add(_result(second.id, 40.0, dataset_name="GLUE", dataset_key="glue"))
    await db_session.commit()

    assert await find_numeric_conflicts(db_session, [first.id, second.id]) == []


@pytest.mark.asyncio
async def test_different_models_are_never_a_conflict(db_session: AsyncSession) -> None:
    first = await _paper(db_session, "a.pdf", title="A")
    second = await _paper(db_session, "b.pdf", title="B")
    db_session.add(_result(first.id, 88.0))
    db_session.add(_result(second.id, 40.0, model_name="T5", model_key="t5"))
    await db_session.commit()

    assert await find_numeric_conflicts(db_session, [first.id, second.id]) == []


@pytest.mark.asyncio
async def test_conflicts_are_ordered_by_severity(db_session: AsyncSession) -> None:
    first = await _paper(db_session, "a.pdf", title="A")
    second = await _paper(db_session, "b.pdf", title="B")
    third = await _paper(db_session, "c.pdf", title="C")
    db_session.add(_result(first.id, 90.0))
    db_session.add(_result(second.id, 80.0))
    db_session.add(_result(third.id, 30.0))
    await db_session.commit()

    conflicts = await find_numeric_conflicts(db_session, [first.id, second.id, third.id])

    gaps = [c.relative_gap for c in conflicts]
    assert gaps == sorted(gaps, reverse=True)


@pytest.mark.asyncio
async def test_claims_from_one_paper_are_never_compared(db_session: AsyncSession) -> None:
    paper = await _paper(
        db_session, "a.pdf", title="Solo", body="We find that retrieval improves accuracy greatly."
    )

    contradictions, agreements = await compare_claims(
        db_session, [paper.id], gateway=LLMGateway(ScriptedProvider([]))
    )

    assert contradictions == []
    assert agreements == []


@pytest.mark.asyncio
async def test_a_contradiction_judgement_is_recorded(db_session: AsyncSession) -> None:
    body = "We find that contrastive pretraining significantly improves retrieval accuracy here."
    first = await _paper(db_session, "a.pdf", title="A", body=body)
    second = await _paper(db_session, "b.pdf", title="B", body=body)

    judgement = json.dumps(
        {"relation": "contradiction", "confidence": 0.8, "reason": "opposite conclusions"}
    )
    gateway = LLMGateway(ScriptedProvider([judgement] * 10))

    contradictions, _ = await compare_claims(db_session, [first.id, second.id], gateway=gateway)

    assert contradictions
    assert contradictions[0].relation == "contradiction"
    assert contradictions[0].confidence == 0.8


@pytest.mark.asyncio
async def test_an_unrelated_judgement_is_discarded(db_session: AsyncSession) -> None:
    body = "We find that contrastive pretraining significantly improves retrieval accuracy here."
    first = await _paper(db_session, "a.pdf", title="A", body=body)
    second = await _paper(db_session, "b.pdf", title="B", body=body)

    judgement = json.dumps({"relation": "unrelated", "confidence": 0.9, "reason": "different"})
    gateway = LLMGateway(ScriptedProvider([judgement] * 10))

    contradictions, agreements = await compare_claims(
        db_session, [first.id, second.id], gateway=gateway
    )

    assert contradictions == []
    assert agreements == []


@pytest.mark.asyncio
async def test_the_report_summarizes_a_clean_corpus(db_session: AsyncSession) -> None:
    first = await _paper(db_session, "a.pdf", title="A")
    second = await _paper(db_session, "b.pdf", title="B")

    report = await analyze_consistency(db_session, [first.id, second.id], include_claims=False)

    assert report.is_consistent
    assert "No conflicts" in report.summary()


@pytest.mark.asyncio
async def test_the_report_needs_two_papers_to_say_anything(db_session: AsyncSession) -> None:
    paper = await _paper(db_session, "a.pdf", title="A")

    report = await analyze_consistency(db_session, [paper.id], include_claims=False)

    assert "At least two papers" in report.summary()


@pytest.mark.asyncio
async def test_the_report_counts_numeric_conflicts(db_session: AsyncSession) -> None:
    first = await _paper(db_session, "a.pdf", title="A")
    second = await _paper(db_session, "b.pdf", title="B")
    db_session.add(_result(first.id, 90.0))
    db_session.add(_result(second.id, 40.0))
    await db_session.commit()

    report = await analyze_consistency(db_session, [first.id, second.id], include_claims=False)

    assert not report.is_consistent
    assert "1 numeric conflict" in report.summary()
