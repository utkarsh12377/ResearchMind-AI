import json

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.graph.builder import build_paper_graph
from app.graph.store import InMemoryGraphStore
from app.llm.gateway import LLMGateway
from app.models import ExperimentResult, Paper
from app.research.gaps import (
    discover_gaps,
    find_coverage_gaps,
    find_evaluation_gaps,
    find_stated_gaps,
    parse_gaps_payload,
)
from app.worker.tasks import _process_paper
from tests.factories import build_pdf
from tests.test_ingestion_task import _store_paper
from tests.test_rag import ScriptedProvider


@pytest.fixture
def graph_store() -> InMemoryGraphStore:
    return InMemoryGraphStore()


async def _paper(
    db: AsyncSession,
    filename: str,
    *,
    body: str = "Body.",
    title: str = "A Paper",
    graph_store=None,  # noqa: ANN001
) -> Paper:
    paper = await _store_paper(db, build_pdf(title=title, body=body), filename)
    await _process_paper(db, paper.id)
    paper.title = title
    await db.commit()
    if graph_store is not None:
        await build_paper_graph(db, paper.id, store=graph_store, use_llm=False)
    return paper


def _result(paper_id, **overrides):  # noqa: ANN001, ANN003, ANN202
    defaults = {
        "paper_id": paper_id,
        "model_name": "BERT",
        "model_key": "bert",
        "dataset_name": "SQuAD",
        "dataset_key": "squad",
        "metric_name": "f1",
        "metric_key": "f1",
        "value": 88.0,
        "confidence": 0.9,
    }
    return ExperimentResult(**{**defaults, **overrides})


@pytest.mark.asyncio
async def test_coverage_gaps_need_a_few_papers(db_session: AsyncSession) -> None:
    paper = await _paper(db_session, "a.pdf")

    assert await find_coverage_gaps(db_session, [paper.id]) == []


@pytest.mark.asyncio
async def test_a_near_universal_entity_absent_from_one_paper_is_a_gap(
    db_session: AsyncSession, graph_store  # noqa: ANN001
) -> None:
    a = await _paper(db_session, "a.pdf", body="Evaluated on GLUE.", graph_store=graph_store)
    b = await _paper(db_session, "b.pdf", body="Also on GLUE.", graph_store=graph_store)
    c = await _paper(db_session, "c.pdf", body="Nothing notable.", graph_store=graph_store)

    gaps = await find_coverage_gaps(db_session, [a.id, b.id, c.id])

    assert any("glue" in gap.description.lower() for gap in gaps)


@pytest.mark.asyncio
async def test_an_entity_in_every_paper_is_not_a_gap(
    db_session: AsyncSession, graph_store  # noqa: ANN001
) -> None:
    papers = [
        await _paper(db_session, f"{i}.pdf", body="Evaluated on GLUE.", graph_store=graph_store)
        for i in range(3)
    ]

    gaps = await find_coverage_gaps(db_session, [p.id for p in papers])

    assert not any("glue" in gap.description.lower() for gap in gaps)


@pytest.mark.asyncio
async def test_an_untested_model_dataset_pairing_is_a_gap(db_session: AsyncSession) -> None:
    a = await _paper(db_session, "a.pdf")
    b = await _paper(db_session, "b.pdf")
    db_session.add(_result(a.id))
    db_session.add(
        _result(b.id, model_name="T5", model_key="t5", dataset_name="GLUE", dataset_key="glue")
    )
    await db_session.commit()

    gaps = await find_evaluation_gaps(db_session, [a.id, b.id])

    descriptions = " ".join(gap.description for gap in gaps)
    assert "BERT" in descriptions or "T5" in descriptions


@pytest.mark.asyncio
async def test_a_single_dataset_yields_no_pairing_gaps(db_session: AsyncSession) -> None:
    a = await _paper(db_session, "a.pdf")
    b = await _paper(db_session, "b.pdf")
    db_session.add(_result(a.id))
    db_session.add(_result(b.id, model_name="T5", model_key="t5"))
    await db_session.commit()

    assert await find_evaluation_gaps(db_session, [a.id, b.id]) == []


@pytest.mark.asyncio
async def test_no_results_means_no_pairing_gaps(db_session: AsyncSession) -> None:
    a = await _paper(db_session, "a.pdf")
    b = await _paper(db_session, "b.pdf")

    assert await find_evaluation_gaps(db_session, [a.id, b.id]) == []


@pytest.mark.asyncio
async def test_a_future_work_sentence_is_a_stated_gap(db_session: AsyncSession) -> None:
    paper = await _paper(
        db_session,
        "a.pdf",
        body="We leave multilingual evaluation to future work in a follow up study.",
    )

    stated = await find_stated_gaps(db_session, [paper.id])

    assert stated
    assert "future work" in stated[0].text.lower()


@pytest.mark.asyncio
async def test_a_paper_without_future_work_states_nothing(db_session: AsyncSession) -> None:
    paper = await _paper(db_session, "a.pdf", body="We report results and conclude.")

    assert await find_stated_gaps(db_session, [paper.id]) == []


@pytest.mark.asyncio
async def test_stated_gaps_are_capped_per_paper(db_session: AsyncSession) -> None:
    paper = await _paper(
        db_session, "a.pdf", body=("We leave this to future work. " * 20)
    )

    assert len(await find_stated_gaps(db_session, [paper.id], per_paper=1)) <= 1


def test_gap_payload_is_ranked_by_confidence() -> None:
    payload = json.dumps(
        {
            "gaps": [
                {"gap": "low", "confidence": 0.2},
                {"gap": "high", "confidence": 0.9},
            ]
        }
    )

    directions = parse_gaps_payload(payload)

    assert [d.gap for d in directions] == ["high", "low"]


def test_gap_payload_requires_a_gap_statement() -> None:
    payload = json.dumps({"gaps": [{"rationale": "orphan rationale"}]})

    assert parse_gaps_payload(payload) == []


def test_gap_payload_on_junk_returns_nothing() -> None:
    assert parse_gaps_payload("nothing to report") == []


@pytest.mark.asyncio
async def test_an_unreconciled_numeric_conflict_becomes_a_gap(
    db_session: AsyncSession,
) -> None:
    a = await _paper(db_session, "a.pdf", title="A")
    b = await _paper(db_session, "b.pdf", title="B")
    db_session.add(_result(a.id, value=90.0))
    db_session.add(_result(b.id, value=40.0))
    await db_session.commit()

    report = await discover_gaps(db_session, [a.id, b.id])

    assert any(gap.kind == "unresolved_conflict" for gap in report.structural)


@pytest.mark.asyncio
async def test_the_report_summarizes_an_empty_corpus(db_session: AsyncSession) -> None:
    report = await discover_gaps(db_session, [])

    assert not report.has_findings
    assert "No gaps identified" in report.summary()


@pytest.mark.asyncio
async def test_directions_are_synthesized_from_the_evidence(
    db_session: AsyncSession, graph_store  # noqa: ANN001
) -> None:
    a = await _paper(db_session, "a.pdf", body="Evaluated on GLUE.", graph_store=graph_store)
    b = await _paper(db_session, "b.pdf", body="Also on GLUE.", graph_store=graph_store)
    c = await _paper(db_session, "c.pdf", body="Nothing notable.", graph_store=graph_store)

    payload = json.dumps(
        {
            "gaps": [
                {
                    "gap": "No multilingual evaluation",
                    "rationale": "All papers evaluate in English",
                    "suggested_direction": "Extend to XTREME",
                    "confidence": 0.7,
                    "evidence_indices": [1],
                }
            ]
        }
    )
    gateway = LLMGateway(ScriptedProvider([payload]))

    report = await discover_gaps(db_session, [a.id, b.id, c.id], gateway=gateway)

    assert report.directions
    assert report.directions[0].gap == "No multilingual evaluation"


@pytest.mark.asyncio
async def test_a_synthesis_failure_keeps_the_structural_findings(
    db_session: AsyncSession, graph_store  # noqa: ANN001
) -> None:
    a = await _paper(db_session, "a.pdf", body="Evaluated on GLUE.", graph_store=graph_store)
    b = await _paper(db_session, "b.pdf", body="Also on GLUE.", graph_store=graph_store)
    c = await _paper(db_session, "c.pdf", body="Nothing notable.", graph_store=graph_store)

    class Broken(ScriptedProvider):
        async def complete(self, messages, **kwargs):  # noqa: ANN001, ANN003, ANN201
            raise RuntimeError("down")

    report = await discover_gaps(
        db_session, [a.id, b.id, c.id], gateway=LLMGateway(Broken([]))
    )

    assert report.structural
    assert report.directions == []
