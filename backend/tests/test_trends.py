import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.graph.builder import build_paper_graph
from app.graph.store import InMemoryGraphStore
from app.llm.gateway import LLMGateway
from app.models import ExperimentResult, Paper
from app.research.trends import (
    EntityTrend,
    analyze_trends,
    build_timeline,
    entity_trends,
    metric_progressions,
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
    year: int | None = None,
    body: str = "Body.",
    title: str = "A Paper",
    graph_store=None,  # noqa: ANN001
) -> Paper:
    paper = await _store_paper(db, build_pdf(title=title, body=body), filename)
    await _process_paper(db, paper.id)
    paper.title = title
    paper.published_year = year
    await db.commit()
    if graph_store is not None:
        await build_paper_graph(db, paper.id, store=graph_store, use_llm=False)
    return paper


def test_a_single_year_is_new_not_rising() -> None:
    """One point is not a direction."""
    trend = EntityTrend(label="Dataset", name="glue", by_year={2023: 5}, total=5)

    assert trend.direction == "new"


def test_growth_against_a_quiet_past_is_rising() -> None:
    trend = EntityTrend(
        label="Model", name="llama", by_year={2020: 1, 2021: 1, 2023: 4, 2024: 6}, total=12
    )

    assert trend.direction == "rising"


def test_a_collapse_in_recent_years_is_declining() -> None:
    trend = EntityTrend(
        label="Model", name="lstm", by_year={2016: 10, 2017: 10, 2023: 1, 2024: 1}, total=22
    )

    assert trend.direction == "declining"


def test_no_earlier_mentions_at_all_is_emerging() -> None:
    trend = EntityTrend(label="Method", name="lora", by_year={2023: 3, 2024: 5}, total=8)

    assert trend.direction == "emerging"


def test_flat_activity_is_steady() -> None:
    trend = EntityTrend(
        label="Dataset", name="squad", by_year={2020: 4, 2021: 4, 2022: 4, 2023: 4}, total=16
    )

    assert trend.direction == "steady"


@pytest.mark.asyncio
async def test_an_empty_corpus_gives_an_empty_timeline(db_session: AsyncSession) -> None:
    timeline = await build_timeline(db_session, [])

    assert timeline.buckets == []
    assert timeline.span is None


@pytest.mark.asyncio
async def test_the_timeline_buckets_by_year(db_session: AsyncSession) -> None:
    a = await _paper(db_session, "a.pdf", year=2019)
    b = await _paper(db_session, "b.pdf", year=2023)
    c = await _paper(db_session, "c.pdf", year=2023)

    timeline = await build_timeline(db_session, [a.id, b.id, c.id])

    assert [bucket.year for bucket in timeline.buckets] == [2019, 2023]
    assert timeline.buckets[1].paper_count == 2
    assert timeline.span == (2019, 2023)


@pytest.mark.asyncio
async def test_undated_papers_are_counted_separately(db_session: AsyncSession) -> None:
    dated = await _paper(db_session, "a.pdf", year=2020)
    undated = await _paper(db_session, "b.pdf")

    timeline = await build_timeline(db_session, [dated.id, undated.id])

    assert timeline.undated_papers == 1
    assert timeline.total_papers == 2


@pytest.mark.asyncio
async def test_entity_trends_track_mentions_across_years(
    db_session: AsyncSession, graph_store  # noqa: ANN001
) -> None:
    a = await _paper(
        db_session, "a.pdf", year=2019, body="Evaluated on GLUE.", graph_store=graph_store
    )
    b = await _paper(
        db_session, "b.pdf", year=2023, body="Also on GLUE.", graph_store=graph_store
    )

    trends = await entity_trends(db_session, [a.id, b.id])

    glue = next(t for t in trends if t.name == "glue")
    assert glue.by_year == {2019: 1, 2023: 1}
    assert glue.first_seen == 2019


@pytest.mark.asyncio
async def test_entity_trends_can_be_scoped_to_a_label(
    db_session: AsyncSession, graph_store  # noqa: ANN001
) -> None:
    paper = await _paper(
        db_session,
        "a.pdf",
        year=2023,
        body="BERT on GLUE with accuracy.",
        graph_store=graph_store,
    )

    trends = await entity_trends(db_session, [paper.id], labels=["Dataset"])

    assert {t.label for t in trends} == {"Dataset"}


@pytest.mark.asyncio
async def test_entity_trends_need_dated_papers(
    db_session: AsyncSession, graph_store  # noqa: ANN001
) -> None:
    paper = await _paper(db_session, "a.pdf", body="Evaluated on GLUE.", graph_store=graph_store)

    assert await entity_trends(db_session, [paper.id]) == []


@pytest.mark.asyncio
async def test_progression_takes_the_best_score_per_year(db_session: AsyncSession) -> None:
    old = await _paper(db_session, "a.pdf", year=2019)
    new = await _paper(db_session, "b.pdf", year=2023)
    for paper, value in ((old, 80.0), (old, 82.0), (new, 91.0)):
        db_session.add(
            ExperimentResult(
                paper_id=paper.id,
                model_name="BERT",
                model_key="bert",
                dataset_name="SQuAD",
                dataset_key="squad",
                metric_name="f1",
                metric_key="f1",
                value=value,
                confidence=0.9,
            )
        )
    await db_session.commit()

    progressions = await metric_progressions(db_session, [old.id, new.id])

    assert progressions
    assert progressions[0].points == [(2019, 82.0, "BERT"), (2023, 91.0, "BERT")]
    assert progressions[0].improvement == pytest.approx(9.0)


@pytest.mark.asyncio
async def test_progression_needs_at_least_two_years(db_session: AsyncSession) -> None:
    paper = await _paper(db_session, "a.pdf", year=2023)
    db_session.add(
        ExperimentResult(
            paper_id=paper.id,
            model_name="BERT",
            model_key="bert",
            dataset_name="SQuAD",
            dataset_key="squad",
            metric_name="f1",
            metric_key="f1",
            value=91.0,
            confidence=0.9,
        )
    )
    await db_session.commit()

    assert await metric_progressions(db_session, [paper.id]) == []


@pytest.mark.asyncio
async def test_the_narrative_is_generated_from_measured_evidence(
    db_session: AsyncSession, graph_store  # noqa: ANN001
) -> None:
    a = await _paper(
        db_session, "a.pdf", year=2019, body="Evaluated on GLUE.", graph_store=graph_store
    )
    b = await _paper(
        db_session, "b.pdf", year=2023, body="Also on GLUE.", graph_store=graph_store
    )
    gateway = LLMGateway(ScriptedProvider(["Evaluation consolidated onto GLUE after 2019."]))

    report = await analyze_trends(db_session, [a.id, b.id], topic="NLP", gateway=gateway)

    assert "GLUE" in report.narrative
    assert report.timeline.buckets


@pytest.mark.asyncio
async def test_no_narrative_is_attempted_for_an_empty_corpus(db_session: AsyncSession) -> None:
    gateway = LLMGateway(ScriptedProvider(["should not be used"]))

    report = await analyze_trends(db_session, [], gateway=gateway)

    assert report.narrative == ""
