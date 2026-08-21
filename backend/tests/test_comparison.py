import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import ExperimentResult, Paper
from app.research.comparison import (
    build_comparison_table,
    build_paper_matrix,
    metric_higher_is_better,
)
from app.worker.tasks import _process_paper
from tests.factories import build_pdf
from tests.test_ingestion_task import _store_paper


async def _paper(db: AsyncSession, filename: str, *, title: str, year: int | None = None) -> Paper:
    paper = await _store_paper(db, build_pdf(title=title), filename)
    await _process_paper(db, paper.id)
    paper.title = title
    if year:
        paper.published_year = year
    await db.commit()
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


def test_direction_is_known_for_error_style_metrics() -> None:
    assert metric_higher_is_better("f1")
    assert metric_higher_is_better("accuracy")
    assert not metric_higher_is_better("perplexity")
    assert not metric_higher_is_better("WER")


@pytest.mark.asyncio
async def test_an_empty_corpus_gives_an_empty_table(db_session: AsyncSession) -> None:
    table = await build_comparison_table(db_session, [])

    assert table.is_empty
    assert "No comparable results" in table.to_markdown()


@pytest.mark.asyncio
async def test_results_group_by_dataset_and_metric(db_session: AsyncSession) -> None:
    first = await _paper(db_session, "a.pdf", title="Paper A")
    second = await _paper(db_session, "b.pdf", title="Paper B")
    db_session.add(_result(first.id, value=88.0))
    db_session.add(_result(second.id, value=91.0, model_name="RoBERTa", model_key="roberta"))
    await db_session.commit()

    table = await build_comparison_table(db_session, [first.id, second.id])

    assert len(table.rows) == 1
    assert len(table.rows[0].cells) == 2


@pytest.mark.asyncio
async def test_the_same_metric_on_a_different_dataset_is_a_separate_row(
    db_session: AsyncSession,
) -> None:
    """Two F1 scores on different datasets are not comparable numbers."""
    paper = await _paper(db_session, "a.pdf", title="Paper A")
    db_session.add(_result(paper.id))
    db_session.add(_result(paper.id, dataset_name="GLUE", dataset_key="glue", value=80.0))
    await db_session.commit()

    table = await build_comparison_table(db_session, [paper.id])

    assert {row.dataset for row in table.rows} == {"SQuAD", "GLUE"}


@pytest.mark.asyncio
async def test_the_best_cell_is_marked_for_a_higher_is_better_metric(
    db_session: AsyncSession,
) -> None:
    first = await _paper(db_session, "a.pdf", title="Paper A")
    second = await _paper(db_session, "b.pdf", title="Paper B")
    db_session.add(_result(first.id, value=88.0))
    db_session.add(_result(second.id, value=91.0))
    await db_session.commit()

    table = await build_comparison_table(db_session, [first.id, second.id])

    best = [cell for cell in table.rows[0].cells if cell.is_best]
    assert len(best) == 1
    assert best[0].value == 91.0


@pytest.mark.asyncio
async def test_the_best_cell_is_the_lowest_for_perplexity(db_session: AsyncSession) -> None:
    first = await _paper(db_session, "a.pdf", title="Paper A")
    second = await _paper(db_session, "b.pdf", title="Paper B")
    db_session.add(
        _result(first.id, metric_name="perplexity", metric_key="perplexity", value=18.0)
    )
    db_session.add(
        _result(second.id, metric_name="perplexity", metric_key="perplexity", value=12.0)
    )
    await db_session.commit()

    table = await build_comparison_table(db_session, [first.id, second.id])

    best = [cell for cell in table.rows[0].cells if cell.is_best]
    assert best[0].value == 12.0


@pytest.mark.asyncio
async def test_results_without_a_dataset_are_counted_not_grouped(
    db_session: AsyncSession,
) -> None:
    """An unattributed number cannot go in a comparison cell, but shouldn't vanish."""
    paper = await _paper(db_session, "a.pdf", title="Paper A")
    db_session.add(_result(paper.id, dataset_name=None, dataset_key=None))
    await db_session.commit()

    table = await build_comparison_table(db_session, [paper.id])

    assert table.is_empty
    assert table.ungrouped == 1


@pytest.mark.asyncio
async def test_the_table_can_be_filtered_by_metric(db_session: AsyncSession) -> None:
    paper = await _paper(db_session, "a.pdf", title="Paper A")
    db_session.add(_result(paper.id))
    db_session.add(_result(paper.id, metric_name="accuracy", metric_key="accuracy", value=90.0))
    await db_session.commit()

    table = await build_comparison_table(db_session, [paper.id], metrics=["F1 Score"])

    assert {row.metric for row in table.rows} == {"f1"}


@pytest.mark.asyncio
async def test_rows_can_require_more_than_one_paper(db_session: AsyncSession) -> None:
    paper = await _paper(db_session, "a.pdf", title="Paper A")
    db_session.add(_result(paper.id))
    await db_session.commit()

    table = await build_comparison_table(db_session, [paper.id], min_papers_per_row=2)

    assert table.is_empty


@pytest.mark.asyncio
async def test_markdown_marks_the_winner_in_bold(db_session: AsyncSession) -> None:
    first = await _paper(db_session, "a.pdf", title="Paper A")
    second = await _paper(db_session, "b.pdf", title="Paper B")
    db_session.add(_result(first.id, value=88.0))
    db_session.add(_result(second.id, value=91.0))
    await db_session.commit()

    markdown = (await build_comparison_table(db_session, [first.id, second.id])).to_markdown()

    assert "**91**" in markdown
    assert markdown.startswith("| Dataset |")


@pytest.mark.asyncio
async def test_the_paper_matrix_summarizes_each_paper(db_session: AsyncSession) -> None:
    paper = await _paper(db_session, "a.pdf", title="Paper A", year=2023)
    db_session.add(_result(paper.id))
    await db_session.commit()

    rows = await build_paper_matrix(db_session, [paper.id])

    assert rows[0].title == "Paper A"
    assert rows[0].datasets == ["SQuAD"]
    assert rows[0].models == ["BERT"]


@pytest.mark.asyncio
async def test_the_paper_matrix_sorts_newest_first(db_session: AsyncSession) -> None:
    old = await _paper(db_session, "a.pdf", title="Old", year=2018)
    new = await _paper(db_session, "b.pdf", title="New", year=2024)

    rows = await build_paper_matrix(db_session, [old.id, new.id])

    assert [row.title for row in rows] == ["New", "Old"]


@pytest.mark.asyncio
async def test_the_paper_matrix_is_empty_without_papers(db_session: AsyncSession) -> None:
    assert await build_paper_matrix(db_session, []) == []


@pytest.mark.asyncio
async def test_stored_results_survive_a_reload(db_session: AsyncSession) -> None:
    paper = await _paper(db_session, "a.pdf", title="Paper A")
    db_session.add(_result(paper.id))
    await db_session.commit()

    rows = (await db_session.scalars(select(ExperimentResult))).all()

    assert rows[0].model_key == "bert"
