import json

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.eval.dataset import DEFAULT_CASES, EvalCase, filter_by_tag, load_cases, save_cases
from app.eval.metrics import (
    MetricScores,
    answer_relevancy,
    citation_validity,
    context_precision,
    context_recall,
    faithfulness,
    judge_case,
    score_case,
)
from app.eval.runner import Thresholds, looks_like_refusal, run_benchmark
from app.llm.gateway import LLMGateway
from app.models import User
from app.retrieval.embeddings import HashEmbeddingProvider
from app.retrieval.indexer import index_paper
from app.retrieval.service import rebuild_sparse_index
from app.retrieval.sparse import BM25Index
from app.retrieval.vector_store import InMemoryVectorStore
from app.worker.tasks import _process_paper
from tests.factories import build_pdf
from tests.test_ingestion_task import _store_paper
from tests.test_rag import ScriptedProvider

# --- Metrics ---------------------------------------------------------------


def test_a_quoted_answer_is_fully_faithful() -> None:
    context = "Dense retrieval uses a dual encoder trained with in-batch negatives."

    assert faithfulness("Dense retrieval uses a dual encoder.", [context]) == 1.0


def test_an_invented_answer_is_unfaithful() -> None:
    context = "Dense retrieval uses a dual encoder."

    invented = "The authors received funding from a national laboratory."

    assert faithfulness(invented, [context]) == 0.0


def test_faithfulness_without_context_is_zero() -> None:
    assert faithfulness("Anything at all.", []) == 0.0


def test_faithfulness_of_an_empty_answer_is_zero() -> None:
    assert faithfulness("", ["some context"]) == 0.0


def test_faithfulness_is_the_share_of_grounded_sentences() -> None:
    context = "Dense retrieval uses a dual encoder."
    answer = "Dense retrieval uses a dual encoder. Funding came from elsewhere entirely."

    assert faithfulness(answer, [context]) == 0.5


def test_relevancy_rewards_engaging_with_the_question() -> None:
    score = answer_relevancy(
        "What encoder does dense retrieval use?", "Dense retrieval uses a dual encoder."
    )

    assert score > 0.5


def test_relevancy_of_an_off_topic_answer_is_low() -> None:
    score = answer_relevancy("What encoder does dense retrieval use?", "Bananas are yellow.")

    assert score == 0.0


def test_relevancy_of_an_empty_answer_is_zero() -> None:
    assert answer_relevancy("What encoder?", "") == 0.0


def test_precision_penalizes_irrelevant_passages() -> None:
    """Recall can be met by returning everything; precision cannot."""
    contexts = ["dense retrieval dual encoder architecture", "unrelated cooking instructions"]

    assert context_precision("dense retrieval encoder", contexts) == 0.5


def test_precision_with_no_retrieval_is_zero() -> None:
    assert context_precision("anything", []) == 0.0


def test_recall_measures_coverage_of_the_expected_answer() -> None:
    score = context_recall("dual encoder architecture", ["it uses a dual encoder architecture"])

    assert score == 1.0


def test_recall_without_a_ground_truth_is_full() -> None:
    assert context_recall("", ["anything"]) == 1.0


def test_recall_without_context_is_zero() -> None:
    assert context_recall("dual encoder", []) == 0.0


def test_an_uncited_answer_scores_zero_for_citations() -> None:
    assert citation_validity("A confident claim with no markers.", 3) == 0.0


def test_valid_citations_score_fully() -> None:
    assert citation_validity("A claim [1] and another [2].", 3) == 1.0


def test_out_of_range_citations_are_penalized() -> None:
    assert citation_validity("A claim [1] and a fabricated one [9].", 2) == 0.5


def test_overall_is_the_unweighted_mean() -> None:
    scores = MetricScores(
        faithfulness=1.0,
        answer_relevancy=1.0,
        context_precision=1.0,
        context_recall=1.0,
        citation_validity=0.0,
    )

    assert scores.overall == 0.8


def test_score_case_fills_every_dimension() -> None:
    scores = score_case(
        question="What encoder is used?",
        answer="A dual encoder is used [1].",
        contexts=["The system uses a dual encoder."],
        ground_truth="dual encoder",
    )

    assert scores.as_dict().keys() >= {"faithfulness", "overall"}
    assert scores.citation_validity == 1.0


@pytest.mark.asyncio
async def test_the_judge_overrides_the_lexical_score() -> None:
    payload = json.dumps(
        {"faithfulness": 0.95, "answer_relevancy": 0.9, "context_precision": 0.8, "reason": "good"}
    )
    gateway = LLMGateway(ScriptedProvider([payload]))

    judged = await judge_case(
        question="q", answer="a paraphrase", contexts=["source text"], gateway=gateway
    )

    assert judged.judged
    assert judged.scores.faithfulness == 0.95
    assert judged.lexical_fallback is not None


@pytest.mark.asyncio
async def test_an_unavailable_judge_falls_back_to_lexical() -> None:
    class Broken(ScriptedProvider):
        async def complete(self, messages, **kwargs):  # noqa: ANN001, ANN003, ANN201
            raise RuntimeError("down")

    judged = await judge_case(
        question="q", answer="a", contexts=["c"], gateway=LLMGateway(Broken([]))
    )

    assert not judged.judged
    assert judged.reason == "judge unavailable"


@pytest.mark.asyncio
async def test_unreadable_judge_output_falls_back_to_lexical() -> None:
    gateway = LLMGateway(ScriptedProvider(["I have opinions but no JSON."]))

    judged = await judge_case(question="q", answer="a", contexts=["c"], gateway=gateway)

    assert not judged.judged


# --- Dataset ---------------------------------------------------------------


def test_the_builtin_dataset_covers_refusals() -> None:
    assert any(case.expect_refusal for case in DEFAULT_CASES)


def test_cases_load_from_a_file(tmp_path) -> None:  # noqa: ANN001
    path = tmp_path / "cases.json"
    save_cases([EvalCase(id="a", question="Q?")], path)

    cases = load_cases(path)

    assert [case.id for case in cases] == ["a"]


def test_a_missing_dataset_file_raises(tmp_path) -> None:  # noqa: ANN001
    with pytest.raises(FileNotFoundError):
        load_cases(tmp_path / "nope.json")


def test_duplicate_case_ids_are_rejected(tmp_path) -> None:  # noqa: ANN001
    path = tmp_path / "cases.json"
    path.write_text(
        json.dumps({"cases": [{"id": "a", "question": "Q?"}, {"id": "a", "question": "R?"}]}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Duplicate"):
        load_cases(path)


def test_a_case_without_a_question_is_rejected() -> None:
    with pytest.raises(ValueError):
        EvalCase.from_dict({"id": "a"})


def test_cases_filter_by_tag() -> None:
    assert all("refusal" in c.tags for c in filter_by_tag(list(DEFAULT_CASES), "refusal"))


# --- Runner ----------------------------------------------------------------


def test_refusal_phrasings_are_recognized() -> None:
    assert looks_like_refusal("The sources do not contain that information.")
    assert looks_like_refusal("I cannot answer from the provided sources.")
    assert not looks_like_refusal("The system uses a dual encoder.")


@pytest.fixture
def provider() -> HashEmbeddingProvider:
    return HashEmbeddingProvider(dimensions=256)


@pytest.fixture
def store() -> InMemoryVectorStore:
    return InMemoryVectorStore()


@pytest.fixture
def bm25() -> BM25Index:
    return BM25Index()


async def _corpus(db: AsyncSession, provider, store, bm25):  # noqa: ANN001, ANN202
    paper = await _store_paper(
        db,
        build_pdf(
            title="Hybrid Retrieval for Science",
            abstract="We present a hybrid retrieval system combining dense and sparse matching.",
            body="The system reaches an accuracy of 91.2 on the benchmark.",
        ),
        "eval.pdf",
    )
    await _process_paper(db, paper.id)
    await index_paper(db, paper.id, provider=provider, store=store)
    await rebuild_sparse_index(db, bm25=bm25)
    return await db.scalar(select(User).limit(1))


@pytest.mark.asyncio
async def test_an_empty_benchmark_does_not_pass(
    db_session: AsyncSession, provider, store, bm25  # noqa: ANN001
) -> None:
    """A gate that passes when nothing ran is not a gate."""
    user = await _corpus(db_session, provider, store, bm25)

    report = await run_benchmark(
        db_session,
        user,
        cases=[],
        gateway=LLMGateway(ScriptedProvider([])),
        provider=provider,
        store=store,
        bm25=bm25,
    )

    assert not report.passed
    assert report.summary() == "No eval cases ran."


@pytest.mark.asyncio
async def test_a_refusal_case_passes_when_the_system_declines(
    db_session: AsyncSession, provider, store, bm25  # noqa: ANN001
) -> None:
    user = await _corpus(db_session, provider, store, bm25)
    case = EvalCase(id="oc", question="What is the melting point of tungsten?", expect_refusal=True)

    gateway = LLMGateway(
        ScriptedProvider(
            [
                '{"sufficient": false, "reason": "unrelated"}',
                "melting point of tungsten",
                "The sources do not contain that information.",
                '{"supported": true, "confidence": 0.9, "unsupported_claims": [], "reason": "ok"}',
            ]
        )
    )

    report = await run_benchmark(
        db_session,
        user,
        cases=[case],
        gateway=gateway,
        provider=provider,
        store=store,
        bm25=bm25,
    )

    assert report.results[0].refused
    assert report.passed


@pytest.mark.asyncio
async def test_a_case_records_every_reason_it_failed(
    db_session: AsyncSession, provider, store, bm25  # noqa: ANN001
) -> None:
    user = await _corpus(db_session, provider, store, bm25)
    case = EvalCase(id="grounded", question="What retrieval approach is described?")

    gateway = LLMGateway(
        ScriptedProvider(
            [
                '{"sufficient": true, "reason": "ok"}',
                "Completely unrelated content about volcanoes and geology.",
                '{"supported": true, "confidence": 0.9, "unsupported_claims": [], "reason": "ok"}',
            ]
        )
    )
    thresholds = Thresholds(faithfulness=0.9, context_precision=0.9, citation_validity=1.0)

    report = await run_benchmark(
        db_session,
        user,
        cases=[case],
        gateway=gateway,
        thresholds=thresholds,
        provider=provider,
        store=store,
        bm25=bm25,
    )

    assert not report.passed
    assert len(report.results[0].failures) >= 2


@pytest.mark.asyncio
async def test_a_case_that_errors_is_a_failure_not_a_crash(
    db_session: AsyncSession, provider, store, bm25  # noqa: ANN001
) -> None:
    user = await _corpus(db_session, provider, store, bm25)

    class Broken(ScriptedProvider):
        async def complete(self, messages, **kwargs):  # noqa: ANN001, ANN003, ANN201
            raise RuntimeError("provider exploded")

    report = await run_benchmark(
        db_session,
        user,
        cases=[EvalCase(id="boom", question="Anything?")],
        gateway=LLMGateway(Broken([])),
        provider=provider,
        store=store,
        bm25=bm25,
    )

    assert not report.passed
    assert "errored" in report.results[0].failures[0]


@pytest.mark.asyncio
async def test_the_report_serializes_for_ci(
    db_session: AsyncSession, provider, store, bm25  # noqa: ANN001
) -> None:
    user = await _corpus(db_session, provider, store, bm25)
    gateway = LLMGateway(
        ScriptedProvider(
            [
                '{"sufficient": true, "reason": "ok"}',
                "A hybrid retrieval system combining dense and sparse matching [1].",
                '{"supported": true, "confidence": 0.9, "unsupported_claims": [], "reason": "ok"}',
            ]
        )
    )

    report = await run_benchmark(
        db_session,
        user,
        cases=[EvalCase(id="a", question="What retrieval approach is described?")],
        gateway=gateway,
        provider=provider,
        store=store,
        bm25=bm25,
    )
    payload = report.as_dict()

    assert json.dumps(payload)
    assert payload["cases"] == 1
    assert "aggregate" in payload


def test_precision_credits_a_passage_matching_the_expected_answer() -> None:
    """A good source answers in the paper's vocabulary, not the asker's."""
    contexts = ["a hybrid system combining dense and sparse matching"]

    without_truth = context_precision("What approach is described?", contexts)
    with_truth = context_precision(
        "What approach is described?", contexts, "hybrid dense and sparse matching"
    )

    assert without_truth == 0.0
    assert with_truth == 1.0


@pytest.mark.asyncio
async def test_llm_dependent_cases_are_skipped_not_failed(
    db_session: AsyncSession, provider, store, bm25  # noqa: ANN001
) -> None:
    """A gate that is permanently red because of a missing key teaches nothing."""
    from app.llm.gateway import EchoProvider

    user = await _corpus(db_session, provider, store, bm25)
    cases = [
        EvalCase(id="needs-model", question="Anything?", expect_refusal=True, requires_llm=True),
        EvalCase(id="offline-ok", question="What retrieval approach is described?"),
    ]

    report = await run_benchmark(
        db_session,
        user,
        cases=cases,
        gateway=LLMGateway(EchoProvider()),
        thresholds=Thresholds(faithfulness=0.0, context_precision=0.0),
        provider=provider,
        store=store,
        bm25=bm25,
    )

    assert report.skipped_count == 1
    assert report.results[0].skipped
    assert report.passed


@pytest.mark.asyncio
async def test_skipped_cases_do_not_move_the_aggregate(
    db_session: AsyncSession, provider, store, bm25  # noqa: ANN001
) -> None:
    from app.llm.gateway import EchoProvider

    user = await _corpus(db_session, provider, store, bm25)
    cases = [
        EvalCase(id="skipped", question="Anything?", requires_llm=True),
        EvalCase(id="graded", question="What retrieval approach is described?"),
    ]

    report = await run_benchmark(
        db_session,
        user,
        cases=cases,
        gateway=LLMGateway(EchoProvider()),
        thresholds=Thresholds(faithfulness=0.0, context_precision=0.0),
        provider=provider,
        store=store,
        bm25=bm25,
    )

    assert len(report.graded) == 1
    assert report.mean("faithfulness") == report.graded[0].scores.faithfulness
