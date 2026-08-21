"""Running the benchmark and deciding whether the result is acceptable.

The gate is what makes this more than a report. CI needs a boolean, and the
thresholds that produce it are configuration rather than constants, because the
number that counts as "good enough" depends on the corpus and moves as the
system improves.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.logging import get_logger
from app.eval.dataset import EvalCase, load_cases
from app.eval.metrics import MetricScores, score_case
from app.llm.gateway import LLMGateway, get_gateway
from app.llm.rag import answer_question
from app.retrieval.service import RetrievalFilters

logger = get_logger(__name__)

REFUSAL_MARKERS = (
    "do not contain", "does not contain", "no information", "not enough information",
    "cannot answer", "can't answer", "not covered", "no relevant", "unable to answer",
    "the sources do not", "insufficient",
)


@dataclass
class CaseResult:
    case_id: str
    question: str
    answer: str
    scores: MetricScores
    source_count: int
    confidence: float
    duration_ms: int
    refused: bool = False
    skipped: bool = False
    passed: bool = True
    failures: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "question": self.question,
            "answer": self.answer[:500],
            "scores": self.scores.as_dict(),
            "source_count": self.source_count,
            "confidence": self.confidence,
            "duration_ms": self.duration_ms,
            "refused": self.refused,
            "skipped": self.skipped,
            "passed": self.passed,
            "failures": self.failures,
        }


@dataclass
class Thresholds:
    faithfulness: float = 0.6
    context_precision: float = 0.3
    citation_validity: float = 0.0

    @classmethod
    def from_settings(cls) -> Thresholds:
        settings = get_settings()
        return cls(
            faithfulness=settings.eval_min_faithfulness,
            context_precision=settings.eval_min_context_precision,
        )


@dataclass
class EvalReport:
    results: list[CaseResult] = field(default_factory=list)
    thresholds: Thresholds = field(default_factory=Thresholds)
    duration_ms: int = 0

    @property
    def case_count(self) -> int:
        return len(self.results)

    @property
    def graded(self) -> list[CaseResult]:
        return [result for result in self.results if not result.skipped]

    @property
    def skipped_count(self) -> int:
        return self.case_count - len(self.graded)

    @property
    def passed_count(self) -> int:
        return sum(1 for result in self.graded if result.passed)

    @property
    def passed(self) -> bool:
        graded = self.graded
        return bool(graded) and self.passed_count == len(graded)

    def mean(self, metric: str) -> float:
        graded = self.graded
        if not graded:
            return 0.0
        total = sum(getattr(result.scores, metric) for result in graded)
        return round(total / len(graded), 4)

    def aggregate(self) -> dict[str, float]:
        return {
            metric: self.mean(metric)
            for metric in (
                "faithfulness",
                "answer_relevancy",
                "context_precision",
                "context_recall",
                "citation_validity",
            )
        }

    def as_dict(self) -> dict:
        return {
            "passed": self.passed,
            "cases": self.case_count,
            "passed_cases": self.passed_count,
            "skipped_cases": self.skipped_count,
            "duration_ms": self.duration_ms,
            "aggregate": self.aggregate(),
            "thresholds": {
                "faithfulness": self.thresholds.faithfulness,
                "context_precision": self.thresholds.context_precision,
                "citation_validity": self.thresholds.citation_validity,
            },
            "results": [result.as_dict() for result in self.results],
        }

    def summary(self) -> str:
        if not self.results:
            return "No eval cases ran."
        verdict = "PASS" if self.passed else "FAIL"
        aggregate = self.aggregate()
        skipped = f" ({self.skipped_count} skipped)" if self.skipped_count else ""
        return (
            f"{verdict} {self.passed_count}/{len(self.graded)} cases{skipped} | "
            f"faithfulness {aggregate['faithfulness']:.2f} | "
            f"precision {aggregate['context_precision']:.2f} | "
            f"recall {aggregate['context_recall']:.2f}"
        )


def _skipped(case: EvalCase) -> CaseResult:
    return CaseResult(
        case_id=case.id,
        question=case.question,
        answer="",
        scores=MetricScores(),
        source_count=0,
        confidence=0.0,
        duration_ms=0,
        skipped=True,
        failures=["skipped: needs a real LLM provider"],
    )


def looks_like_refusal(answer: str) -> bool:
    lowered = (answer or "").lower()
    return any(marker in lowered for marker in REFUSAL_MARKERS)


def _evaluate(case: EvalCase, result: CaseResult, thresholds: Thresholds) -> None:
    """Decide whether one case passed, recording every reason it did not."""
    failures: list[str] = []

    if case.expect_refusal:
        # An out-of-corpus question has no right answer, so the only thing
        # worth checking is that the system declined instead of inventing one.
        if not result.refused:
            failures.append("expected a refusal but got a substantive answer")
        result.failures = failures
        result.passed = not failures
        return

    if result.refused:
        failures.append("refused an answerable question")

    if result.scores.faithfulness < thresholds.faithfulness:
        failures.append(
            f"faithfulness {result.scores.faithfulness:.2f} < {thresholds.faithfulness:.2f}"
        )
    if result.scores.context_precision < thresholds.context_precision:
        failures.append(
            f"context_precision {result.scores.context_precision:.2f} "
            f"< {thresholds.context_precision:.2f}"
        )
    if result.scores.citation_validity < thresholds.citation_validity:
        failures.append(
            f"citation_validity {result.scores.citation_validity:.2f} "
            f"< {thresholds.citation_validity:.2f}"
        )

    result.failures = failures
    result.passed = not failures


async def run_case(
    db: AsyncSession,
    user,  # noqa: ANN001
    case: EvalCase,
    *,
    gateway: LLMGateway,
    thresholds: Thresholds,
    limit: int = 6,
    **retrieval_kwargs,
) -> CaseResult:
    started = time.time()

    answer = await answer_question(
        db,
        user,
        case.question,
        limit=limit,
        filters=RetrievalFilters(),
        gateway=gateway,
        **retrieval_kwargs,
    )

    contexts = [chunk.content for chunk in answer.sources]
    scores = score_case(
        question=case.question,
        answer=answer.answer,
        contexts=contexts,
        ground_truth=case.ground_truth,
        source_count=len(answer.sources),
    )

    result = CaseResult(
        case_id=case.id,
        question=case.question,
        answer=answer.answer,
        scores=scores,
        source_count=len(answer.sources),
        confidence=answer.confidence,
        duration_ms=int((time.time() - started) * 1000),
        refused=looks_like_refusal(answer.answer) or not answer.sources,
    )
    _evaluate(case, result, thresholds)
    return result


async def run_benchmark(
    db: AsyncSession,
    user,  # noqa: ANN001
    *,
    cases: list[EvalCase] | None = None,
    dataset_path: str | None = None,
    gateway: LLMGateway | None = None,
    thresholds: Thresholds | None = None,
    limit: int = 6,
    **retrieval_kwargs,
) -> EvalReport:
    """Run every case sequentially and gate the aggregate.

    Sequential on purpose: the cases share a retrieval backend and an LLM
    provider, and running them concurrently turns rate-limit behaviour into part
    of what is being measured.
    """
    resolved_cases = cases if cases is not None else load_cases(dataset_path)
    resolved_thresholds = thresholds or Thresholds.from_settings()
    resolved_gateway = gateway or get_gateway()
    have_real_model = resolved_gateway.primary.name != "echo"

    started = time.time()
    report = EvalReport(thresholds=resolved_thresholds)

    for case in resolved_cases:
        if case.requires_llm and not have_real_model:
            report.results.append(_skipped(case))
            continue

        try:
            result = await run_case(
                db,
                user,
                case,
                gateway=resolved_gateway,
                thresholds=resolved_thresholds,
                limit=limit,
                **retrieval_kwargs,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("eval_case_errored", case_id=case.id, error=str(exc))
            result = CaseResult(
                case_id=case.id,
                question=case.question,
                answer="",
                scores=MetricScores(),
                source_count=0,
                confidence=0.0,
                duration_ms=0,
                passed=False,
                failures=[f"errored: {exc}"],
            )
        report.results.append(result)

    report.duration_ms = int((time.time() - started) * 1000)
    logger.info(
        "benchmark_complete",
        cases=report.case_count,
        passed=report.passed_count,
        faithfulness=report.mean("faithfulness"),
    )
    return report
