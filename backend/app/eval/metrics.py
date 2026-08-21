"""Retrieval and answer quality metrics.

These follow the RAGAS definitions -- faithfulness, answer relevancy, context
precision, context recall -- but are computed here rather than pulled from the
library, for two reasons. The library's implementations each require an LLM
judge, which makes the suite unrunnable without keys and non-deterministic when
it does run; and a metric you cannot read the source of is a metric you cannot
debug when it disagrees with your own reading of an answer.

Each metric has a lexical implementation that always works, and can be swapped
for a judged one where a model is available.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.core.logging import get_logger
from app.llm.gateway import LLMGateway, Message, Role
from app.llm.json_output import clamp_confidence, load_json_object
from app.llm.rag import extract_citations

logger = get_logger(__name__)

STOPWORDS = frozenset(
    """a an and are as at be by for from has have in is it its of on or that the
    to was were will with this these those we our they their he she them can could
    should would than then there here when where which who whom what how why
    do does did done been being also into over under between about such only
    very just now both each any all more most other some""".split()
)

SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
MIN_TOKEN_LENGTH = 3
SUPPORT_THRESHOLD = 0.3


@dataclass
class MetricScores:
    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    context_precision: float = 0.0
    context_recall: float = 0.0
    citation_validity: float = 0.0

    @property
    def overall(self) -> float:
        """Unweighted mean.

        Deliberately not weighted: any weighting here would encode an opinion
        about which failure matters most, and that opinion belongs in the
        thresholds a caller sets, not baked into the score.
        """
        values = [
            self.faithfulness,
            self.answer_relevancy,
            self.context_precision,
            self.context_recall,
            self.citation_validity,
        ]
        return round(sum(values) / len(values), 4)

    def as_dict(self) -> dict[str, float]:
        return {
            "faithfulness": round(self.faithfulness, 4),
            "answer_relevancy": round(self.answer_relevancy, 4),
            "context_precision": round(self.context_precision, 4),
            "context_recall": round(self.context_recall, 4),
            "citation_validity": round(self.citation_validity, 4),
            "overall": self.overall,
        }


def tokenize(text: str) -> set[str]:
    """Content words, keeping identifier-shaped tokens intact.

    The character class allows dots and hyphens so "0.89", "rouge-l", and
    "f1.score" survive as single tokens. Trailing punctuation is then stripped,
    without which a sentence-final "encoder." would never match the same word
    written mid-sentence -- which quietly deflates every overlap score.
    """
    words = re.findall(r"[a-z0-9][a-z0-9_\-.]*", (text or "").lower())
    tokens = (word.strip(".-_") for word in words)
    return {t for t in tokens if len(t) >= MIN_TOKEN_LENGTH and t not in STOPWORDS}


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in SENTENCE_SPLIT.split(text or "") if s.strip()]


def _overlap(claim: str, evidence: str) -> float:
    claim_terms = tokenize(claim)
    if not claim_terms:
        return 1.0
    return len(claim_terms & tokenize(evidence)) / len(claim_terms)


def faithfulness(answer: str, contexts: list[str]) -> float:
    """Share of answer sentences that are grounded in the retrieved context.

    A sentence counts as grounded when enough of its content words appear in the
    context. This is a lower bound on true faithfulness -- a paraphrase scores
    below a quotation -- which is the right direction for a safety metric to err.
    """
    sentences = split_sentences(answer)
    if not sentences:
        return 0.0

    evidence = " ".join(contexts)
    if not evidence.strip():
        return 0.0

    grounded = sum(1 for s in sentences if _overlap(s, evidence) >= SUPPORT_THRESHOLD)
    return grounded / len(sentences)


def answer_relevancy(question: str, answer: str) -> float:
    """How much of the question's content the answer actually engages with."""
    question_terms = tokenize(question)
    if not question_terms:
        return 0.0
    if not answer.strip():
        return 0.0
    return len(question_terms & tokenize(answer)) / len(question_terms)


def context_precision(question: str, contexts: list[str], ground_truth: str = "") -> float:
    """Share of retrieved passages that are relevant to the question.

    This is the metric that catches a retriever padding its results: recall can
    be met by returning everything, precision cannot.

    A passage counts as relevant if it overlaps the question or the expected
    answer. Scoring against the question alone punishes exactly the passages
    worth retrieving -- a good source answers the question in the paper's own
    vocabulary rather than echoing the asker's.
    """
    if not contexts:
        return 0.0

    def relevant(context: str) -> bool:
        if _overlap(question, context) >= SUPPORT_THRESHOLD:
            return True
        return bool(ground_truth) and _overlap(ground_truth, context) >= SUPPORT_THRESHOLD

    return sum(1 for context in contexts if relevant(context)) / len(contexts)


def context_recall(ground_truth: str, contexts: list[str]) -> float:
    """How much of the expected answer the retrieved context covers."""
    truth_terms = tokenize(ground_truth)
    if not truth_terms:
        return 1.0
    if not contexts:
        return 0.0
    return len(truth_terms & tokenize(" ".join(contexts))) / len(truth_terms)


def citation_validity(answer: str, source_count: int) -> float:
    """Whether the answer cites, and whether its markers point at real sources.

    Scored as a three-way outcome rather than a ratio: an answer with no
    citations at all is a different failure from one whose citations are wrong,
    and averaging them into one number hides which happened.
    """
    markers = re.findall(r"\[(\d+)\]", answer or "")
    if not markers:
        return 0.0

    valid = extract_citations(answer, source_count)
    distinct = {int(m) for m in markers}
    if not distinct:
        return 0.0
    return len(valid) / len(distinct)


def score_case(
    *,
    question: str,
    answer: str,
    contexts: list[str],
    ground_truth: str = "",
    source_count: int | None = None,
) -> MetricScores:
    return MetricScores(
        faithfulness=faithfulness(answer, contexts),
        answer_relevancy=answer_relevancy(question, answer),
        context_precision=context_precision(question, contexts, ground_truth),
        context_recall=context_recall(ground_truth, contexts) if ground_truth else 1.0,
        citation_validity=citation_validity(
            answer, source_count if source_count is not None else len(contexts)
        ),
    )


JUDGE_SYSTEM = (
    "You grade a retrieval-augmented answer against its sources.\n"
    "Score each dimension from 0.0 to 1.0. Judge only what is written; do not "
    "reward plausible-sounding claims the sources do not support.\n"
    "Respond with strict JSON only, no prose, no code fences:\n"
    '{"faithfulness": 0.0, "answer_relevancy": 0.0, "context_precision": 0.0, '
    '"reason": "<one sentence>"}'
)


@dataclass
class JudgedScore:
    scores: MetricScores
    reason: str = ""
    judged: bool = False
    lexical_fallback: MetricScores | None = field(default=None)


async def judge_case(
    *,
    question: str,
    answer: str,
    contexts: list[str],
    gateway: LLMGateway,
    ground_truth: str = "",
) -> JudgedScore:
    """Grade with a model, falling back to the lexical scores if it fails."""
    lexical = score_case(
        question=question, answer=answer, contexts=contexts, ground_truth=ground_truth
    )

    prompt = (
        f"Question: {question}\n\n"
        f"Sources:\n{chr(10).join(contexts)}\n\n"
        f"Answer:\n{answer}\n\nGrades:"
    )
    try:
        completion = await gateway.complete(
            [Message(Role.SYSTEM, JUDGE_SYSTEM), Message(Role.USER, prompt)],
            temperature=0.0,
            max_tokens=400,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("judge_failed", error=str(exc))
        return JudgedScore(scores=lexical, reason="judge unavailable", judged=False)

    payload = load_json_object(completion.text)
    if payload is None:
        return JudgedScore(scores=lexical, reason="judge output unreadable", judged=False)

    judged = MetricScores(
        faithfulness=clamp_confidence(payload.get("faithfulness"), lexical.faithfulness),
        answer_relevancy=clamp_confidence(
            payload.get("answer_relevancy"), lexical.answer_relevancy
        ),
        context_precision=clamp_confidence(
            payload.get("context_precision"), lexical.context_precision
        ),
        context_recall=lexical.context_recall,
        citation_validity=lexical.citation_validity,
    )
    return JudgedScore(
        scores=judged,
        reason=str(payload.get("reason", ""))[:300],
        judged=True,
        lexical_fallback=lexical,
    )
