"""Finding where papers agree and where they disagree.

Two detectors, because disagreement shows up in two different shapes. Numeric
conflicts are found arithmetically: the same model, dataset, and metric with
materially different numbers is a fact about the rows, not a judgement call.
Claim-level conflicts need a model, but only after a cheap filter has narrowed
the candidate pairs — comparing every passage against every other passage is
quadratic and mostly compares unrelated things.
"""

from __future__ import annotations

import asyncio
import uuid
from collections import defaultdict
from dataclasses import dataclass, field

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.llm.gateway import LLMGateway, Message, Role
from app.llm.json_output import clamp_confidence, load_json_object
from app.llm.prompts import COMPARE_CLAIMS
from app.models import DocumentChunk, ExperimentResult, Paper
from app.research.comparison import metric_higher_is_better

logger = get_logger(__name__)

#: Relative gap above which two reported numbers are treated as a conflict.
#: Below this, the difference is within the range of ordinary run-to-run
#: variance and reporting-precision differences.
NUMERIC_CONFLICT_THRESHOLD = 0.05
MAX_CLAIM_PAIRS = 24
MIN_CLAIM_CHARS = 60
MAX_CLAIM_CHARS = 600
CLAIM_OVERLAP_THRESHOLD = 0.25

CLAIM_MARKERS = (
    "we find", "we show", "we observe", "results show", "our results",
    "outperforms", "improves", "degrades", "significantly", "demonstrates",
    "suggests that", "indicates that", "conclude", "however", "in contrast",
)


@dataclass
class NumericConflict:
    model: str
    dataset: str
    metric: str
    left_paper: str
    left_value: float
    right_paper: str
    right_value: float

    @property
    def relative_gap(self) -> float:
        scale = max(abs(self.left_value), abs(self.right_value), 1e-9)
        return abs(self.left_value - self.right_value) / scale

    def describe(self) -> str:
        return (
            f"{self.model} on {self.dataset} ({self.metric}): "
            f"{self.left_value:g} in '{self.left_paper}' vs "
            f"{self.right_value:g} in '{self.right_paper}'"
        )


@dataclass
class ClaimPair:
    relation: str
    confidence: float
    reason: str
    left_paper: str
    left_claim: str
    right_paper: str
    right_claim: str


@dataclass
class ConsistencyReport:
    numeric_conflicts: list[NumericConflict] = field(default_factory=list)
    contradictions: list[ClaimPair] = field(default_factory=list)
    agreements: list[ClaimPair] = field(default_factory=list)
    papers_examined: int = 0

    @property
    def is_consistent(self) -> bool:
        return not self.numeric_conflicts and not self.contradictions

    def summary(self) -> str:
        if self.papers_examined < 2:
            return "At least two papers are needed to compare claims."
        if self.is_consistent:
            return f"No conflicts found across {self.papers_examined} papers."
        return (
            f"{len(self.numeric_conflicts)} numeric conflict(s) and "
            f"{len(self.contradictions)} contradicting claim(s) across "
            f"{self.papers_examined} papers."
        )


async def find_numeric_conflicts(
    db: AsyncSession,
    paper_ids: list[uuid.UUID],
    *,
    threshold: float = NUMERIC_CONFLICT_THRESHOLD,
) -> list[NumericConflict]:
    if len(paper_ids) < 2:
        return []

    rows = (
        await db.execute(
            select(ExperimentResult, Paper.title)
            .join(Paper, Paper.id == ExperimentResult.paper_id)
            .where(ExperimentResult.paper_id.in_(paper_ids))
        )
    ).all()

    grouped: dict[tuple[str, str, str], list[tuple[ExperimentResult, str]]] = defaultdict(list)
    for result, title in rows:
        if not result.dataset_key:
            continue
        grouped[(result.model_key, result.dataset_key, result.metric_key)].append(
            (result, title or "Untitled")
        )

    conflicts: list[NumericConflict] = []
    for (_, _, _), entries in grouped.items():
        by_paper = {entry[0].paper_id: entry for entry in entries}
        if len(by_paper) < 2:
            continue

        ordered = list(by_paper.values())
        for index, (left, left_title) in enumerate(ordered):
            for right, right_title in ordered[index + 1 :]:
                conflict = NumericConflict(
                    model=left.model_name,
                    dataset=left.dataset_name or left.dataset_key or "unknown",
                    metric=left.metric_name,
                    left_paper=left_title,
                    left_value=left.value,
                    right_paper=right_title,
                    right_value=right.value,
                )
                if conflict.relative_gap >= threshold:
                    conflicts.append(conflict)

    conflicts.sort(key=lambda c: c.relative_gap, reverse=True)
    logger.info("numeric_conflicts_found", count=len(conflicts))
    return conflicts


def extract_claims(text: str, *, limit: int = 6) -> list[str]:
    """Sentences that assert a finding, rather than describe setup."""
    claims = []
    for raw in _split_sentences(text):
        sentence = raw.strip()
        if not MIN_CLAIM_CHARS <= len(sentence) <= MAX_CLAIM_CHARS:
            continue
        lowered = sentence.lower()
        if any(marker in lowered for marker in CLAIM_MARKERS):
            claims.append(sentence)
        if len(claims) >= limit:
            break
    return claims


def _split_sentences(text: str) -> list[str]:
    parts: list[str] = []
    current: list[str] = []
    for char in text:
        current.append(char)
        if char in ".!?":
            parts.append("".join(current))
            current = []
    if current:
        parts.append("".join(current))
    return parts


def _topical_overlap(left: str, right: str) -> float:
    left_terms = {w for w in left.lower().split() if len(w) > 4}
    right_terms = {w for w in right.lower().split() if len(w) > 4}
    if not left_terms or not right_terms:
        return 0.0
    return len(left_terms & right_terms) / len(left_terms | right_terms)


def _parse_judgement(raw: str) -> tuple[str, float, str]:
    payload = load_json_object(raw)
    if payload is None:
        return "unrelated", 0.0, "Could not parse the comparison"

    relation = str(payload.get("relation", "unrelated")).strip().lower()
    if relation not in {"agreement", "contradiction", "unrelated"}:
        relation = "unrelated"
    return (
        relation,
        clamp_confidence(payload.get("confidence"), default=0.0),
        str(payload.get("reason", ""))[:400],
    )


async def compare_claims(
    db: AsyncSession,
    paper_ids: list[uuid.UUID],
    *,
    gateway: LLMGateway,
    max_pairs: int = MAX_CLAIM_PAIRS,
) -> tuple[list[ClaimPair], list[ClaimPair]]:
    if len(paper_ids) < 2:
        return [], []

    claims_by_paper: dict[str, list[str]] = {}
    for paper_id in paper_ids:
        paper = await db.get(Paper, paper_id)
        if paper is None:
            continue
        chunks = (
            await db.scalars(
                select(DocumentChunk)
                .where(DocumentChunk.paper_id == paper_id)
                .order_by(DocumentChunk.chunk_index)
                .limit(30)
            )
        ).all()
        text = " ".join(chunk.content for chunk in chunks)
        found = extract_claims(text)
        if found:
            claims_by_paper[paper.title or paper.original_filename] = found

    pairs = _candidate_pairs(claims_by_paper, max_pairs)
    if not pairs:
        return [], []

    judgements = await asyncio.gather(
        *(_judge(gateway, *pair) for pair in pairs), return_exceptions=True
    )

    contradictions: list[ClaimPair] = []
    agreements: list[ClaimPair] = []
    for result in judgements:
        if isinstance(result, BaseException) or result is None:
            continue
        if result.relation == "contradiction":
            contradictions.append(result)
        elif result.relation == "agreement":
            agreements.append(result)

    contradictions.sort(key=lambda c: c.confidence, reverse=True)
    agreements.sort(key=lambda c: c.confidence, reverse=True)
    return contradictions, agreements


def _candidate_pairs(
    claims_by_paper: dict[str, list[str]], max_pairs: int
) -> list[tuple[str, str, str, str]]:
    """Pair claims from different papers that are about the same thing."""
    titles = list(claims_by_paper)
    scored: list[tuple[float, tuple[str, str, str, str]]] = []

    for i, left_title in enumerate(titles):
        for right_title in titles[i + 1 :]:
            for left_claim in claims_by_paper[left_title]:
                for right_claim in claims_by_paper[right_title]:
                    overlap = _topical_overlap(left_claim, right_claim)
                    if overlap >= CLAIM_OVERLAP_THRESHOLD:
                        scored.append(
                            (overlap, (left_title, left_claim, right_title, right_claim))
                        )

    scored.sort(key=lambda item: item[0], reverse=True)
    return [pair for _, pair in scored[:max_pairs]]


async def _judge(
    gateway: LLMGateway,
    left_paper: str,
    left_claim: str,
    right_paper: str,
    right_claim: str,
) -> ClaimPair | None:
    try:
        completion = await gateway.complete(
            [
                Message(Role.SYSTEM, COMPARE_CLAIMS.system),
                Message(
                    Role.USER,
                    COMPARE_CLAIMS.render(
                        paper_a=left_paper,
                        claim_a=left_claim,
                        paper_b=right_paper,
                        claim_b=right_claim,
                    ),
                ),
            ],
            temperature=0.0,
            max_tokens=250,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("claim_comparison_failed", error=str(exc))
        return None

    relation, confidence, reason = _parse_judgement(completion.text)
    if relation == "unrelated":
        return None

    return ClaimPair(
        relation=relation,
        confidence=confidence,
        reason=reason,
        left_paper=left_paper,
        left_claim=left_claim,
        right_paper=right_paper,
        right_claim=right_claim,
    )


async def analyze_consistency(
    db: AsyncSession,
    paper_ids: list[uuid.UUID],
    *,
    gateway: LLMGateway | None = None,
    include_claims: bool = True,
) -> ConsistencyReport:
    report = ConsistencyReport(papers_examined=len(paper_ids))
    report.numeric_conflicts = await find_numeric_conflicts(db, paper_ids)

    if include_claims and gateway is not None:
        contradictions, agreements = await compare_claims(db, paper_ids, gateway=gateway)
        report.contradictions = contradictions
        report.agreements = agreements

    logger.info(
        "consistency_analyzed",
        papers=len(paper_ids),
        numeric=len(report.numeric_conflicts),
        contradictions=len(report.contradictions),
    )
    return report


def better_of(metric: str, left: float, right: float) -> float:
    return max(left, right) if metric_higher_is_better(metric) else min(left, right)
