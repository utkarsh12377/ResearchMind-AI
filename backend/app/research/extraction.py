"""Pulling structured facts out of papers: results, hyperparameters, methodology.

Prose is a bad database. A question like "which model does best on SQuAD" is
answerable in one query once the numbers live in rows, and unanswerable without
re-reading every PDF if they don't. Extraction runs once per paper at ingestion
time and everything downstream — comparison tables, trends, contradiction
detection — reads the rows rather than the text.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import asdict, dataclass, field

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.graph.extraction import GAZETTEER
from app.graph.schema import NodeLabel, normalize_key
from app.llm.gateway import LLMGateway, Message, Role
from app.llm.json_output import clamp_confidence, load_json_object
from app.llm.prompts import EXTRACT_EXPERIMENTS, EXTRACT_METHODOLOGY
from app.models import DocumentChunk, ExperimentResult, Paper

logger = get_logger(__name__)

MAX_EXTRACTION_CHARS = 18000
MAX_RESULTS_PER_PAPER = 200

METRIC_ALIASES = {
    "f1": "f1",
    "f-1": "f1",
    "f1 score": "f1",
    "f1-score": "f1",
    "em": "exact match",
    "exact-match": "exact match",
    "acc": "accuracy",
    "top-1": "top-1 accuracy",
    "top-5": "top-5 accuracy",
    "rouge-l": "rouge-l",
    "bleu-4": "bleu",
    "ndcg@10": "ndcg",
    "auroc": "roc-auc",
}

#: How far either side of a metric mention to look for a dataset name. One
#: sentence, roughly: "we evaluate on SQuAD and report an F1 of 71.0" should
#: attribute, and a dataset named three paragraphs earlier should not.
DATASET_WINDOW_CHARS = 240

_DATASET_PATTERNS = [
    (term, re.compile(rf"(?<!\w){re.escape(term)}(?!\w)", re.IGNORECASE))
    for term in GAZETTEER[NodeLabel.DATASET.value] + GAZETTEER[NodeLabel.BENCHMARK.value]
]

RESULT_SENTENCE = re.compile(
    r"(?P<metric>accuracy|f1(?:[ -]score)?|exact match|em|bleu|rouge(?:-l)?|meteor|"
    r"perplexity|ndcg(?:@\d+)?|mrr|map(?:@\d+)?|recall(?:@\d+)?|precision|wer|cer|auc|"
    r"roc-auc|top-\d)\s*(?:score\s*)?(?:of|is|was|reaches|reaching|achieves|achieving|=|:)?"
    r"\s*(?P<value>\d{1,3}(?:\.\d+)?)\s*(?P<unit>%)?",
    re.IGNORECASE,
)


@dataclass
class ExtractedResult:
    model: str
    metric: str
    value: float
    dataset: str | None = None
    task: str | None = None
    unit: str | None = None
    split: str | None = None
    confidence: float = 0.5
    evidence: str = ""


@dataclass
class Hyperparameter:
    name: str
    value: str


@dataclass
class Methodology:
    approach: str = ""
    architecture: str = ""
    training: str = ""
    hyperparameters: list[Hyperparameter] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)

    @property
    def is_empty(self) -> bool:
        return not any((self.approach, self.architecture, self.training, self.limitations))


def canonical_metric(name: str) -> str:
    key = normalize_key(name)
    return METRIC_ALIASES.get(key, key)


def nearby_dataset(text: str, position: int) -> str | None:
    """A known dataset named close enough to a metric to plausibly be its subject.

    Restricted to a window around the mention rather than the whole document.
    Attributing a number to whichever dataset happened to appear first in the
    paper is how a rule-based extractor ends up confidently wrong.
    """
    start = max(0, position - DATASET_WINDOW_CHARS)
    window = text[start : position + DATASET_WINDOW_CHARS]

    closest: tuple[int, str] | None = None
    for term, pattern in _DATASET_PATTERNS:
        match = pattern.search(window)
        if match is None:
            continue
        distance = abs((start + match.start()) - position)
        if closest is None or distance < closest[0]:
            closest = (distance, term)

    return closest[1] if closest else None


def extract_results_with_rules(text: str, *, default_model: str = "") -> list[ExtractedResult]:
    """Regex pass over result sentences.

    Deliberately conservative. It only fires on a metric name adjacent to a
    number, and attributes the number to the paper's own model unless the LLM
    pass says otherwise, because guessing the model from prose is exactly where
    a rule-based extractor starts inventing things.

    The dataset is treated differently: it comes from the known-entity lexicon
    and only counts when it appears within about a sentence of the metric.
    Without it every offline extraction is ungroupable, since comparison needs
    a dataset to decide which numbers belong in the same cell.
    """
    results: list[ExtractedResult] = []
    seen: set[tuple[str, float]] = set()
    haystack = text[:MAX_EXTRACTION_CHARS]

    for match in RESULT_SENTENCE.finditer(haystack):
        try:
            value = float(match.group("value"))
        except ValueError:
            continue

        metric = canonical_metric(match.group("metric"))
        if (metric, value) in seen:
            continue
        seen.add((metric, value))

        dataset = nearby_dataset(haystack, match.start())
        start = max(0, match.start() - 100)
        results.append(
            ExtractedResult(
                model=default_model or "reported system",
                metric=metric,
                value=value,
                dataset=dataset,
                unit=match.group("unit"),
                # Slightly higher when a dataset was found, since a metric
                # sitting next to a known benchmark name is more likely to be a
                # real reported result than a number in passing.
                confidence=0.5 if dataset else 0.45,
                evidence=haystack[start : match.end() + 40].replace("\n", " ").strip(),
            )
        )
        if len(results) >= MAX_RESULTS_PER_PAPER:
            break

    return results


def parse_results_payload(raw: str) -> list[ExtractedResult]:
    payload = load_json_object(raw)
    if payload is None:
        return []

    results = []
    for item in payload.get("results", []) or []:
        if not isinstance(item, dict):
            continue
        model = str(item.get("model", "")).strip()
        metric = str(item.get("metric", "")).strip()
        if not model or not metric:
            continue
        try:
            value = float(item.get("value"))
        except (TypeError, ValueError):
            continue

        results.append(
            ExtractedResult(
                model=model[:255],
                metric=canonical_metric(metric)[:128],
                value=value,
                dataset=(str(item.get("dataset", "")).strip() or None),
                task=(str(item.get("task", "")).strip() or None),
                unit=(str(item.get("unit", "")).strip() or None),
                split=(str(item.get("split", "")).strip() or None),
                confidence=clamp_confidence(item.get("confidence")),
            )
        )
    return results[:MAX_RESULTS_PER_PAPER]


def parse_methodology_payload(raw: str) -> Methodology:
    payload = load_json_object(raw)
    if payload is None:
        return Methodology()

    hyperparameters = []
    for item in payload.get("hyperparameters", []) or []:
        if isinstance(item, dict) and item.get("name"):
            hyperparameters.append(
                Hyperparameter(str(item["name"])[:120], str(item.get("value", ""))[:120])
            )

    limitations = [
        str(item)[:400]
        for item in (payload.get("limitations", []) or [])
        if str(item).strip()
    ]

    return Methodology(
        approach=str(payload.get("approach", ""))[:2000],
        architecture=str(payload.get("architecture", ""))[:2000],
        training=str(payload.get("training", ""))[:2000],
        hyperparameters=hyperparameters[:40],
        limitations=limitations[:20],
    )


async def paper_text(db: AsyncSession, paper: Paper, *, limit_chunks: int = 40) -> str:
    parts = [paper.abstract] if paper.abstract else []
    chunks = (
        await db.scalars(
            select(DocumentChunk)
            .where(DocumentChunk.paper_id == paper.id)
            .order_by(DocumentChunk.chunk_index)
            .limit(limit_chunks)
        )
    ).all()
    parts.extend(chunk.content for chunk in chunks)
    return "\n\n".join(part for part in parts if part)[:MAX_EXTRACTION_CHARS]


async def extract_experiments(
    db: AsyncSession,
    paper_id: uuid.UUID,
    *,
    gateway: LLMGateway | None = None,
    use_llm: bool = True,
) -> list[ExtractedResult]:
    paper = await db.get(Paper, paper_id)
    if paper is None:
        raise ValueError(f"Paper {paper_id} does not exist")

    text = await paper_text(db, paper)
    if not text.strip():
        return []

    title = paper.title or paper.original_filename
    results = extract_results_with_rules(text, default_model=title[:80])

    if use_llm and gateway is not None:
        try:
            completion = await gateway.complete(
                [
                    Message(Role.SYSTEM, EXTRACT_EXPERIMENTS.system),
                    Message(
                        Role.USER,
                        EXTRACT_EXPERIMENTS.render(paper_title=title, text=text),
                    ),
                ],
                temperature=0.0,
                max_tokens=1500,
            )
            llm_results = parse_results_payload(completion.text)
            if llm_results:
                results = _merge_results(llm_results, results)
        except Exception as exc:  # noqa: BLE001
            logger.warning("experiment_extraction_failed", error=str(exc))

    await _persist_results(db, paper_id, results)
    logger.info("experiments_extracted", paper_id=str(paper_id), count=len(results))
    return results


def _merge_results(
    primary: list[ExtractedResult], secondary: list[ExtractedResult]
) -> list[ExtractedResult]:
    """LLM rows win; rule rows only fill metrics the model missed entirely."""
    covered = {canonical_metric(r.metric) for r in primary}
    merged = list(primary)
    merged.extend(r for r in secondary if canonical_metric(r.metric) not in covered)
    return merged


async def _persist_results(
    db: AsyncSession, paper_id: uuid.UUID, results: list[ExtractedResult]
) -> None:
    await db.execute(delete(ExperimentResult).where(ExperimentResult.paper_id == paper_id))
    await db.flush()

    for item in results:
        db.add(
            ExperimentResult(
                paper_id=paper_id,
                model_name=item.model[:255],
                model_key=normalize_key(item.model)[:255],
                dataset_name=item.dataset[:255] if item.dataset else None,
                dataset_key=normalize_key(item.dataset)[:255] if item.dataset else None,
                task_name=item.task[:255] if item.task else None,
                metric_name=item.metric[:128],
                metric_key=canonical_metric(item.metric)[:128],
                value=item.value,
                unit=item.unit[:32] if item.unit else None,
                split=item.split[:64] if item.split else None,
                confidence=item.confidence,
                evidence=item.evidence or None,
            )
        )
    await db.commit()


async def extract_methodology(
    db: AsyncSession,
    paper_id: uuid.UUID,
    *,
    gateway: LLMGateway,
) -> Methodology:
    paper = await db.get(Paper, paper_id)
    if paper is None:
        raise ValueError(f"Paper {paper_id} does not exist")

    text = await paper_text(db, paper)
    if not text.strip():
        return Methodology()

    try:
        completion = await gateway.complete(
            [
                Message(Role.SYSTEM, EXTRACT_METHODOLOGY.system),
                Message(
                    Role.USER,
                    EXTRACT_METHODOLOGY.render(
                        paper_title=paper.title or paper.original_filename, text=text
                    ),
                ),
            ],
            temperature=0.0,
            max_tokens=1200,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("methodology_extraction_failed", error=str(exc))
        return Methodology()

    return parse_methodology_payload(completion.text)
