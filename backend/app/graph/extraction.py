"""Turning paper text into graph entities and relations.

Two extractors run in sequence. A gazetteer pass catches the well-known names
(datasets, model families, metrics, tasks) with no model call at all; the LLM
pass then handles everything the lexicon can't know about. Running the cheap one
first means the graph is never empty just because no API key is configured, and
it gives the LLM output something to be checked against.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.core.logging import get_logger
from app.graph.schema import ALLOWED_EDGES, ENTITY_LABELS, NodeLabel, RelationType, normalize_key
from app.llm.gateway import LLMGateway, Message, Role
from app.llm.json_output import clamp_confidence, load_json_object
from app.llm.prompts import EXTRACT_ENTITIES

logger = get_logger(__name__)

MAX_TEXT_CHARS = 12000
MIN_MENTION_CHARS = 2


@dataclass
class EntityMention:
    label: str
    name: str
    confidence: float = 0.5
    evidence: str = ""

    @property
    def key(self) -> str:
        return normalize_key(self.name)


@dataclass
class RelationMention:
    type: str
    source: str
    target: str
    confidence: float = 0.5
    evidence: str = ""


@dataclass
class ExtractionResult:
    entities: list[EntityMention] = field(default_factory=list)
    relations: list[RelationMention] = field(default_factory=list)

    def merge(self, other: ExtractionResult) -> ExtractionResult:
        seen = {(e.label, e.key) for e in self.entities}
        for entity in other.entities:
            if (entity.label, entity.key) not in seen:
                seen.add((entity.label, entity.key))
                self.entities.append(entity)

        seen_rel = {
            (r.type, normalize_key(r.source), normalize_key(r.target)) for r in self.relations
        }
        for relation in other.relations:
            signature = (
                relation.type,
                normalize_key(relation.source),
                normalize_key(relation.target),
            )
            if signature not in seen_rel:
                seen_rel.add(signature)
                self.relations.append(relation)
        return self


GAZETTEER: dict[str, tuple[str, ...]] = {
    NodeLabel.DATASET.value: (
        "imagenet", "mnist", "cifar-10", "cifar-100", "coco", "squad", "squad 2.0",
        "glue", "superglue", "wikitext", "c4", "the pile", "common crawl", "ms marco",
        "natural questions", "hotpotqa", "triviaqa", "beir", "wikisql", "spider",
        "librispeech", "conll-2003", "snli", "multinli", "openwebtext", "arxiv",
    ),
    NodeLabel.MODEL.value: (
        "bert", "roberta", "albert", "distilbert", "electra", "gpt-2", "gpt-3", "gpt-4",
        "t5", "flan-t5", "bart", "llama", "llama 2", "llama 3", "mistral", "mixtral",
        "claude", "gemini", "palm", "vit", "resnet", "efficientnet", "clip", "whisper",
        "transformer", "lstm", "gru", "colbert", "dpr", "splade", "e5", "bge",
    ),
    NodeLabel.METRIC.value: (
        "accuracy", "f1", "f1 score", "precision", "recall", "bleu", "rouge", "rouge-l",
        "meteor", "perplexity", "map", "mrr", "ndcg", "auc", "roc-auc", "exact match",
        "wer", "cer", "iou", "map@k", "recall@k", "hit rate",
    ),
    NodeLabel.TASK.value: (
        "question answering", "text classification", "named entity recognition",
        "machine translation", "summarization", "sentiment analysis", "image classification",
        "object detection", "semantic segmentation", "speech recognition",
        "information retrieval", "language modeling", "code generation",
        "reading comprehension", "entity linking", "relation extraction",
    ),
    NodeLabel.METHOD.value: (
        "self-attention", "multi-head attention", "contrastive learning", "fine-tuning",
        "prompt tuning", "lora", "qlora", "knowledge distillation", "quantization",
        "reinforcement learning from human feedback", "rlhf", "chain-of-thought",
        "retrieval-augmented generation", "beam search", "dropout", "batch normalization",
        "layer normalization", "gradient checkpointing", "mixture of experts",
    ),
    NodeLabel.BENCHMARK.value: (
        "glue", "superglue", "mmlu", "big-bench", "helm", "beir", "hellaswag",
        "arc", "truthfulqa", "gsm8k", "humaneval", "mbpp", "lambada",
    ),
}

_GAZETTEER_PATTERNS = {
    label: [(term, re.compile(rf"(?<!\w){re.escape(term)}(?!\w)", re.IGNORECASE)) for term in terms]
    for label, terms in GAZETTEER.items()
}

_RELATION_FOR_LABEL = {
    NodeLabel.DATASET.value: RelationType.USES_DATASET.value,
    NodeLabel.MODEL.value: RelationType.PROPOSES.value,
    NodeLabel.METRIC.value: RelationType.REPORTS.value,
    NodeLabel.TASK.value: RelationType.ADDRESSES.value,
    NodeLabel.METHOD.value: RelationType.USES_METHOD.value,
    NodeLabel.BENCHMARK.value: RelationType.EVALUATES_ON.value,
}


def extract_with_gazetteer(text: str, *, paper_title: str = "") -> ExtractionResult:
    """Match known research entities by surface form."""
    result = ExtractionResult()
    if not text.strip():
        return result

    haystack = text[:MAX_TEXT_CHARS]
    for label, patterns in _GAZETTEER_PATTERNS.items():
        for term, pattern in patterns:
            match = pattern.search(haystack)
            if match is None:
                continue
            result.entities.append(
                EntityMention(
                    label=label,
                    name=term,
                    confidence=0.9,
                    evidence=_snippet(haystack, match.start()),
                )
            )
            relation = _RELATION_FOR_LABEL.get(label)
            if relation and paper_title:
                result.relations.append(
                    RelationMention(relation, paper_title, term, confidence=0.9)
                )
    return result


def _snippet(text: str, position: int, width: int = 120) -> str:
    start = max(0, position - width // 2)
    return text[start : start + width].replace("\n", " ").strip()


async def extract_with_llm(
    text: str, *, paper_title: str, gateway: LLMGateway
) -> ExtractionResult:
    """Ask the model for entities and relations the lexicon would miss."""
    if not text.strip():
        return ExtractionResult()

    prompt = EXTRACT_ENTITIES.render(
        paper_title=paper_title or "Untitled",
        labels=", ".join(sorted(ENTITY_LABELS)),
        relations=", ".join(sorted(ALLOWED_EDGES)),
        text=text[:MAX_TEXT_CHARS],
    )
    completion = await gateway.complete(
        [
            Message(Role.SYSTEM, EXTRACT_ENTITIES.system),
            Message(Role.USER, prompt),
        ],
        temperature=0.0,
        max_tokens=1500,
    )
    return parse_extraction_payload(completion.text, paper_title=paper_title)


def parse_extraction_payload(raw: str, *, paper_title: str = "") -> ExtractionResult:
    """Read the model's JSON, discarding anything the schema doesn't allow.

    Extraction is the one place where a hallucination becomes durable state, so
    unknown labels, unknown relation types, and edges between the wrong kinds of
    node are dropped here rather than written and cleaned up later.
    """
    result = ExtractionResult()
    payload = load_json_object(raw)
    if payload is None:
        return result

    for item in payload.get("entities", []) or []:
        if not isinstance(item, dict):
            continue
        label = str(item.get("type", "")).strip().title().replace(" ", "")
        name = str(item.get("name", "")).strip()
        if label not in ENTITY_LABELS or len(name) < MIN_MENTION_CHARS:
            continue
        result.entities.append(
            EntityMention(
                label=label,
                name=name,
                confidence=clamp_confidence(item.get("confidence", 0.6)),
                evidence=str(item.get("evidence", ""))[:300],
            )
        )

    known = {normalize_key(e.name) for e in result.entities}
    known.add(normalize_key(paper_title))

    for item in payload.get("relations", []) or []:
        if not isinstance(item, dict):
            continue
        rel_type = str(item.get("type", "")).strip().upper().replace(" ", "_")
        source = str(item.get("source", "")).strip()
        target = str(item.get("target", "")).strip()
        if rel_type not in ALLOWED_EDGES or not source or not target:
            continue
        if normalize_key(source) not in known or normalize_key(target) not in known:
            logger.debug("relation_dropped_unknown_endpoint", type=rel_type)
            continue
        result.relations.append(
            RelationMention(
                type=rel_type,
                source=source,
                target=target,
                confidence=clamp_confidence(item.get("confidence", 0.6)),
                evidence=str(item.get("evidence", ""))[:300],
            )
        )

    return result


async def extract_entities(
    text: str,
    *,
    paper_title: str = "",
    gateway: LLMGateway | None = None,
    use_llm: bool = True,
) -> ExtractionResult:
    result = extract_with_gazetteer(text, paper_title=paper_title)

    if use_llm and gateway is not None:
        try:
            llm_result = await extract_with_llm(text, paper_title=paper_title, gateway=gateway)
            result.merge(llm_result)
        except Exception as exc:  # noqa: BLE001
            logger.warning("llm_extraction_failed", error=str(exc))

    logger.info(
        "entities_extracted",
        entities=len(result.entities),
        relations=len(result.relations),
        title=paper_title[:80],
    )
    return result
