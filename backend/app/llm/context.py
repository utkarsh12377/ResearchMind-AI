"""Context assembly under a token budget.

Retrieval routinely returns more relevant text than fits in a prompt, and
naively truncating the tail silently discards the lowest-ranked sources — which
are often the ones that would have covered the gap. This module makes the
trade-off explicit: pack what fits, and when it doesn't fit, compress rather
than drop.

Two strategies, cheapest first:
1. Extractive selection — keep the sentences that actually bear on the query.
   Free, fast, and lossless for the sentences it keeps.
2. Abstractive compression — an LLM rewrite, used only when extraction alone
   still overflows. Costs a model call per passage, so it is opt-in.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.core.logging import get_logger
from app.llm.gateway import LLMGateway, Message, Role
from app.llm.prompts import COMPRESS_CONTEXT
from app.retrieval.reranker import extract_supporting_sentences

logger = get_logger(__name__)

# Matches the estimate used in chunking, so budgets are consistent end to end.
CHARS_PER_TOKEN = 4

# Leave room for the system prompt, the question, and the model's own output.
DEFAULT_RESERVED_TOKENS = 1200


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN)


@dataclass
class PackedContext:
    chunks: list
    used_tokens: int
    dropped_count: int
    compressed_count: int

    @property
    def was_reduced(self) -> bool:
        return self.dropped_count > 0 or self.compressed_count > 0


def _shrink_by_extraction(chunk, query: str, target_tokens: int):  # noqa: ANN001, ANN202
    """Replace a chunk's text with only its query-relevant sentences."""
    sentences = extract_supporting_sentences(query, chunk.content, limit=6)
    if not sentences:
        return None

    kept: list[str] = []
    used = 0
    for sentence in sentences:
        cost = estimate_tokens(sentence)
        if used + cost > target_tokens:
            break
        kept.append(sentence)
        used += cost

    if not kept:
        return None

    import copy

    reduced = copy.copy(chunk)
    reduced.content = " ".join(kept)
    return reduced


def pack_context(
    chunks: list,
    query: str,
    *,
    max_tokens: int,
    reserved_tokens: int = DEFAULT_RESERVED_TOKENS,
) -> PackedContext:
    """Fit chunks into a token budget, preferring extraction over dropping.

    Chunks arrive in relevance order, so packing greedily from the top keeps
    the best evidence. A chunk that doesn't fit whole is first reduced to its
    query-relevant sentences; only if that still doesn't fit is it dropped.
    """
    budget = max(0, max_tokens - reserved_tokens)
    packed: list = []
    used = 0
    dropped = 0
    compressed = 0

    for chunk in chunks:
        cost = estimate_tokens(chunk.content)

        if used + cost <= budget:
            packed.append(chunk)
            used += cost
            continue

        remaining = budget - used
        # Below this there isn't room for a meaningful excerpt, so stop rather
        # than emitting a fragment that costs tokens and says nothing.
        if remaining < 40:
            dropped += 1
            continue

        reduced = _shrink_by_extraction(chunk, query, remaining)
        if reduced is None:
            dropped += 1
            continue

        packed.append(reduced)
        used += estimate_tokens(reduced.content)
        compressed += 1

    if dropped or compressed:
        logger.info(
            "context_packed",
            kept=len(packed),
            dropped=dropped,
            compressed=compressed,
            used_tokens=used,
            budget=budget,
        )

    return PackedContext(
        chunks=packed, used_tokens=used, dropped_count=dropped, compressed_count=compressed
    )


async def compress_chunk(
    gateway: LLMGateway, chunk, query: str, *, max_tokens: int = 256  # noqa: ANN001
):
    """Abstractively compress one passage against a query.

    Only worth its cost for long passages where extraction loses too much;
    callers decide when to reach for it.
    """
    completion = await gateway.complete(
        [
            Message(Role.SYSTEM, COMPRESS_CONTEXT.system),
            Message(
                Role.USER,
                COMPRESS_CONTEXT.render(question=query, passage=chunk.content),
            ),
        ],
        temperature=0.0,
        max_tokens=max_tokens,
    )

    import copy

    compressed = copy.copy(chunk)
    compressed.content = completion.text.strip() or chunk.content
    return compressed


async def compress_context(
    gateway: LLMGateway,
    chunks: list,
    query: str,
    *,
    max_tokens: int,
    reserved_tokens: int = DEFAULT_RESERVED_TOKENS,
) -> PackedContext:
    """Pack context, falling back to LLM compression when extraction isn't enough.

    Passages are compressed concurrently — they are independent, and doing them
    serially would multiply latency by the number of oversized chunks.
    """
    packed = pack_context(
        chunks, query, max_tokens=max_tokens, reserved_tokens=reserved_tokens
    )
    if not packed.dropped_count:
        return packed

    import asyncio

    budget = max(0, max_tokens - reserved_tokens)
    compressed_chunks = await asyncio.gather(
        *(compress_chunk(gateway, chunk, query) for chunk in chunks[: len(chunks)]),
        return_exceptions=True,
    )

    usable = [c for c in compressed_chunks if not isinstance(c, Exception)]
    failures = len(compressed_chunks) - len(usable)
    if failures:
        logger.warning("context_compression_partial_failure", failed=failures)

    return pack_context(usable, query, max_tokens=budget + reserved_tokens,
                        reserved_tokens=reserved_tokens)
