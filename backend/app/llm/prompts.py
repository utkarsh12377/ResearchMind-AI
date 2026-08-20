"""Central prompt registry.

Prompts are the system's actual behavior specification, so they live in one
versioned module rather than scattered as inline f-strings. That makes them
reviewable in diffs, reusable across agents, and swappable for evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Prompt:
    name: str
    system: str
    template: str

    def render(self, **values: object) -> str:
        return self.template.format(**values)


# --- Grounded question answering -------------------------------------------

RAG_ANSWER = Prompt(
    name="rag_answer",
    system=(
        "You are a research assistant answering questions about scientific papers.\n"
        "\n"
        "Rules:\n"
        "- Answer ONLY from the provided sources. Do not use outside knowledge.\n"
        "- Cite every claim with the bracketed source number, e.g. [1] or [2][3].\n"
        "- If the sources do not contain the answer, say so plainly and explain "
        "what is missing. Never guess.\n"
        "- Quote exact numbers, metric names, and dataset names as they appear.\n"
        "- Prefer precision over completeness. A short, fully supported answer "
        "is better than a long, partly speculative one."
    ),
    template=(
        "Sources:\n{sources}\n\n"
        "Question: {question}\n\n"
        "Answer the question using only the sources above, citing each claim."
    ),
)


# --- Retrieval sufficiency grading (Corrective RAG) ------------------------

GRADE_RETRIEVAL = Prompt(
    name="grade_retrieval",
    system=(
        "You judge whether retrieved sources are sufficient to answer a question.\n"
        "Respond with strict JSON only, no prose, no code fences:\n"
        '{"sufficient": true|false, "reason": "<one sentence>", '
        '"missing": "<what else is needed, or empty string>"}'
    ),
    template="Question: {question}\n\nSources:\n{sources}\n\nAre these sufficient?",
)


# --- Query rewriting for corrective re-retrieval ----------------------------

REWRITE_QUERY = Prompt(
    name="rewrite_query",
    system=(
        "You rewrite search queries to improve retrieval over a corpus of "
        "scientific papers.\n"
        "Return ONLY the rewritten query text, with no preamble or quotes.\n"
        "Prefer specific technical terms, method names, dataset names, and "
        "metric names over general phrasing."
    ),
    template=(
        "Original question: {question}\n"
        "What was missing from the first attempt: {missing}\n\n"
        "Rewritten search query:"
    ),
)


# --- Answer verification (Self-RAG / reflection) ----------------------------

VERIFY_ANSWER = Prompt(
    name="verify_answer",
    system=(
        "You verify whether an answer is fully supported by its sources.\n"
        "Check each claim against the sources. A claim is unsupported if the "
        "sources do not state it, even if it is plausible or generally true.\n"
        "Respond with strict JSON only, no prose, no code fences:\n"
        '{"supported": true|false, "confidence": 0.0-1.0, '
        '"unsupported_claims": ["..."], "reason": "<one sentence>"}'
    ),
    template="Question: {question}\n\nSources:\n{sources}\n\nAnswer:\n{answer}",
)


# --- Context compression ---------------------------------------------------

COMPRESS_CONTEXT = Prompt(
    name="compress_context",
    system=(
        "You compress source passages while preserving everything needed to "
        "answer a specific question.\n"
        "Keep exact numbers, metric names, dataset names, model names, and "
        "stated limitations verbatim. Drop background and unrelated detail.\n"
        "Return only the compressed passage."
    ),
    template="Question: {question}\n\nPassage:\n{passage}\n\nCompressed passage:",
)


REGISTRY = {
    prompt.name: prompt
    for prompt in (
        RAG_ANSWER,
        GRADE_RETRIEVAL,
        REWRITE_QUERY,
        VERIFY_ANSWER,
        COMPRESS_CONTEXT,
    )
}


def get_prompt(name: str) -> Prompt:
    try:
        return REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"Unknown prompt {name!r}. Available: {', '.join(sorted(REGISTRY))}"
        ) from None


def format_sources(chunks) -> str:  # noqa: ANN001
    """Render retrieved chunks as numbered sources for citation.

    Numbering is 1-based and stable within a request, so the bracketed markers
    the model emits can be mapped back to concrete chunks for verification.
    """
    blocks = []
    for index, chunk in enumerate(chunks, start=1):
        header = f"[{index}] {chunk.citation}"
        blocks.append(f"{header}\n{chunk.content.strip()}")
    return "\n\n".join(blocks)
