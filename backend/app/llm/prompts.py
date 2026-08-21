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


# --- Knowledge graph extraction --------------------------------------------

EXTRACT_ENTITIES = Prompt(
    name="extract_entities",
    system=(
        "You extract a knowledge graph from scientific text.\n"
        "Only use the entity types and relation types you are given. Never "
        "invent a type. Only extract what the text actually states; if a "
        "relation is implied but not stated, leave it out.\n"
        "Use the entity name exactly as written in the text.\n"
        "Respond with strict JSON only, no prose, no code fences:\n"
        '{"entities": [{"type": "...", "name": "...", "confidence": 0.0-1.0, '
        '"evidence": "<quote>"}], "relations": [{"type": "...", "source": "...", '
        '"target": "...", "confidence": 0.0-1.0}]}'
    ),
    template=(
        "Paper: {paper_title}\n"
        "Entity types: {labels}\n"
        "Relation types: {relations}\n\n"
        "Text:\n{text}\n\n"
        "Extraction:"
    ),
)


NL_TO_CYPHER = Prompt(
    name="nl_to_cypher",
    system=(
        "You translate questions into read-only Cypher for a research "
        "knowledge graph.\n"
        "Rules:\n"
        "- Use only the labels, relationship types, and properties in the schema.\n"
        "- Read-only: MATCH, OPTIONAL MATCH, WHERE, WITH, RETURN, ORDER BY, LIMIT.\n"
        "- Never write CREATE, MERGE, SET, DELETE, REMOVE, LOAD, or CALL.\n"
        "- Always end with a LIMIT clause.\n"
        "- Node keys are lowercase; compare against lowercase literals.\n"
        "Return only the Cypher query, nothing else."
    ),
    template="Schema:\n{schema}\n\nQuestion: {question}\n\nCypher:",
)


# --- Cross-paper analysis ---------------------------------------------------

COMPARE_CLAIMS = Prompt(
    name="compare_claims",
    system=(
        "You judge whether two statements from different papers agree, "
        "contradict, or are unrelated.\n"
        "Contradiction requires that both statements are about the same thing "
        "and cannot both be true. Different results on different datasets or "
        "under different settings are not a contradiction.\n"
        "Respond with strict JSON only, no prose, no code fences:\n"
        '{"relation": "agreement"|"contradiction"|"unrelated", '
        '"confidence": 0.0-1.0, "reason": "<one sentence>"}'
    ),
    template=(
        "Statement A ({paper_a}):\n{claim_a}\n\n"
        "Statement B ({paper_b}):\n{claim_b}\n\nJudgement:"
    ),
)


EXTRACT_EXPERIMENTS = Prompt(
    name="extract_experiments",
    system=(
        "You extract reported experimental results from scientific text.\n"
        "One record per (model, dataset, metric) triple that has a number "
        "attached. Do not extract targets, hypotheses, or numbers from related "
        "work unless they are attributed to a named model.\n"
        "Values are plain numbers, with any unit reported separately.\n"
        "Respond with strict JSON only, no prose, no code fences:\n"
        '{"results": [{"model": "...", "dataset": "...", "task": "...", '
        '"metric": "...", "value": 0.0, "unit": "...", "split": "...", '
        '"confidence": 0.0-1.0}]}'
    ),
    template="Paper: {paper_title}\n\nText:\n{text}\n\nResults:",
)


EXTRACT_METHODOLOGY = Prompt(
    name="extract_methodology",
    system=(
        "You summarize the methodology of a paper in a fixed structure.\n"
        "Use only what the text states; write 'not stated' for anything absent.\n"
        "Respond with strict JSON only, no prose, no code fences:\n"
        '{"approach": "...", "architecture": "...", "training": "...", '
        '"hyperparameters": [{"name": "...", "value": "..."}], '
        '"limitations": ["..."]}'
    ),
    template="Paper: {paper_title}\n\nText:\n{text}\n\nMethodology:",
)


SYNTHESIZE_TREND = Prompt(
    name="synthesize_trend",
    system=(
        "You describe how a research area changed over time, based only on the "
        "supplied per-year evidence.\n"
        "Be concrete: name the shift, when it happened, and what drove it. "
        "Do not speculate beyond the evidence. Three sentences maximum."
    ),
    template="Topic: {topic}\n\nEvidence by year:\n{evidence}\n\nTrend:",
)


# --- Literature review generation -------------------------------------------

REVIEW_OUTLINE = Prompt(
    name="review_outline",
    system=(
        "You plan the structure of a literature review.\n"
        "Sections must be thematic, not one-section-per-paper. Every section "
        "must be answerable from the supplied papers.\n"
        "Respond with strict JSON only, no prose, no code fences:\n"
        '{"title": "...", "sections": [{"heading": "...", "focus": "...", '
        '"paper_indices": [1, 2]}]}'
    ),
    template="Topic: {topic}\n\nAvailable papers:\n{papers}\n\nOutline:",
)


REVIEW_SECTION = Prompt(
    name="review_section",
    system=(
        "You write one section of a literature review.\n"
        "Rules:\n"
        "- Every factual claim ends with a citation marker like [1] or [2][3].\n"
        "- Synthesize across papers: compare and contrast, do not summarize "
        "each paper in turn.\n"
        "- Name disagreements explicitly where the sources disagree.\n"
        "- If the sources do not cover the focus, say so in one sentence.\n"
        "- Prose only. No headings, no bullet lists."
    ),
    template=(
        "Review topic: {topic}\n"
        "Section: {heading}\n"
        "Focus: {focus}\n\n"
        "Sources:\n{sources}\n\nSection text:"
    ),
)


IDENTIFY_GAPS = Prompt(
    name="identify_gaps",
    system=(
        "You identify research gaps from a body of work.\n"
        "A gap must be grounded in the evidence: something the papers "
        "explicitly call future work, a combination none of them tried, or an "
        "inconsistency none of them resolve. Do not list generic gaps that "
        "would apply to any field.\n"
        "Respond with strict JSON only, no prose, no code fences:\n"
        '{"gaps": [{"gap": "...", "rationale": "...", "evidence_indices": [1], '
        '"suggested_direction": "...", "confidence": 0.0-1.0}]}'
    ),
    template="Research area: {topic}\n\nEvidence:\n{evidence}\n\nGaps:",
)


REGISTRY = {
    prompt.name: prompt
    for prompt in (
        RAG_ANSWER,
        GRADE_RETRIEVAL,
        REWRITE_QUERY,
        VERIFY_ANSWER,
        COMPRESS_CONTEXT,
        EXTRACT_ENTITIES,
        NL_TO_CYPHER,
        COMPARE_CLAIMS,
        EXTRACT_EXPERIMENTS,
        EXTRACT_METHODOLOGY,
        SYNTHESIZE_TREND,
        REVIEW_OUTLINE,
        REVIEW_SECTION,
        IDENTIFY_GAPS,
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
