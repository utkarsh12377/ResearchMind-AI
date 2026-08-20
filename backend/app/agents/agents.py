"""The individual agents.

Each agent is a plain async function over `ResearchState` that returns only the
keys it changed. Keeping them free of graph wiring means every one can be
tested in isolation, and the graph topology can be rearranged without touching
agent logic.

Every agent records a `Step` and is individually failure-tolerant: an agent
that throws degrades the answer rather than aborting the run, because a partial
answer with a recorded error is more useful than a 500.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable

from app.agents.prompts import CRITIQUE, PLAN, REASON, REPORT, REVISE, SUMMARIZE
from app.agents.state import AgentName, Intent, Plan, ResearchState, Step
from app.agents.tools import SearchTool, get_search_tool, wrap_untrusted
from app.core.logging import get_logger
from app.llm.context import pack_context
from app.llm.gateway import LLMGateway, Message, Role
from app.llm.prompts import format_sources
from app.llm.rag import _parse_json_response, extract_citations, verify_answer
from app.retrieval.service import RetrievalFilters, retrieve

logger = get_logger(__name__)

MAX_REVISIONS = 2
MAX_CONTEXT_TOKENS = 6000


class AgentContext:
    """Dependencies the agents need, injected rather than imported globally.

    This is what lets the whole graph run against a scripted LLM and an
    in-memory index in tests.
    """

    def __init__(
        self,
        db,  # noqa: ANN001
        user,  # noqa: ANN001
        gateway: LLMGateway,
        search_tool: SearchTool | None = None,
        **retrieval_kwargs,
    ) -> None:
        self.db = db
        self.user = user
        self.gateway = gateway
        self._search_tool = search_tool
        self.retrieval_kwargs = retrieval_kwargs

    @property
    def search_tool(self) -> SearchTool:
        if self._search_tool is None:
            self._search_tool = get_search_tool()
        return self._search_tool


def _step(agent: AgentName, summary: str, started: float, detail: str = "") -> Step:
    return Step(
        agent=agent.value,
        summary=summary,
        detail=detail,
        duration_ms=int((time.time() - started) * 1000),
    )


def resilient(agent: AgentName) -> Callable:
    """Wrap an agent so a failure degrades the run instead of aborting it."""

    def decorator(func: Callable[..., Awaitable[dict]]) -> Callable[..., Awaitable[dict]]:
        async def wrapper(state: ResearchState, context: AgentContext) -> dict:
            started = time.time()
            try:
                return await func(state, context)
            except Exception as exc:  # noqa: BLE001 - deliberate: no agent may kill the run
                logger.warning("agent_failed", agent=agent.value, error=str(exc))
                return {
                    "steps": [_step(agent, f"{agent.value} failed", started, str(exc)[:300])],
                    "errors": [f"{agent.value}: {exc}"],
                }

        return wrapper

    return decorator


# --- Planner ---------------------------------------------------------------


@resilient(AgentName.PLANNER)
async def planner_agent(state: ResearchState, context: AgentContext) -> dict:
    """Classify intent and decompose the request into sub-questions."""
    started = time.time()

    completion = await context.gateway.complete(
        [
            Message(Role.SYSTEM, PLAN.system),
            Message(Role.USER, PLAN.render(question=state["question"])),
        ],
        temperature=0.0,
        max_tokens=600,
    )

    data = _parse_json_response(completion.text)
    try:
        intent = Intent(data.get("intent", "question"))
    except ValueError:
        # An unrecognized intent shouldn't fail the run; a plain question is
        # the safe default because it exercises the fewest agents.
        intent = Intent.QUESTION

    sub_questions = [str(q) for q in (data.get("sub_questions") or []) if str(q).strip()]
    plan = Plan(
        intent=intent,
        # Always fall back to the original question so retrieval never runs on
        # an empty query list.
        sub_questions=sub_questions[:4] or [state["question"]],
        reasoning=str(data.get("reasoning", "")),
        needs_web_search=bool(data.get("needs_web_search", False)),
    )

    return {
        "plan": plan,
        "usage_tokens": completion.usage.total_tokens,
        "steps": [
            _step(
                AgentName.PLANNER,
                f"Planned {intent.value} with {len(plan.sub_questions)} sub-question(s)",
                started,
                plan.reasoning,
            )
        ],
    }


# --- Retriever -------------------------------------------------------------


@resilient(AgentName.RETRIEVER)
async def retriever_agent(state: ResearchState, context: AgentContext) -> dict:
    """Retrieve passages for every sub-question and merge the results."""
    started = time.time()
    plan: Plan | None = state.get("plan")
    queries = plan.sub_questions if plan else [state["question"]]

    filters = RetrievalFilters(**state.get("filters", {}))
    seen: set = set()
    merged: list = []

    for query in queries:
        results = await retrieve(
            context.db,
            context.user,
            query,
            limit=state.get("limit", 8),
            filters=filters,
            **context.retrieval_kwargs,
        )
        # Deduplicate across sub-questions: overlapping queries routinely
        # return the same passage, and duplicates waste context budget.
        for chunk in results:
            if chunk.chunk_id not in seen:
                seen.add(chunk.chunk_id)
                merged.append(chunk)

    return {
        "chunks": merged,
        "steps": [
            _step(
                AgentName.RETRIEVER,
                f"Retrieved {len(merged)} unique passages across {len(queries)} quer(ies)",
                started,
            )
        ],
    }


# --- Web search ------------------------------------------------------------


@resilient(AgentName.WEB_SEARCH)
async def web_search_agent(state: ResearchState, context: AgentContext) -> dict:
    """Fetch external context when the planner asked for it.

    Results that trip the injection detector are dropped rather than passed
    along: a page trying to hijack the agent has already disqualified itself as
    evidence, and passing it through would rely on the model to resist it.
    """
    started = time.time()
    plan: Plan | None = state.get("plan")

    if not plan or not plan.needs_web_search:
        return {"steps": [_step(AgentName.WEB_SEARCH, "Web search not required", started)]}

    results = await context.search_tool.search(state["question"])
    safe = [result for result in results if not result.flagged]
    dropped = len(results) - len(safe)

    return {
        "web_results": safe,
        "steps": [
            _step(
                AgentName.WEB_SEARCH,
                f"Fetched {len(safe)} external result(s)",
                started,
                f"{dropped} dropped as suspected prompt injection" if dropped else "",
            )
        ],
    }


# --- Ranker ----------------------------------------------------------------


@resilient(AgentName.RANKER)
async def ranker_agent(state: ResearchState, context: AgentContext) -> dict:
    """Order merged passages and trim them to the context budget."""
    started = time.time()
    chunks = state.get("chunks", [])
    if not chunks:
        return {"steps": [_step(AgentName.RANKER, "No passages to rank", started)]}

    # Passages arrive from several sub-question searches, each ranked only
    # within its own result set, so re-sort globally before packing.
    ordered = sorted(chunks, key=lambda c: (c.rerank_score or c.score), reverse=True)
    packed = pack_context(ordered, state["question"], max_tokens=MAX_CONTEXT_TOKENS)

    detail = ""
    if packed.was_reduced:
        detail = f"{packed.compressed_count} compressed, {packed.dropped_count} dropped"

    return {
        "chunks": packed.chunks,
        "steps": [
            _step(
                AgentName.RANKER,
                f"Ranked and packed {len(packed.chunks)} passages "
                f"(~{packed.used_tokens} tokens)",
                started,
                detail,
            )
        ],
    }


# --- Reasoner --------------------------------------------------------------


@resilient(AgentName.REASONER)
async def reasoner_agent(state: ResearchState, context: AgentContext) -> dict:
    """Synthesize a cited draft answer from the ranked passages."""
    started = time.time()
    chunks = state.get("chunks", [])

    if not chunks:
        return {
            "draft_answer": (
                "I could not find anything in your library that addresses this question."
            ),
            "steps": [_step(AgentName.REASONER, "No sources available to answer", started)],
        }

    plan: Plan | None = state.get("plan")
    sub_questions = "\n".join(f"- {q}" for q in (plan.sub_questions if plan else []))
    # A literature review needs a structured report, not a paragraph answer.
    prompt = REPORT if plan and plan.intent == Intent.LITERATURE_REVIEW else REASON

    sources = format_sources(chunks)
    # External text is appended already wrapped and labeled untrusted, so the
    # model is told explicitly that it is data rather than instruction.
    external = wrap_untrusted(state.get("web_results") or [])
    if external:
        separator = chr(10) * 2
        sources = sources + separator + external

    completion = await context.gateway.complete(
        [
            Message(Role.SYSTEM, prompt.system),
            Message(
                Role.USER,
                prompt.render(
                    question=state["question"],
                    sub_questions=sub_questions,
                    sources=sources,
                ),
            ),
        ],
        max_tokens=3000,
    )

    return {
        "draft_answer": completion.text,
        "usage_tokens": completion.usage.total_tokens,
        "steps": [
            _step(
                AgentName.REASONER,
                f"Drafted an answer from {len(chunks)} sources",
                started,
            )
        ],
    }


# --- Critic ----------------------------------------------------------------


@resilient(AgentName.CRITIC)
async def critic_agent(state: ResearchState, context: AgentContext) -> dict:
    """Check the draft for unsupported claims and unanswered parts."""
    started = time.time()
    draft = state.get("draft_answer", "")
    chunks = state.get("chunks", [])

    if not draft or not chunks:
        return {
            "needs_revision": False,
            "steps": [_step(AgentName.CRITIC, "Nothing to critique", started)],
        }

    completion = await context.gateway.complete(
        [
            Message(Role.SYSTEM, CRITIQUE.system),
            Message(
                Role.USER,
                CRITIQUE.render(
                    question=state["question"],
                    answer=draft,
                    sources=format_sources(chunks),
                ),
            ),
        ],
        temperature=0.0,
        max_tokens=600,
    )

    data = _parse_json_response(completion.text)
    issues = [str(issue) for issue in (data.get("issues") or [])]
    # An unparseable critique is treated as "no revision needed": looping on a
    # critic we can't read would burn tokens without improving anything.
    needs_revision = bool(data.get("needs_revision", False)) and bool(issues)

    if state.get("revision_count", 0) >= MAX_REVISIONS:
        needs_revision = False

    return {
        "critique": "\n".join(f"- {issue}" for issue in issues),
        "needs_revision": needs_revision,
        "usage_tokens": completion.usage.total_tokens,
        "steps": [
            _step(
                AgentName.CRITIC,
                f"Found {len(issues)} issue(s)" if issues else "No issues found",
                started,
                str(data.get("reason", "")),
            )
        ],
    }


# --- Reflector -------------------------------------------------------------


@resilient(AgentName.REFLECTOR)
async def reflector_agent(state: ResearchState, context: AgentContext) -> dict:
    """Revise the draft to address the critique."""
    started = time.time()

    completion = await context.gateway.complete(
        [
            Message(Role.SYSTEM, REVISE.system),
            Message(
                Role.USER,
                REVISE.render(
                    question=state["question"],
                    answer=state.get("draft_answer", ""),
                    critique=state.get("critique", ""),
                    sources=format_sources(state.get("chunks", [])),
                ),
            ),
        ],
        max_tokens=3000,
    )

    revised = completion.text.strip()
    return {
        # Keep the previous draft if the reviser returned nothing usable.
        "draft_answer": revised or state.get("draft_answer", ""),
        "revision_count": state.get("revision_count", 0) + 1,
        "usage_tokens": completion.usage.total_tokens,
        "steps": [
            _step(
                AgentName.REFLECTOR,
                f"Revised the answer (pass {state.get('revision_count', 0) + 1})",
                started,
            )
        ],
    }


# --- Verifier --------------------------------------------------------------


@resilient(AgentName.VERIFIER)
async def verifier_agent(state: ResearchState, context: AgentContext) -> dict:
    """Check the final answer against its sources and score confidence."""
    started = time.time()
    answer = state.get("draft_answer", "")
    chunks = state.get("chunks", [])

    if not answer or not chunks:
        return {
            "confidence": 0.0,
            "steps": [_step(AgentName.VERIFIER, "Nothing to verify", started)],
        }

    verification, usage = await verify_answer(
        context.gateway, state["question"], answer, chunks
    )

    confidence = verification.confidence if verification.supported else min(
        verification.confidence, 0.35
    )

    return {
        "verification": {
            "supported": verification.supported,
            "confidence": verification.confidence,
            "unsupported_claims": verification.unsupported_claims,
            "reason": verification.reason,
        },
        "confidence": round(confidence, 2),
        "usage_tokens": usage.total_tokens,
        "steps": [
            _step(
                AgentName.VERIFIER,
                "Answer is supported by sources"
                if verification.supported
                else f"Flagged {len(verification.unsupported_claims)} unsupported claim(s)",
                started,
                verification.reason,
            )
        ],
    }


# --- Citation --------------------------------------------------------------


@resilient(AgentName.CITATION)
async def citation_agent(state: ResearchState, context: AgentContext) -> dict:
    """Resolve citation markers to sources and finalize the answer."""
    started = time.time()
    answer = state.get("draft_answer", "")
    chunks = state.get("chunks", [])

    citations = extract_citations(answer, len(chunks))

    return {
        "final_answer": answer,
        "citations": citations,
        "steps": [
            _step(
                AgentName.CITATION,
                f"Resolved {len(citations)} citation(s)",
                started,
                # Surfacing this matters: an uncited answer over available
                # sources is a quality signal the UI should show.
                "" if citations or not chunks else "Answer cites no sources",
            )
        ],
    }


# --- Summarizer ------------------------------------------------------------


@resilient(AgentName.SUMMARIZER)
async def summarizer_agent(state: ResearchState, context: AgentContext) -> dict:
    """Prepend an executive summary to long reports."""
    started = time.time()
    answer = state.get("final_answer") or state.get("draft_answer", "")

    if len(answer) < 1200:
        return {"steps": [_step(AgentName.SUMMARIZER, "Answer short enough to skip", started)]}

    completion = await context.gateway.complete(
        [
            Message(Role.SYSTEM, SUMMARIZE.system),
            Message(Role.USER, SUMMARIZE.render(question=state["question"], answer=answer)),
        ],
        max_tokens=400,
    )

    summary = completion.text.strip()
    if not summary:
        return {"steps": [_step(AgentName.SUMMARIZER, "No summary produced", started)]}

    return {
        "final_answer": f"**Summary.** {summary}\n\n{answer}",
        "usage_tokens": completion.usage.total_tokens,
        "steps": [_step(AgentName.SUMMARIZER, "Added an executive summary", started)],
    }


def serialize_plan(plan: Plan | None) -> dict | None:
    return plan.as_dict() if plan else None


def serialize_state(state: ResearchState) -> dict:
    """Render the graph result as an API-friendly payload."""
    chunks = state.get("chunks", [])
    return {
        "question": state.get("question", ""),
        "answer": state.get("final_answer") or state.get("draft_answer", ""),
        "plan": serialize_plan(state.get("plan")),
        "citations": state.get("citations", []),
        "confidence": state.get("confidence", 0.0),
        "verification": state.get("verification"),
        "revision_count": state.get("revision_count", 0),
        "usage_tokens": state.get("usage_tokens", 0),
        "errors": state.get("errors", []),
        "steps": [step.as_dict() for step in state.get("steps", [])],
        "sources": [
            {
                "index": index,
                "chunk_id": str(chunk.chunk_id),
                "paper_id": str(chunk.paper_id),
                "paper_title": chunk.paper_title,
                "citation": chunk.citation,
                "page_number": chunk.page_number,
                "content": chunk.content,
            }
            for index, chunk in enumerate(chunks, start=1)
        ],
    }


__all__ = [
    "AgentContext",
    "citation_agent",
    "critic_agent",
    "planner_agent",
    "ranker_agent",
    "reasoner_agent",
    "web_search_agent",
    "reflector_agent",
    "serialize_state",
    "summarizer_agent",
    "verifier_agent",
]
