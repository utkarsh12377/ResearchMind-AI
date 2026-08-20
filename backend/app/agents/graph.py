"""The research graph.

Topology:

    planner -> retriever -> ranker -> reasoner -> critic
                                                    |
                                    needs_revision? |
                                       yes -> reflector -> critic  (bounded)
                                        no -> verifier -> citation -> summarizer

The critic/reflector cycle is the reason this is a graph rather than a
pipeline: revision is conditional and repeatable, which a linear chain cannot
express. The loop is bounded by MAX_REVISIONS inside the critic, because a
critic and a reviser left to argue will happily burn tokens forever.

LangGraph owns state merging and edge routing; the agents stay plain functions,
so the topology can be rearranged without touching agent logic.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from functools import partial

from langgraph.graph import END, StateGraph

from app.agents.agents import (
    AgentContext,
    citation_agent,
    critic_agent,
    planner_agent,
    ranker_agent,
    reasoner_agent,
    reflector_agent,
    retriever_agent,
    serialize_state,
    summarizer_agent,
    verifier_agent,
)
from app.agents.state import ResearchState, initial_state
from app.core.logging import get_logger
from app.llm.gateway import LLMGateway, get_gateway

logger = get_logger(__name__)


def _route_after_critic(state: ResearchState) -> str:
    """Send a flagged draft back for revision, otherwise move to verification."""
    return "reflector" if state.get("needs_revision") else "verifier"


def build_graph(context: AgentContext):  # noqa: ANN201 - LangGraph's compiled type
    """Wire the agents into a compiled LangGraph.

    The context is bound into each node with partial() rather than smuggled
    through state: dependencies aren't data, and keeping them out of state
    means the state stays serializable for checkpointing and streaming.
    """
    graph = StateGraph(ResearchState)

    graph.add_node("planner", partial(planner_agent, context=context))
    graph.add_node("retriever", partial(retriever_agent, context=context))
    graph.add_node("ranker", partial(ranker_agent, context=context))
    graph.add_node("reasoner", partial(reasoner_agent, context=context))
    graph.add_node("critic", partial(critic_agent, context=context))
    graph.add_node("reflector", partial(reflector_agent, context=context))
    graph.add_node("verifier", partial(verifier_agent, context=context))
    graph.add_node("citation", partial(citation_agent, context=context))
    graph.add_node("summarizer", partial(summarizer_agent, context=context))

    graph.set_entry_point("planner")
    graph.add_edge("planner", "retriever")
    graph.add_edge("retriever", "ranker")
    graph.add_edge("ranker", "reasoner")
    graph.add_edge("reasoner", "critic")

    graph.add_conditional_edges(
        "critic",
        _route_after_critic,
        {"reflector": "reflector", "verifier": "verifier"},
    )
    # Back to the critic so a revision is re-checked rather than trusted.
    graph.add_edge("reflector", "critic")

    graph.add_edge("verifier", "citation")
    graph.add_edge("citation", "summarizer")
    graph.add_edge("summarizer", END)

    return graph.compile()



async def run_research(
    db,  # noqa: ANN001
    user,  # noqa: ANN001
    question: str,
    *,
    limit: int = 8,
    filters: dict | None = None,
    gateway: LLMGateway | None = None,
    **retrieval_kwargs,
) -> dict:
    """Run the full agent graph and return a serialized result."""
    context = AgentContext(db, user, gateway or get_gateway(), **retrieval_kwargs)
    graph = build_graph(context)

    state = initial_state(question, _user_id(user), limit=limit, filters=filters)
    final = await graph.ainvoke(state)

    logger.info(
        "research_completed",
        steps=len(final.get("steps", [])),
        revisions=final.get("revision_count", 0),
        confidence=final.get("confidence", 0.0),
        tokens=final.get("usage_tokens", 0),
        errors=len(final.get("errors", [])),
    )
    return serialize_state(final)


async def stream_research(
    db,  # noqa: ANN001
    user,  # noqa: ANN001
    question: str,
    *,
    limit: int = 8,
    filters: dict | None = None,
    gateway: LLMGateway | None = None,
    **retrieval_kwargs,
) -> AsyncIterator[dict]:
    """Stream agent progress as each node completes.

    Multi-agent answers take long enough that a silent wait is a bad
    experience; emitting each agent's step as it finishes turns the latency
    into visible progress and makes the reasoning auditable live.
    """
    context = AgentContext(db, user, gateway or get_gateway(), **retrieval_kwargs)
    graph = build_graph(context)
    state = initial_state(question, _user_id(user), limit=limit, filters=filters)

    # stream_mode="values" yields the full accumulated state after each node,
    # rather than that node's delta, so the final payload is complete and new
    # steps can be diffed against what has already been sent.
    emitted = 0
    latest: dict = dict(state)

    async for snapshot in graph.astream(state, stream_mode="values"):
        latest = snapshot
        steps = snapshot.get("steps") or []
        for step in steps[emitted:]:
            yield {"type": "step", **step.as_dict()}
        emitted = len(steps)

    yield {"type": "result", **serialize_state(latest)}


def _user_id(user) -> uuid.UUID:  # noqa: ANN001
    return getattr(user, "id", uuid.uuid4())
