"""Shared state for the multi-agent research graph.

One typed state object flows through every node. Each agent reads what it needs
and returns only the keys it changed, so LangGraph merges updates rather than
letting agents overwrite each other's work.

The state doubles as an audit trail: every agent appends a `Step` describing
what it did and why, which is what the UI streams as the agent timeline and
what makes a multi-step answer explainable after the fact.
"""

from __future__ import annotations

import operator
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Annotated, TypedDict


class AgentName(str, Enum):
    PLANNER = "planner"
    RETRIEVER = "retriever"
    GRAPH_RETRIEVER = "graph_retriever"
    WEB_SEARCH = "web_search"
    RANKER = "ranker"
    REASONER = "reasoner"
    VERIFIER = "verifier"
    CRITIC = "critic"
    REFLECTOR = "reflector"
    SUMMARIZER = "summarizer"
    CITATION = "citation"
    REPORTER = "reporter"


class Intent(str, Enum):
    """What the user is actually asking for.

    The planner classifies this, and it determines which agents run: a simple
    factual lookup shouldn't pay for report generation, and a literature review
    shouldn't stop at a single retrieval pass.
    """

    QUESTION = "question"
    COMPARE = "compare"
    LITERATURE_REVIEW = "literature_review"
    GAP_ANALYSIS = "gap_analysis"
    SUMMARIZE = "summarize"


@dataclass
class Step:
    """One agent's contribution, recorded for streaming and explainability."""

    agent: str
    summary: str
    detail: str = ""
    duration_ms: int = 0
    started_at: float = field(default_factory=time.time)

    def as_dict(self) -> dict:
        return {
            "agent": self.agent,
            "summary": self.summary,
            "detail": self.detail,
            "duration_ms": self.duration_ms,
        }


@dataclass
class Plan:
    intent: Intent
    sub_questions: list[str] = field(default_factory=list)
    reasoning: str = ""
    needs_web_search: bool = False

    def as_dict(self) -> dict:
        return {
            "intent": self.intent.value,
            "sub_questions": self.sub_questions,
            "reasoning": self.reasoning,
            "needs_web_search": self.needs_web_search,
        }


class ResearchState(TypedDict, total=False):
    """State passed between agents.

    `steps` and `usage_tokens` use `operator.add` reducers so concurrent or
    sequential nodes accumulate rather than clobber. Everything else is
    last-write-wins, which is what we want for values a single agent owns.
    """

    # --- Input ---
    question: str
    user_id: uuid.UUID
    limit: int
    filters: dict

    # --- Planner ---
    plan: Plan | None

    # --- Retrieval ---
    chunks: list
    web_results: list
    graph_paths: list

    # --- Reasoning ---
    draft_answer: str
    final_answer: str
    citations: list[int]

    # --- Quality control ---
    critique: str
    needs_revision: bool
    revision_count: int
    verification: dict | None
    confidence: float

    # --- Bookkeeping ---
    steps: Annotated[list[Step], operator.add]
    usage_tokens: Annotated[int, operator.add]
    errors: Annotated[list[str], operator.add]


def initial_state(
    question: str,
    user_id: uuid.UUID,
    *,
    limit: int = 8,
    filters: dict | None = None,
) -> ResearchState:
    return ResearchState(
        question=question,
        user_id=user_id,
        limit=limit,
        filters=filters or {},
        plan=None,
        chunks=[],
        web_results=[],
        graph_paths=[],
        draft_answer="",
        final_answer="",
        citations=[],
        critique="",
        needs_revision=False,
        revision_count=0,
        verification=None,
        confidence=0.0,
        steps=[],
        usage_tokens=0,
        errors=[],
    )
