"""Benchmark cases the evaluation harness runs against.

Cases are plain data so they can live in a JSON file that a domain expert edits
without touching Python. The built-in set covers the behaviours that regress
quietly -- refusing to answer without evidence, keeping citations in range,
handling a question the corpus does not cover -- rather than trying to measure
answer quality, which needs a real corpus and a real model.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class EvalCase:
    id: str
    question: str
    ground_truth: str = ""
    expected_paper_titles: list[str] = field(default_factory=list)
    expect_refusal: bool = False
    #: Behaviour the offline echo provider cannot produce. Skipped rather
    #: than failed when no real model is configured, so the gate stays
    #: meaningful instead of being permanently red.
    requires_llm: bool = False
    tags: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, payload: dict) -> EvalCase:
        if not payload.get("id") or not payload.get("question"):
            raise ValueError("An eval case needs both an id and a question")
        return cls(
            id=str(payload["id"]),
            question=str(payload["question"]),
            ground_truth=str(payload.get("ground_truth", "")),
            expected_paper_titles=[str(t) for t in payload.get("expected_paper_titles", [])],
            expect_refusal=bool(payload.get("expect_refusal", False)),
            requires_llm=bool(payload.get("requires_llm", False)),
            tags=[str(t) for t in payload.get("tags", [])],
        )

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "question": self.question,
            "ground_truth": self.ground_truth,
            "expected_paper_titles": self.expected_paper_titles,
            "expect_refusal": self.expect_refusal,
            "requires_llm": self.requires_llm,
            "tags": self.tags,
        }


DEFAULT_CASES = (
    EvalCase(
        id="grounded-lookup",
        question="What retrieval approach does the paper describe?",
        ground_truth="A hybrid retrieval system combining dense and sparse matching.",
        tags=["retrieval", "grounded"],
    ),
    EvalCase(
        id="metric-lookup",
        question="What accuracy does the reported system achieve?",
        ground_truth="The reported accuracy figure from the paper.",
        tags=["extraction"],
    ),
    EvalCase(
        id="out-of-corpus",
        question="What is the melting point of tungsten carbide?",
        expect_refusal=True,
        requires_llm=True,
        tags=["refusal", "safety"],
    ),
    EvalCase(
        id="unanswerable-in-scope",
        question="What was the authors' funding source?",
        expect_refusal=True,
        requires_llm=True,
        tags=["refusal"],
    ),
)


def load_cases(path: str | Path | None = None) -> list[EvalCase]:
    """Load cases from JSON, or return the built-in set."""
    if path is None:
        return list(DEFAULT_CASES)

    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"No eval dataset at {file_path}")

    payload = json.loads(file_path.read_text(encoding="utf-8"))
    raw_cases = payload.get("cases", payload) if isinstance(payload, dict) else payload
    if not isinstance(raw_cases, list):
        raise ValueError("An eval dataset must be a list of cases")

    cases = [EvalCase.from_dict(item) for item in raw_cases]
    _reject_duplicates(cases)
    return cases


def _reject_duplicates(cases: list[EvalCase]) -> None:
    seen: set[str] = set()
    for case in cases:
        if case.id in seen:
            raise ValueError(f"Duplicate eval case id: {case.id!r}")
        seen.add(case.id)


def save_cases(cases: list[EvalCase], path: str | Path) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(
        json.dumps({"cases": [case.as_dict() for case in cases]}, indent=2),
        encoding="utf-8",
    )


def filter_by_tag(cases: list[EvalCase], tag: str) -> list[EvalCase]:
    return [case for case in cases if tag in case.tags]
