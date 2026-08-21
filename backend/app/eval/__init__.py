"""Offline evaluation of the retrieval and answering stack."""

from app.eval.dataset import EvalCase, load_cases
from app.eval.metrics import MetricScores, score_case
from app.eval.runner import EvalReport, Thresholds, run_benchmark

__all__ = [
    "EvalCase",
    "EvalReport",
    "MetricScores",
    "Thresholds",
    "load_cases",
    "run_benchmark",
    "score_case",
]
