"""Reading JSON back out of a language model's response.

Models asked for strict JSON still wrap it in code fences, prefix it with "Here
is the JSON:", or trail an explanation after the closing brace. Every structured
extractor in the system hits the same three failure modes, so the salvage logic
lives here once instead of being re-improvised per call site.
"""

from __future__ import annotations

import json
import re

from app.core.logging import get_logger

logger = get_logger(__name__)

FENCE_PATTERN = re.compile(r"```(?:json)?", re.IGNORECASE)


def load_json_object(raw: str) -> dict | None:
    """Best-effort parse of a JSON object, or None if nothing usable is there."""
    text = FENCE_PATTERN.sub("", raw or "").strip().strip("`").strip()
    if not text:
        return None

    parsed = _try_parse(text)
    if parsed is None:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            logger.warning("json_output_unparseable", preview=text[:120])
            return None
        parsed = _try_parse(text[start : end + 1])

    if parsed is None:
        logger.warning("json_output_unparseable", preview=text[:120])
        return None
    return parsed if isinstance(parsed, dict) else None


def _try_parse(text: str) -> object | None:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def clamp_confidence(value: object, default: float = 0.5) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default
