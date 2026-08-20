"""Natural language to Cypher, with a validator standing between the two.

Letting a model write database queries is only acceptable if something other
than the model decides what is allowed to run. The generator here is ordinary;
the validator is the actual feature. It is a whitelist — an allowed set of
clauses, labels, and relationship types — because a blacklist of dangerous
keywords is defeated by the first spelling you did not think of.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.core.config import get_settings
from app.core.logging import get_logger
from app.graph.schema import ALLOWED_EDGES, NodeLabel
from app.graph.store import GraphStore, get_graph_store
from app.llm.gateway import LLMGateway, Message, Role
from app.llm.prompts import NL_TO_CYPHER

logger = get_logger(__name__)

ALLOWED_LABELS = frozenset(label.value for label in NodeLabel)
ALLOWED_RELATIONSHIPS = frozenset(ALLOWED_EDGES)

FORBIDDEN_KEYWORDS = (
    "create", "merge", "set", "delete", "detach", "remove", "drop",
    "foreach", "call", "load", "using", "periodic", "commit",
    "apoc", "dbms", "db.", "terminate", "grant", "revoke", "alter",
)

LABEL_PATTERN = re.compile(r":\s*([A-Za-z_][A-Za-z0-9_]*)")
REL_PATTERN = re.compile(r"\[\s*\w*\s*:\s*([A-Za-z_|`\s]+?)\s*(?:\*[\d.]*)?\s*\]")
LIMIT_PATTERN = re.compile(r"\blimit\s+(\d+)\s*$", re.IGNORECASE)


class UnsafeCypherError(ValueError):
    """Raised when a generated query fails validation."""


@dataclass
class GraphQueryResult:
    question: str
    cypher: str
    rows: list[dict] = field(default_factory=list)
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.error is None


def schema_description() -> str:
    """The schema the generator is allowed to reference."""
    lines = ["Nodes:"]
    for label in NodeLabel:
        extra = (
            " {key, name, paper_id, workspace_id, year}"
            if label is NodeLabel.PAPER
            else " {key, name}"
        )
        lines.append(f"  (:{label.value}{extra})")

    lines.append("Relationships:")
    for rel_type, (start, end) in sorted(ALLOWED_EDGES.items()):
        lines.append(f"  (:{start})-[:{rel_type}]->(:{end})")
    return "\n".join(lines)


def validate_cypher(cypher: str, *, max_limit: int | None = None) -> str:
    """Return a safe, normalized query or raise UnsafeCypherError."""
    settings = get_settings()
    max_limit = max_limit or settings.graph_max_query_limit

    query = _strip_fences(cypher).strip().rstrip(";").strip()
    if not query:
        raise UnsafeCypherError("Empty query")

    if ";" in query:
        raise UnsafeCypherError("Multiple statements are not allowed")

    scrubbed = _strip_string_literals(query)
    lowered = scrubbed.lower()

    for keyword in FORBIDDEN_KEYWORDS:
        if re.search(rf"(?<![\w.]){re.escape(keyword)}", lowered):
            raise UnsafeCypherError(f"Disallowed keyword: {keyword}")

    if not re.match(r"^\s*(match|optional\s+match|with|unwind)\b", lowered):
        raise UnsafeCypherError("Query must start with MATCH, OPTIONAL MATCH, WITH, or UNWIND")

    if " return " not in f" {lowered} ":
        raise UnsafeCypherError("Query must RETURN something")

    unknown_labels = {
        label for label in LABEL_PATTERN.findall(scrubbed) if label not in ALLOWED_LABELS
    } - _relationship_tokens(scrubbed)
    if unknown_labels:
        raise UnsafeCypherError(f"Unknown label(s): {', '.join(sorted(unknown_labels))}")

    unknown_rels = _relationship_tokens(scrubbed) - ALLOWED_RELATIONSHIPS
    if unknown_rels:
        raise UnsafeCypherError(f"Unknown relationship(s): {', '.join(sorted(unknown_rels))}")

    return _enforce_limit(query, max_limit)


def _relationship_tokens(query: str) -> set[str]:
    tokens: set[str] = set()
    for match in REL_PATTERN.findall(query):
        for part in match.replace("`", "").split("|"):
            token = part.strip()
            if token:
                tokens.add(token)
    return tokens


def _enforce_limit(query: str, max_limit: int) -> str:
    match = LIMIT_PATTERN.search(query)
    if match is None:
        return f"{query} LIMIT {max_limit}"

    requested = int(match.group(1))
    if requested <= max_limit:
        return query
    return LIMIT_PATTERN.sub(f"LIMIT {max_limit}", query)


def _strip_fences(text: str) -> str:
    if "```" in text:
        return re.sub(r"```(?:cypher|sql)?", "", text)
    return text


def _strip_string_literals(query: str) -> str:
    """Blank out quoted strings so their contents can't trip keyword checks.

    A paper legitimately titled "Deleting Noisy Labels" should not be read as a
    DELETE, and equally a keyword hidden inside quotes cannot execute.
    """
    return re.sub(r"'[^']*'|\"[^\"]*\"", "''", query)


async def generate_cypher(question: str, *, gateway: LLMGateway) -> str:
    completion = await gateway.complete(
        [
            Message(Role.SYSTEM, NL_TO_CYPHER.system),
            Message(
                Role.USER,
                NL_TO_CYPHER.render(schema=schema_description(), question=question),
            ),
        ],
        temperature=0.0,
        max_tokens=400,
    )
    return completion.text.strip()


async def answer_graph_question(
    question: str,
    *,
    gateway: LLMGateway,
    store: GraphStore | None = None,
) -> GraphQueryResult:
    store = store or get_graph_store()

    try:
        raw = await generate_cypher(question, gateway=gateway)
        cypher = validate_cypher(raw)
    except UnsafeCypherError as exc:
        logger.warning("cypher_rejected", question=question[:120], reason=str(exc))
        return GraphQueryResult(question=question, cypher="", error=str(exc))
    except Exception as exc:  # noqa: BLE001
        logger.warning("cypher_generation_failed", error=str(exc))
        return GraphQueryResult(question=question, cypher="", error="Could not generate a query")

    try:
        rows = await store.run_read(cypher)
    except NotImplementedError as exc:
        return GraphQueryResult(question=question, cypher=cypher, error=str(exc))
    except Exception as exc:  # noqa: BLE001
        logger.warning("cypher_execution_failed", cypher=cypher, error=str(exc))
        return GraphQueryResult(question=question, cypher=cypher, error="Query execution failed")

    logger.info("graph_question_answered", rows=len(rows))
    return GraphQueryResult(question=question, cypher=cypher, rows=rows)
