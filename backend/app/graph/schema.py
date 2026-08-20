"""Knowledge graph schema: the node labels and relationship types we allow.

Everything that reads or writes the graph goes through these constants. Keeping
the vocabulary closed is what makes the NL-to-Cypher layer safe to expose: a
generated query naming a label that isn't here is rejected before it reaches the
database.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class NodeLabel(str, Enum):
    PAPER = "Paper"
    AUTHOR = "Author"
    INSTITUTION = "Institution"
    DATASET = "Dataset"
    MODEL = "Model"
    TASK = "Task"
    METRIC = "Metric"
    METHOD = "Method"
    BENCHMARK = "Benchmark"
    VENUE = "Venue"


class RelationType(str, Enum):
    AUTHORED_BY = "AUTHORED_BY"
    AFFILIATED_WITH = "AFFILIATED_WITH"
    CITES = "CITES"
    EVALUATES_ON = "EVALUATES_ON"
    USES_DATASET = "USES_DATASET"
    PROPOSES = "PROPOSES"
    USES_METHOD = "USES_METHOD"
    REPORTS = "REPORTS"
    ADDRESSES = "ADDRESSES"
    PUBLISHED_IN = "PUBLISHED_IN"
    COMPARES_TO = "COMPARES_TO"


ENTITY_LABELS = frozenset(
    label.value
    for label in NodeLabel
    if label is not NodeLabel.PAPER
)

#: Which relationships are legal between which labels. Extraction output that
#: doesn't fit one of these shapes is dropped rather than written, so a
#: hallucinated triple can't quietly corrupt the graph.
ALLOWED_EDGES: dict[str, tuple[str, str]] = {
    RelationType.AUTHORED_BY.value: (NodeLabel.PAPER.value, NodeLabel.AUTHOR.value),
    RelationType.AFFILIATED_WITH.value: (NodeLabel.AUTHOR.value, NodeLabel.INSTITUTION.value),
    RelationType.CITES.value: (NodeLabel.PAPER.value, NodeLabel.PAPER.value),
    RelationType.EVALUATES_ON.value: (NodeLabel.PAPER.value, NodeLabel.BENCHMARK.value),
    RelationType.USES_DATASET.value: (NodeLabel.PAPER.value, NodeLabel.DATASET.value),
    RelationType.PROPOSES.value: (NodeLabel.PAPER.value, NodeLabel.MODEL.value),
    RelationType.USES_METHOD.value: (NodeLabel.PAPER.value, NodeLabel.METHOD.value),
    RelationType.REPORTS.value: (NodeLabel.PAPER.value, NodeLabel.METRIC.value),
    RelationType.ADDRESSES.value: (NodeLabel.PAPER.value, NodeLabel.TASK.value),
    RelationType.PUBLISHED_IN.value: (NodeLabel.PAPER.value, NodeLabel.VENUE.value),
    RelationType.COMPARES_TO.value: (NodeLabel.MODEL.value, NodeLabel.MODEL.value),
}


@dataclass
class GraphNode:
    label: str
    key: str
    properties: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.label not in {item.value for item in NodeLabel}:
            raise ValueError(f"Unknown node label: {self.label!r}")
        self.key = normalize_key(self.key)
        if not self.key:
            raise ValueError("Graph nodes need a non-empty key")

    @property
    def uid(self) -> str:
        return f"{self.label}:{self.key}"


@dataclass
class GraphRelationship:
    type: str
    start: GraphNode
    end: GraphNode
    properties: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        expected = ALLOWED_EDGES.get(self.type)
        if expected is None:
            raise ValueError(f"Unknown relationship type: {self.type!r}")
        if (self.start.label, self.end.label) != expected:
            raise ValueError(
                f"{self.type} must connect {expected[0]} -> {expected[1]}, "
                f"got {self.start.label} -> {self.end.label}"
            )

    @property
    def uid(self) -> str:
        return f"{self.start.uid}-[{self.type}]->{self.end.uid}"


def normalize_key(value: str) -> str:
    """Collapse a surface form into a stable node key.

    Entity mentions vary across papers ("BERT-Base", "bert base", "BERT  Base")
    and each variant would otherwise become its own node, fragmenting exactly
    the connections the graph exists to make.
    """
    lowered = " ".join(str(value).lower().split())
    return lowered.strip(" .,;:()[]{}\"'")


CONSTRAINTS = tuple(
    f"CREATE CONSTRAINT {label.value.lower()}_key IF NOT EXISTS "
    f"FOR (n:{label.value}) REQUIRE n.key IS UNIQUE"
    for label in NodeLabel
)

INDEXES = (
    "CREATE INDEX paper_workspace IF NOT EXISTS FOR (n:Paper) ON (n.workspace_id)",
    "CREATE INDEX paper_year IF NOT EXISTS FOR (n:Paper) ON (n.year)",
)
