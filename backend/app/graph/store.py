"""Graph storage backends.

Neo4j is the production target. The in-memory backend exists so the graph
features are runnable and testable without a database container — the same
reason the vector store ships a memory backend — and it implements the identical
interface so nothing above this layer knows which one it is talking to.
"""

from __future__ import annotations

import asyncio
import json
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from app.core.config import get_settings
from app.core.logging import get_logger
from app.graph.schema import CONSTRAINTS, INDEXES, GraphNode, GraphRelationship

logger = get_logger(__name__)


@dataclass
class NeighborHit:
    node: GraphNode
    relationship: str
    direction: str
    distance: int


class GraphStore(ABC):
    name: str = "graph"

    @abstractmethod
    async def upsert_nodes(self, nodes: list[GraphNode]) -> int: ...

    @abstractmethod
    async def upsert_relationships(self, relationships: list[GraphRelationship]) -> int: ...

    @abstractmethod
    async def neighbors(
        self,
        label: str,
        key: str,
        *,
        depth: int = 1,
        relationship_types: list[str] | None = None,
        limit: int = 50,
    ) -> list[NeighborHit]: ...

    @abstractmethod
    async def find_nodes(
        self, label: str, *, contains: str | None = None, limit: int = 50
    ) -> list[GraphNode]: ...

    @abstractmethod
    async def run_read(self, cypher: str, params: dict | None = None) -> list[dict]: ...

    @abstractmethod
    async def delete_paper(self, paper_id: str) -> int: ...

    @abstractmethod
    async def counts(self) -> dict[str, int]: ...

    async def ensure_schema(self) -> None:
        return None


class InMemoryGraphStore(GraphStore):
    """Adjacency-list graph with optional JSON persistence."""

    name = "memory"

    def __init__(self, persist_path: str | Path | None = None) -> None:
        self._nodes: dict[str, GraphNode] = {}
        self._edges: dict[str, GraphRelationship] = {}
        self._out: dict[str, set[str]] = defaultdict(set)
        self._in: dict[str, set[str]] = defaultdict(set)
        self._lock = asyncio.Lock()
        self._persist_path = Path(persist_path) if persist_path else None
        if self._persist_path and self._persist_path.exists():
            self._load()

    def _load(self) -> None:
        try:
            payload = json.loads(self._persist_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("graph_index_unreadable", path=str(self._persist_path), error=str(exc))
            return

        for raw in payload.get("nodes", []):
            node = GraphNode(raw["label"], raw["key"], raw.get("properties", {}))
            self._nodes[node.uid] = node
        for raw in payload.get("edges", []):
            start = self._nodes.get(raw["start"])
            end = self._nodes.get(raw["end"])
            if start is None or end is None:
                continue
            edge = GraphRelationship(raw["type"], start, end, raw.get("properties", {}))
            self._register_edge(edge)

    def _flush(self) -> None:
        if self._persist_path is None:
            return
        self._persist_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "nodes": [
                {"label": n.label, "key": n.key, "properties": n.properties}
                for n in self._nodes.values()
            ],
            "edges": [
                {
                    "type": e.type,
                    "start": e.start.uid,
                    "end": e.end.uid,
                    "properties": e.properties,
                }
                for e in self._edges.values()
            ],
        }
        tmp = self._persist_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        tmp.replace(self._persist_path)

    def _register_edge(self, edge: GraphRelationship) -> None:
        self._edges[edge.uid] = edge
        self._out[edge.start.uid].add(edge.uid)
        self._in[edge.end.uid].add(edge.uid)

    async def upsert_nodes(self, nodes: list[GraphNode]) -> int:
        if not nodes:
            return 0
        async with self._lock:
            for node in nodes:
                existing = self._nodes.get(node.uid)
                if existing is None:
                    self._nodes[node.uid] = node
                else:
                    existing.properties.update(node.properties)
            self._flush()
        return len(nodes)

    async def upsert_relationships(self, relationships: list[GraphRelationship]) -> int:
        if not relationships:
            return 0
        async with self._lock:
            for edge in relationships:
                self._nodes.setdefault(edge.start.uid, edge.start)
                self._nodes.setdefault(edge.end.uid, edge.end)
                self._register_edge(edge)
            self._flush()
        return len(relationships)

    async def neighbors(
        self,
        label: str,
        key: str,
        *,
        depth: int = 1,
        relationship_types: list[str] | None = None,
        limit: int = 50,
    ) -> list[NeighborHit]:
        from app.graph.schema import normalize_key

        start_uid = f"{label}:{normalize_key(key)}"
        if start_uid not in self._nodes:
            return []

        wanted = set(relationship_types or [])
        seen = {start_uid}
        frontier = [start_uid]
        hits: list[NeighborHit] = []

        for distance in range(1, max(depth, 1) + 1):
            next_frontier: list[str] = []
            for uid in frontier:
                for edge_uid in self._out.get(uid, set()):
                    edge = self._edges[edge_uid]
                    if wanted and edge.type not in wanted:
                        continue
                    if edge.end.uid in seen:
                        continue
                    seen.add(edge.end.uid)
                    next_frontier.append(edge.end.uid)
                    hits.append(NeighborHit(edge.end, edge.type, "out", distance))
                for edge_uid in self._in.get(uid, set()):
                    edge = self._edges[edge_uid]
                    if wanted and edge.type not in wanted:
                        continue
                    if edge.start.uid in seen:
                        continue
                    seen.add(edge.start.uid)
                    next_frontier.append(edge.start.uid)
                    hits.append(NeighborHit(edge.start, edge.type, "in", distance))
            frontier = next_frontier
            if not frontier:
                break

        return hits[:limit]

    async def find_nodes(
        self, label: str, *, contains: str | None = None, limit: int = 50
    ) -> list[GraphNode]:
        needle = (contains or "").lower().strip()
        matches = [
            node
            for node in self._nodes.values()
            if node.label == label and (not needle or needle in node.key)
        ]
        matches.sort(key=lambda n: n.key)
        return matches[:limit]

    async def run_read(self, cypher: str, params: dict | None = None) -> list[dict]:
        raise NotImplementedError(
            "The in-memory graph does not execute Cypher. Configure Neo4j "
            "(GRAPH_STORE_BACKEND=neo4j) to use natural-language graph queries."
        )

    async def delete_paper(self, paper_id: str) -> int:
        uid = f"Paper:{paper_id.lower()}"
        async with self._lock:
            if uid not in self._nodes:
                return 0
            attached = self._out.pop(uid, set()) | self._in.pop(uid, set())
            for edge_uid in attached:
                edge = self._edges.pop(edge_uid, None)
                if edge is None:
                    continue
                self._out[edge.start.uid].discard(edge_uid)
                self._in[edge.end.uid].discard(edge_uid)
            del self._nodes[uid]
            self._flush()
        return len(attached) + 1

    async def counts(self) -> dict[str, int]:
        by_label: dict[str, int] = defaultdict(int)
        for node in self._nodes.values():
            by_label[node.label] += 1
        by_label["_relationships"] = len(self._edges)
        return dict(by_label)


class Neo4jGraphStore(GraphStore):
    """Neo4j-backed store using the official async driver."""

    name = "neo4j"

    def __init__(self, uri: str, user: str, password: str) -> None:
        try:
            from neo4j import AsyncGraphDatabase
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise RuntimeError(
                "GRAPH_STORE_BACKEND=neo4j requires the 'neo4j' package. "
                "Install it with: pip install neo4j"
            ) from exc

        self._driver = AsyncGraphDatabase.driver(uri, auth=(user, password))

    async def ensure_schema(self) -> None:
        async with self._driver.session() as session:
            for statement in (*CONSTRAINTS, *INDEXES):
                await session.run(statement)

    async def upsert_nodes(self, nodes: list[GraphNode]) -> int:
        if not nodes:
            return 0
        grouped: dict[str, list[dict]] = defaultdict(list)
        for node in nodes:
            grouped[node.label].append({"key": node.key, "props": node.properties})

        async with self._driver.session() as session:
            for label, rows in grouped.items():
                await session.run(
                    f"UNWIND $rows AS row MERGE (n:{label} {{key: row.key}}) "
                    "SET n += row.props",
                    rows=rows,
                )
        return len(nodes)

    async def upsert_relationships(self, relationships: list[GraphRelationship]) -> int:
        if not relationships:
            return 0
        grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
        for edge in relationships:
            signature = (edge.start.label, edge.type, edge.end.label)
            grouped[signature].append(
                {"start": edge.start.key, "end": edge.end.key, "props": edge.properties}
            )

        async with self._driver.session() as session:
            for (start_label, rel_type, end_label), rows in grouped.items():
                await session.run(
                    "UNWIND $rows AS row "
                    f"MATCH (a:{start_label} {{key: row.start}}) "
                    f"MATCH (b:{end_label} {{key: row.end}}) "
                    f"MERGE (a)-[r:{rel_type}]->(b) "
                    "SET r += row.props",
                    rows=rows,
                )
        return len(relationships)

    async def neighbors(
        self,
        label: str,
        key: str,
        *,
        depth: int = 1,
        relationship_types: list[str] | None = None,
        limit: int = 50,
    ) -> list[NeighborHit]:
        from app.graph.schema import normalize_key

        rel_filter = ""
        if relationship_types:
            rel_filter = ":" + "|".join(relationship_types)

        cypher = (
            f"MATCH path = (start:{label} {{key: $key}})-[{rel_filter}*1..{max(depth, 1)}]-(other) "
            "RETURN other, labels(other) AS labels, length(path) AS distance, "
            "type(last(relationships(path))) AS rel LIMIT $limit"
        )
        rows = await self.run_read(cypher, {"key": normalize_key(key), "limit": limit})
        hits = []
        for row in rows:
            props = dict(row["other"])
            node_label = row["labels"][0] if row["labels"] else label
            hits.append(
                NeighborHit(
                    GraphNode(node_label, props.get("key", ""), props),
                    row["rel"],
                    "out",
                    int(row["distance"]),
                )
            )
        return hits

    async def find_nodes(
        self, label: str, *, contains: str | None = None, limit: int = 50
    ) -> list[GraphNode]:
        cypher = f"MATCH (n:{label}) "
        if contains:
            cypher += "WHERE n.key CONTAINS $needle "
        cypher += "RETURN n ORDER BY n.key LIMIT $limit"
        rows = await self.run_read(
            cypher, {"needle": (contains or "").lower(), "limit": limit}
        )
        return [GraphNode(label, dict(r["n"]).get("key", ""), dict(r["n"])) for r in rows]

    async def run_read(self, cypher: str, params: dict | None = None) -> list[dict]:
        async with self._driver.session(default_access_mode="READ") as session:
            result = await session.run(cypher, **(params or {}))
            return [record.data() async for record in result]

    async def delete_paper(self, paper_id: str) -> int:
        async with self._driver.session() as session:
            result = await session.run(
                "MATCH (p:Paper {key: $key}) DETACH DELETE p RETURN count(p) AS deleted",
                key=paper_id.lower(),
            )
            record = await result.single()
        return int(record["deleted"]) if record else 0

    async def counts(self) -> dict[str, int]:
        rows = await self.run_read(
            "MATCH (n) RETURN labels(n)[0] AS label, count(n) AS total"
        )
        counts = {row["label"]: int(row["total"]) for row in rows if row["label"]}
        rels = await self.run_read("MATCH ()-[r]->() RETURN count(r) AS total")
        counts["_relationships"] = int(rels[0]["total"]) if rels else 0
        return counts

    async def close(self) -> None:
        await self._driver.close()


_store: GraphStore | None = None


def get_graph_store() -> GraphStore:
    global _store
    if _store is not None:
        return _store

    settings = get_settings()
    backend = settings.graph_store_backend.lower()
    if backend == "neo4j":
        _store = Neo4jGraphStore(settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password)
    elif backend == "memory":
        path = Path(settings.storage_local_path) / "graph" / "graph.json"
        _store = InMemoryGraphStore(persist_path=path)
    else:
        raise ValueError(f"Unknown graph store backend: {settings.graph_store_backend!r}")

    logger.info("graph_store_ready", backend=_store.name)
    return _store


def reset_graph_store() -> None:
    global _store
    _store = None
