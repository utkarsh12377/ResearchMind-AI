# API Reference

Interactive docs are served at `/docs` (Swagger) and `/redoc` when the service is
running. This page covers the parts a schema cannot express: what each endpoint
is for, and where the behaviour is deliberate.

Base path: `/api/v1`.

## Authentication

Two credential types resolve to the same identity.

```http
POST /api/v1/auth/register    {"email": "...", "password": "..."}
POST /api/v1/auth/login       username=...&password=...   (form-encoded, OAuth2)
GET  /api/v1/auth/me
```

Login is form-encoded because it follows the OAuth2 password flow, which is what
lets FastAPI's `/docs` authorize button work without custom wiring.

Interactive clients send `Authorization: Bearer <jwt>`. Programmatic clients send
`X-API-Key: <key>`. Keys are shown once at creation and stored as SHA-256 hashes
with a short display prefix, so a database dump contains nothing usable.

```http
GET    /api/v1/auth/api-keys
POST   /api/v1/auth/api-keys       {"name": "..."}
DELETE /api/v1/auth/api-keys/{id}
```

## Errors

Every error is the same envelope, so a client parses one shape:

```json
{ "error": { "type": "not_found", "message": "Paper does not exist" } }
```

`401` unauthenticated · `403` authenticated but not permitted · `404` missing ·
`409` conflict · `413` body too large · `422` validation · `429` rate limited.

Every response also carries `X-Request-ID`. Quote it when reporting a failure —
it is the one string that finds the corresponding log lines.

## Papers

```http
POST   /api/v1/papers            multipart/form-data, field "file"
GET    /api/v1/papers?limit=&offset=
GET    /api/v1/papers/{id}
DELETE /api/v1/papers/{id}
GET    /api/v1/papers/{id}/assets?kind=table|figure
```

Upload returns `201` as soon as the file is durably stored; parsing, embedding,
graph building, and result extraction run as separate background tasks. Poll
`status` (`pending` → `processing` → `ready` | `failed`) rather than waiting on
the request.

Uploads are verified against the PDF magic bytes, not the declared content type,
and filenames are sanitized before storage. Re-uploading identical bytes into the
same workspace is a `409`; the constraint is in the database so concurrent
uploads cannot race past it.

## Search

```http
POST /api/v1/search
{
  "query": "cross encoder reranking",
  "limit": 10,
  "filters": { "paper_ids": [], "kinds": ["table"], "year_from": 2020 },
  "use_reranker": true
}
```

Each result carries `dense_rank`, `sparse_rank`, and `rerank_score`. This is not
decoration: the ranking is a fusion of two retrievers plus a reranking stage, and
a bare score would make it unauditable. A result with both ranks populated was
found independently by semantic and keyword search, which is the strongest signal
the pipeline produces.

`supporting_sentences` are extracted from the passage, so a UI can show the part
that matched rather than a wall of text.

## Chat and research

```http
POST /api/v1/chat              {"question": "...", "filters": {}}
POST /api/v1/chat/stream       text/event-stream
POST /api/v1/research          {"question": "...", "filters": {}}
POST /api/v1/research/stream   text/event-stream
```

`/chat` is single-turn RAG with the corrective loop: grade the retrieval, rewrite
and re-retrieve if it is weak, answer, verify the answer against its sources.

`/research` runs the full agent graph — planner, retriever, graph retriever, web
search, ranker, reasoner, critic, reflector, verifier, citation, summarizer —
and returns the recorded timeline alongside the answer.

Both streaming endpoints require a POST with an `Authorization` header, which
`EventSource` cannot do. Clients use `fetch` and parse the SSE frames themselves;
`streamResearch` in the frontend client is a worked example, including buffering
frames across chunk boundaries.

`confidence` is deliberately conservative: an answer the verifier flagged is
capped at 0.35 even when retrieval looked strong.

## Knowledge graph

```http
GET  /api/v1/graph/overview
GET  /api/v1/graph/network?view=entities|citations|authors&labels=&min_papers=
GET  /api/v1/graph/neighbors?label=Dataset&key=squad&depth=2
POST /api/v1/graph/query       {"question": "which papers share a dataset?"}
```

`/network` returns flat node and edge lists because that is what a force-directed
layout consumes; nested paths would ship the same node several times.

`/query` generates read-only Cypher and validates it against a whitelist of
clauses, labels, and relationship types before running anything. The generated
query is returned whether it ran or not — when validation rejects one, the
rejected query is the most useful thing to show.

Only the Neo4j backend executes Cypher. On the in-memory backend this endpoint
returns an explicit error rather than pretending to parse the query.

## Insights

```http
POST /api/v1/insights/comparison   {"paper_ids": [], "metrics": [], "datasets": []}
POST /api/v1/insights/matrix       {"paper_ids": []}
POST /api/v1/insights/trends       {"topic": "...", "paper_ids": []}
POST /api/v1/insights/consistency  {"paper_ids": [], "include_claims": true}
POST /api/v1/insights/gaps         {"topic": "...", "paper_ids": []}
POST /api/v1/insights/review       {"topic": "...", "paper_ids": [], "max_sections": 6}
```

An empty `paper_ids` means the caller's whole library. Requested ids are
intersected with what the caller can read, so passing someone else's id narrows
the scope to nothing rather than returning a distinguishable error.

`/comparison` groups on the normalized (dataset, metric) pair, so two F1 scores
on different datasets stay in different rows. Results with no dataset attached
are reported in `ungrouped_results` rather than dropped.

`/consistency` runs two detectors. Numeric conflicts are arithmetic — same model,
dataset, and metric, differing by more than the noise floor — and need no model
call. Claim conflicts need one, but only after a lexical filter narrows the
candidate pairs.

`/review` returns markdown and BibTeX alongside the structured sections. Citation
numbering is global across the whole review, and markers pointing outside what a
section was actually given are dropped as fabricated references.

## Admin

```http
GET  /api/v1/admin/status
POST /api/v1/admin/reindex-sparse
```

Superuser only — both read across every workspace. `/status` answers the question
that gets asked during an incident: which backend is each pluggable component
currently resolved to.

## System

```http
GET /health     {"status": "ok"}
GET /metrics    Prometheus exposition format
```

`/metrics` is not routed through the Ingress or the NGINX config; Prometheus
scrapes the pod directly.

## Rate limits

Applied per-IP on the authentication routes, which are where an unauthenticated
attacker otherwise gets unlimited attempts. Exceeding one returns `429`.
