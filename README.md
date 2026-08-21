# ResearchMind AI

An AI research platform for reasoning across a corpus of scientific papers — hybrid
retrieval, a LangGraph multi-agent system, an automatically constructed knowledge
graph, cross-paper comparison and contradiction detection, and literature-review
generation with citations that are checked rather than trusted.

Not a "chat with a PDF" demo. Every layer is an interface with at least two
implementations, one of which runs offline, so the whole system is runnable and
testable with no containers, no API keys, and no external services.

## What it does

**Ingestion.** PDFs are parsed layout-aware; pages with a thin text layer are
routed to OCR; tables become Markdown and figures are matched to their captions;
the bibliography is parsed into rows with DOI and arXiv identifiers. Chunking
follows the paper's own section structure and carries breadcrumb paths
(`2 Methods > 2.1 Encoder`), so every passage knows where it came from.

**Retrieval.** Dense embeddings and a hand-written BM25 index run concurrently
and are fused by reciprocal rank, then a cross-encoder reranks an over-fetched
candidate set. Every result reports which retriever found it and where the
reranker moved it.

**Reasoning.** A multi-provider LLM gateway with fallback and token accounting,
token-budgeted context packing that compresses rather than truncates, and the
Self-RAG / Corrective RAG loop: grade the retrieval, rewrite and re-retrieve if
it is weak, answer, then verify the answer against its sources.

**Agents.** Ten agents in a LangGraph graph — planner, retriever, graph
retriever, web search, ranker, reasoner, critic, reflector, verifier, citation,
summarizer — with a bounded critic↔reflector revision cycle. Each step streams
to the UI as it completes.

**Knowledge graph.** Entities and relations extracted into Neo4j against a closed
schema, GraphRAG expansion that reaches papers similarity search misses, and
natural-language questions translated into read-only Cypher behind a whitelist
validator.

**Cross-paper analysis.** Comparison tables grouped on the normalized
(dataset, metric) pair, trend analysis and timelines computed from extracted rows
rather than summarised from text, numeric and claim-level contradiction
detection, research gap discovery, and a cited literature review exportable as
Markdown and BibTeX.

**Operations.** Prometheus metrics with deliberate label sets, Grafana dashboards
and alert rules, an evaluation benchmark that gates CI on retrieval quality, and
Kubernetes manifests with a migration Job and rollback on failed deploy.

## Documentation

| Document | Contents |
| --- | --- |
| [`docs/architecture.md`](./docs/architecture.md) | Component diagram, request and ingestion sequence diagrams, retrieval design, design principles |
| [`docs/api.md`](./docs/api.md) | Endpoint reference and the reasoning behind each |
| [`docs/security.md`](./docs/security.md) | Threat model, controls, and stated gaps |
| [`docs/deployment.md`](./docs/deployment.md) | Local, Compose, and Kubernetes deployment |
| [`docs/milestones.md`](./docs/milestones.md) | The 40-milestone build plan this project followed |
| [`CONTRIBUTING.md`](./CONTRIBUTING.md) | Conventions, testing philosophy, commit style |

## Repository layout

```
ResearchMind-AI/
  backend/        FastAPI, async SQLAlchemy, Celery, LangGraph
    app/
      agents/     Multi-agent graph, tools, prompt-injection defences
      core/       Config, logging, security, storage, metrics, middleware
      eval/       Retrieval and answer quality benchmark
      graph/      Knowledge graph: schema, stores, extraction, GraphRAG, Cypher
      ingestion/  Parsing, OCR, tables, figures, references, chunking
      llm/        Gateway, prompts, context packing, corrective RAG
      research/   Extraction, comparison, trends, contradictions, gaps, reviews
      retrieval/  Embeddings, vector stores, BM25, fusion, reranking
  frontend/       Next.js App Router, TypeScript, Tailwind, shadcn/ui
  infra/          Compose, Kubernetes, NGINX, Prometheus, Grafana
  docs/           Architecture, API, security, deployment, milestones
```

## Running it

The fastest path needs nothing installed beyond Python and Node:

```bash
cd backend
python -m venv .venv && .venv/Scripts/activate    # or source .venv/bin/activate
pip install -e ".[dev]"
pytest                                            # 600+ tests, no services needed
uvicorn app.main:app --reload
```

```bash
cd frontend
npm ci && npm run dev
```

That runs against SQLite, a deterministic local embedding provider, an in-memory
vector store, and an offline LLM stub. To point it at real infrastructure:

```bash
cp .env.example .env    # set provider keys and backends
cd infra && docker compose up --build
```

See [`docs/deployment.md`](./docs/deployment.md) for the Compose profiles,
Kubernetes manifests, and the full configuration table.

## Tech stack

**Backend** — Python 3.11, FastAPI, async SQLAlchemy 2.0, Alembic, Celery, Redis,
structlog, PyMuPDF, Tesseract

**AI** — LangGraph, a pluggable LLM gateway (OpenAI / Anthropic / Gemini),
pluggable embeddings (OpenAI / BGE / Jina), cross-encoder reranking

**Data** — Postgres, Qdrant / FAISS, Neo4j, content-addressed blob storage

**Frontend** — Next.js (App Router), TypeScript, Tailwind, shadcn/ui, TanStack
Query, Framer Motion

**Ops** — Docker Compose, Kubernetes, NGINX, GitHub Actions, Prometheus, Grafana

## Testing

```bash
cd backend && pytest && ruff check .
cd frontend && npm run lint && npm run typecheck && npm run build
```

The backend suite is written against real artefacts rather than mocks: tests
build actual PDFs with PyMuPDF and push them through the real ingestion, indexing,
and retrieval pipeline. Several genuine bugs in this codebase were caught that
way and would not have been caught by a mock — including a desync where papers
were dense-indexed but never added to the sparse index, making every new upload
invisible to half of hybrid search until restart.

## License

MIT — see [`LICENSE`](./LICENSE).
