# ResearchMind AI

An AI-powered research platform for understanding, comparing, and reasoning over large corpora of
scientific papers — semantic + hybrid retrieval, a LangGraph multi-agent reasoning system, an
automatically constructed knowledge graph, and literature-review generation with verifiable citations.

This is not a "chat with a PDF" demo. It's built as a modular, enterprise-style system: a document
processing pipeline (parsing, OCR, table/figure/equation/reference extraction), a hybrid retrieval
engine (dense + BM25 + reranking + metadata/citation-aware filtering), a pluggable multi-provider LLM
gateway, a Neo4j knowledge graph linking papers/authors/datasets/models/methods, and an
agentic reasoning layer implementing Self-RAG / Corrective RAG / reflection patterns.

See [`docs/architecture.md`](./docs/architecture.md) for the full system design and
[`docs/milestones.md`](./docs/milestones.md) for the incremental build plan this project follows.

## Project status

Actively being built milestone-by-milestone (see roadmap above). Currently on **Milestone 1: repo
scaffolding**.

## Repository layout

```
ResearchMind-AI/
  frontend/     Next.js 14 (App Router) + TypeScript + Tailwind + shadcn/ui
  backend/      FastAPI (async) + SQLAlchemy + Celery
  infra/        docker-compose, Kubernetes manifests, NGINX, Prometheus/Grafana
  docs/         architecture, milestones, API docs, diagrams
  .github/      CI/CD workflows
```

## Tech stack

- **Frontend**: Next.js, React, TypeScript, Tailwind CSS, shadcn/ui, Framer Motion
- **Backend**: Python, FastAPI, async SQLAlchemy 2.0, Celery, Redis
- **AI**: LangGraph (multi-agent orchestration), a pluggable LLM Gateway (OpenAI / Anthropic / Gemini
  / open-weight models), pluggable embedding providers (OpenAI / BGE / Jina)
- **Retrieval**: FAISS (dev) / Qdrant (prod) dense vectors, BM25 sparse search, hybrid fusion,
  cross-encoder reranking
- **Knowledge graph**: Neo4j
- **Evaluation**: RAGAS, DeepEval, custom benchmarks
- **Deployment**: Docker Compose, Kubernetes, NGINX, GitHub Actions, Prometheus/Grafana

## Local development

### Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (for the full stack: Postgres,
  Redis, and — from later milestones — Qdrant and Neo4j)
- Node.js 20+ and npm
- Python 3.11–3.12 (the backend targets this range; a newer/older interpreter on your `PATH` is fine
  for other tools but shouldn't be used to run the backend directly — use a matching interpreter or
  Docker instead)

### Option A — Docker Compose (recommended, matches production topology)

```bash
cp .env.example .env   # fill in real secrets/API keys
cd infra
docker compose up --build
```

- Backend: http://localhost:8000/health
- Frontend: http://localhost:3000

### Option B — Run services natively (fastest inner loop while iterating)

Backend:

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate      # Windows
pip install -e ".[dev]"
alembic upgrade head        # requires a running Postgres (see DATABASE_URL in .env)
uvicorn app.main:app --reload
```

Frontend:

```bash
cd frontend
npm install
npm run dev
```

## Testing

```bash
# backend
cd backend && pytest

# frontend
cd frontend && npm run lint && npm run build
```

## License

MIT — see [`LICENSE`](./LICENSE).
