# Deployment

Three ways to run this, in increasing order of setup cost.

## 1. Local, no containers

The whole stack has an offline path. Every pluggable layer ships a default that
needs no network and no key, so a fresh clone runs end to end:

| Layer | Offline default | Production |
| --- | --- | --- |
| Embeddings | `hash` (deterministic, local) | `openai` / `jina` / `bge` |
| Vector store | `memory` (exact, persisted to disk) | `qdrant` |
| Reranker | `lexical` | `cross-encoder` |
| Graph store | `memory` (JSON-backed) | `neo4j` |
| LLM | `echo` | `anthropic` / `openai` / `gemini` |
| Database | SQLite (tests) | Postgres |
| Queue | Celery eager mode | Celery + Redis |

The hash embedding provider is not a fallback anyone should ship — it has no
semantics at all, and it says so in its own docstring. It exists so the retrieval
pipeline is runnable and testable without an API key, which is what keeps the
test suite honest about the pipeline rather than about a provider.

```bash
cd backend
py -3.10 -m venv .venv && .venv/Scripts/activate
pip install -e ".[dev]"
pytest
uvicorn app.main:app --reload
```

```bash
cd frontend
npm ci
npm run dev
```

## 2. Docker Compose

```bash
cp .env.example .env      # fill in keys as needed
cd infra
docker compose up --build
```

Default services: Postgres, Redis, backend, worker, frontend.

Profile-gated services stay down until asked for, so the day-one stack is four
containers rather than eight:

```bash
docker compose --profile phase2 up          # + Qdrant
docker compose --profile phase5 up          # + Neo4j
docker compose --profile observability up   # + Prometheus, Grafana
```

Grafana is on `:3001` with the dashboard already provisioned; Prometheus is on
`:9090`.

### Line endings

`.gitattributes` forces LF on shell scripts. This is not cosmetic: a CRLF
`entrypoint.sh` produces `bad interpreter: /bin/sh^M` inside the Linux
container, and the error names the shell rather than the line ending, which
makes it a genuinely confusing hour to lose.

## 3. Kubernetes

Manifests are in `infra/k8s/base`, assembled with Kustomize.

```bash
cp infra/k8s/base/secret.example.yaml infra/k8s/base/secret.yaml
# fill in real values -- secret.yaml is git-ignored
kubectl apply -f infra/k8s/base/secret.yaml
kubectl apply -k infra/k8s/base
```

What the manifests set up:

- **backend** — 2 replicas, HPA to 8 on CPU, PodDisruptionBudget, separate
  startup and liveness probes.
- **worker** — Celery, 120s termination grace period, higher memory request
  than the API because PDF rasterisation for OCR is the memory peak in the
  system.
- **frontend** — 2 replicas behind a Service.
- **migrate** — a Job, not an entrypoint step.
- **stateful** — Postgres, Redis, Qdrant, Neo4j as StatefulSets. These are the
  first things to replace with managed services; the app only needs connection
  strings, so swapping any of them is a ConfigMap edit.
- **ingress** — NGINX with buffering disabled on the SSE routes and a body size
  matching `max_upload_bytes`.

### Why migrations are a Job

With two API replicas, running Alembic in the container entrypoint means two
processes racing for the same lock on every rollout. A Job runs once and the
deploy waits on it, so a failed migration stops the rollout instead of leaving
one replica on the new schema and one on the old.

### Storage

`paper-storage` is `ReadWriteMany` because the API writes the uploaded blob and
a worker reads it back. If the cluster has no RWX StorageClass, the seam to
change is the `StorageBackend` interface — swap `LocalStorageBackend` for an S3
or GCS implementation and the PVC disappears entirely.

## CI/CD

`ci.yml` runs on every push and pull request:

1. **backend** — ruff, pytest
2. **evaluation** — ingests a synthetic corpus through the real pipeline and
   gates on retrieval quality, so a regression in chunking or fusion fails even
   when every unit test passes
3. **frontend** — eslint, `tsc --noEmit`, `next build`

`security.yml` runs `pip-audit`, `npm audit`, and a secret scan on push and
weekly. The schedule matters more than the push trigger: advisories usually land
after the code depending on the package was last touched.

`cd.yml` builds both images and tags them with the commit SHA as well as
`latest`. `latest` alone makes a rollback impossible to express, because there
is nothing left to roll back to.

Publishing and deploying are separate triggers. Every push to `main` publishes
images; the deploy job runs only for a `v*` tag or a manual dispatch. A green
build on `main` therefore means the images exist, not that an environment
moved. When it does run, the deploy applies migrations as a Job, applies the
manifests, waits for the rollout, smoke-tests the Service from inside the
cluster, and rolls back on failure.

The deploy job needs a `KUBE_CONFIG` repository secret holding a
base64-encoded kubeconfig. Without it there is nothing to deploy to, which is
the other reason the job is not wired to every push.

## Configuration

Every setting lives in `app/core/config.py` as one `Settings` class and is
documented in `.env.example`. The ones that change behaviour most:

| Variable | Effect |
| --- | --- |
| `EMBEDDING_PROVIDER` | `hash` \| `openai` \| `jina` \| `bge` |
| `VECTOR_STORE_BACKEND` | `memory` \| `faiss` \| `qdrant` |
| `RERANKER_BACKEND` | `lexical` \| `cross-encoder` \| `none` |
| `GRAPH_STORE_BACKEND` | `memory` \| `neo4j` — only `neo4j` runs generated Cypher |
| `DEFAULT_LLM_PROVIDER` | `echo` \| `openai` \| `anthropic` \| `gemini` |
| `FALLBACK_LLM_PROVIDER` | Used only on non-4xx failures; streaming never falls back |
| `CELERY_TASK_ALWAYS_EAGER` | Runs ingestion inline, no broker needed |
| `WEB_SEARCH_ENABLED` | Off unless a provider and key are set |
| `GRAPH_EXTRACTION_ENABLED` | The one ingest stage that costs a model call per paper |

## Operating notes

**The BM25 index is in-process memory.** Dense vectors persist; term statistics
do not. Startup rebuilds them from Postgres, and `POST /api/v1/admin/reindex-sparse`
rebuilds them on demand. If hybrid search starts behaving like dense-only search,
this is the first thing to check — the `SparseIndexEmpty` alert exists for
exactly that.

**`/metrics` is not exposed publicly.** Prometheus scrapes the pod directly.
Both the Ingress and the NGINX config leave it off, because route names, error
rates, and traffic volume are free reconnaissance.

**Fallback is not a retry.** The LLM gateway falls back to a second provider on
outages but never on 4xx, because a malformed request or a content-filter
refusal fails identically everywhere and retrying it elsewhere just doubles the
cost. Streaming never falls back at all — tokens already sent cannot be
retracted, so a mid-stream switch would splice two answers together.
