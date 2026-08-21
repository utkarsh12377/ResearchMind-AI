# Milestone Roadmap

Build plan for ResearchMind AI. Each milestone was implemented, tested, and pushed before the next
one started. All 40 are complete. See [`docs/architecture.md`](./architecture.md) for
the system design these milestones build toward.

## Phase 0 — Foundation
- [x] 1. Repo scaffolding: monorepo layout, Docker Compose skeleton (profile-gated), FastAPI health
      endpoint, Next.js shell, CI pipeline, docs skeleton.
- [x] 2. Core backend architecture: settings, structured logging, exception handling, async DB
      session management, Alembic migrations, base domain models (User, Paper, Workspace).
- [x] 3. AuthN/AuthZ: JWT auth, register/login, API keys, roles, rate limiting.
- [x] 4. Frontend shell: layout, dark mode, shadcn setup, auth pages, protected routing, typed API
      client + React Query.

## Phase 1 — Ingestion Pipeline
- [x] 5. Paper upload & storage abstraction, Paper entity + status tracking, Celery skeleton.
- [x] 6. PDF parsing (PyMuPDF): text/layout extraction, metadata (title/authors/abstract).
- [x] 7. OCR pipeline for scanned papers (text-layer density check + Tesseract/PaddleOCR).
- [x] 8. Table & figure extraction (Docling/Camelot) with captions.
- [x] 9. Equation & reference/citation extraction, DOI/arXiv resolution.
- [x] 10. Layout-aware hierarchical chunking (paper -> section -> paragraph/table/figure nodes).

## Phase 2 — Embedding & Retrieval Core
- [x] 11. Pluggable embedding provider interface (OpenAI/BGE/Jina) + batch embedding Celery tasks + cache.
- [x] 12. VectorStore abstraction: FAISS dev backend + Qdrant prod backend, collection schema.
- [x] 13. BM25 sparse index + hybrid RRF fusion.
- [x] 14. Cross-encoder reranking stage.
- [x] 15. Metadata filter DSL + citation-aware retrieval boosting.
- [x] 16. Retrieval API + basic streamed RAG chat endpoint with source citations.

## Phase 3 — LLM Gateway & Reasoning
- [x] 17. LLM Gateway: multi-provider, streaming, retries/fallback, token accounting, prompt registry.
- [x] 18. Context compression / long-context optimization (recursive summarization, token budget manager).
- [x] 19. Self-RAG / Corrective RAG loop (retrieval sufficiency scoring + corrective re-retrieval).
- [x] 20. Reflection & verification agent (answer-vs-source check, hallucination flagging, confidence score).

## Phase 4 — Multi-Agent System (LangGraph)
- [x] 21. LangGraph skeleton: state schema, Planner + Retriever agent nodes, streamed intermediate steps.
- [x] 22. Agents batch 1: Ranking, Verification, Reasoning, Citation.
- [x] 23. Agents batch 2: Critic, Reflection, Summarizer, Report Generator - end-to-end lit-review draft.
- [x] 24. Web Search Agent + external tool-use framework (sandboxed).
- [x] 25. Agent orchestrator API + frontend agent-status/timeline streaming view.

## Phase 5 — Knowledge Graph
- [x] 26. Neo4j schema + LLM-assisted entity/relation extraction pipeline.
- [x] 27. GraphRAG: NL->Cypher query engine, Graph Retrieval Agent wired into main pipeline.
- [x] 28. Graph analytics: author/citation/dataset/model graphs, contradiction/agreement detection.
- [x] 29. Graph query endpoints tailored for frontend visualization.

## Phase 6 — Advanced Research Features
- [x] 30. Structured extraction: dataset/model/metric/hyperparameter/experiment/methodology extractors
      + comparison tables.
- [x] 31. Research trend analysis & timeline generation.
- [x] 32. Literature review / report generation pipeline (outline -> draft -> citation weaving -> export).
- [x] 33. Research gap discovery & future-direction suggestion agent chain.

## Phase 7 — Evaluation & Observability
- [x] 34. RAGAS + DeepEval harness, custom benchmark, CI-gated eval.
- [x] 35. Prometheus + Grafana + tracing + admin dashboard (usage/cost/latency).
- [x] 36. Frontend feature completion: KG visualization, timeline, comparison view, citation explorer,
      dataset/model explorer, trends dashboard, search filters.
- [x] 37. Security hardening: input validation audit, prompt-injection mitigations for agent tools,
      secrets management, dependency scanning.

## Phase 8 — Deployment
- [x] 38. Kubernetes manifests/Helm chart, NGINX ingress.
- [x] 39. CI/CD build+push+deploy workflow (staging).
- [x] 40. Final docs pass: architecture/sequence/class diagrams, Swagger polish, deployment +
      contribution guides.
