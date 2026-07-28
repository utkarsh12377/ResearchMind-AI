# Milestone Roadmap

Living tracker for the incremental build of ResearchMind AI. Each milestone is implemented, tested,
and pushed to `main` before the next one starts. See [`docs/architecture.md`](./architecture.md) for
the system design these milestones build toward.

## Phase 0 — Foundation
- [x] 1. Repo scaffolding: monorepo layout, Docker Compose skeleton (profile-gated), FastAPI health
      endpoint, Next.js shell, CI pipeline, docs skeleton.
- [x] 2. Core backend architecture: settings, structured logging, exception handling, async DB
      session management, Alembic migrations, base domain models (User, Paper, Workspace).
- [x] 3. AuthN/AuthZ: JWT auth, register/login, API keys, roles, rate limiting.
- [ ] 4. Frontend shell: layout, dark mode, shadcn setup, auth pages, protected routing, typed API
      client + React Query.

## Phase 1 — Ingestion Pipeline
- [ ] 5. Paper upload & storage abstraction, Paper entity + status tracking, Celery skeleton.
- [ ] 6. PDF parsing (PyMuPDF): text/layout extraction, metadata (title/authors/abstract).
- [ ] 7. OCR pipeline for scanned papers (text-layer density check + Tesseract/PaddleOCR).
- [ ] 8. Table & figure extraction (Docling/Camelot) with captions.
- [ ] 9. Equation & reference/citation extraction, DOI/arXiv resolution.
- [ ] 10. Layout-aware hierarchical chunking (paper -> section -> paragraph/table/figure nodes).

## Phase 2 — Embedding & Retrieval Core
- [ ] 11. Pluggable embedding provider interface (OpenAI/BGE/Jina) + batch embedding Celery tasks + cache.
- [ ] 12. VectorStore abstraction: FAISS dev backend + Qdrant prod backend, collection schema.
- [ ] 13. BM25 sparse index + hybrid RRF fusion.
- [ ] 14. Cross-encoder reranking stage.
- [ ] 15. Metadata filter DSL + citation-aware retrieval boosting.
- [ ] 16. Retrieval API + basic streamed RAG chat endpoint with source citations.

## Phase 3 — LLM Gateway & Reasoning
- [ ] 17. LLM Gateway: multi-provider, streaming, retries/fallback, token accounting, prompt registry.
- [ ] 18. Context compression / long-context optimization (recursive summarization, token budget manager).
- [ ] 19. Self-RAG / Corrective RAG loop (retrieval sufficiency scoring + corrective re-retrieval).
- [ ] 20. Reflection & verification agent (answer-vs-source check, hallucination flagging, confidence score).

## Phase 4 — Multi-Agent System (LangGraph)
- [ ] 21. LangGraph skeleton: state schema, Planner + Retriever agent nodes, streamed intermediate steps.
- [ ] 22. Agents batch 1: Ranking, Verification, Reasoning, Citation.
- [ ] 23. Agents batch 2: Critic, Reflection, Summarizer, Report Generator - end-to-end lit-review draft.
- [ ] 24. Web Search Agent + external tool-use framework (sandboxed).
- [ ] 25. Agent orchestrator API + frontend agent-status/timeline streaming view.

## Phase 5 — Knowledge Graph
- [ ] 26. Neo4j schema + LLM-assisted entity/relation extraction pipeline.
- [ ] 27. GraphRAG: NL->Cypher query engine, Graph Retrieval Agent wired into main pipeline.
- [ ] 28. Graph analytics: author/citation/dataset/model graphs, contradiction/agreement detection.
- [ ] 29. Graph query endpoints tailored for frontend visualization.

## Phase 6 — Advanced Research Features
- [ ] 30. Structured extraction: dataset/model/metric/hyperparameter/experiment/methodology extractors
      + comparison tables.
- [ ] 31. Research trend analysis & timeline generation.
- [ ] 32. Literature review / report generation pipeline (outline -> draft -> citation weaving -> export).
- [ ] 33. Research gap discovery & future-direction suggestion agent chain.

## Phase 7 — Evaluation & Observability
- [ ] 34. RAGAS + DeepEval harness, custom benchmark, CI-gated eval.
- [ ] 35. Prometheus + Grafana + tracing + admin dashboard (usage/cost/latency).
- [ ] 36. Frontend feature completion: KG visualization, timeline, comparison view, citation explorer,
      dataset/model explorer, trends dashboard, search filters.
- [ ] 37. Security hardening: input validation audit, prompt-injection mitigations for agent tools,
      secrets management, dependency scanning.

## Phase 8 — Deployment
- [ ] 38. Kubernetes manifests/Helm chart, NGINX ingress.
- [ ] 39. CI/CD build+push+deploy workflow (staging).
- [ ] 40. Final docs pass: architecture/sequence/class diagrams, Swagger polish, deployment +
      contribution guides.
