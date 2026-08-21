# Contributing

## Setup

```bash
cd backend
python -m venv .venv && .venv/Scripts/activate
pip install -e ".[dev]"
pytest

cd ../frontend
npm ci
npm run lint && npm run typecheck
```

No containers or API keys are needed to run either suite. If a change makes them
necessary, that is a design problem with the change rather than a setup step to
document.

## Conventions

**Adding a backend that already has an interface.** Implement the abstract base
in the relevant module, register it in the factory function, add a settings
value, and document it in `.env.example`. Do not add a conditional to the caller
— if a caller needs to know which backend it has, the interface is wrong.

**Adding a prompt.** Prompts go in `app/llm/prompts.py`, never inline. They are
the system's actual behaviour specification, and keeping them in one module makes
them reviewable in a diff and swappable for evaluation.

**Parsing model output.** Use `app.llm.json_output.load_json_object`. Models
asked for strict JSON still wrap it in fences and trail explanations after the
closing brace; the salvage logic exists so nobody reimplements it per call site.

**Writing model output to state.** Validate against a closed schema first.
Extraction is the one place a hallucination becomes durable, so unknown types and
malformed shapes are dropped rather than stored and cleaned up later.

**Comments.** Comment the decision, not the mechanism. `# increment the counter`
is noise; a note explaining why fusion is reciprocal-rank rather than a weighted
sum is the thing a reader cannot recover from the code.

## Testing

Tests use real artefacts. `tests/factories.py` builds actual PDFs with PyMuPDF
and pushes them through the real ingestion, indexing, and retrieval pipeline.
This is slower than mocking and catches things mocking cannot — the BM25 desync
bug, the eager-Celery nested event loop, and the unique-index violation on
reprocessing were all found this way.

Test names state the behaviour, not the function:

```python
def test_authorization_is_applied_before_scoring_not_after() -> None:
def test_out_of_range_citations_are_discarded() -> None:
```

A test whose name is `test_retrieve` tells a future reader nothing about what
broke when it fails.

Where a test encodes a non-obvious decision, put the reason in a docstring:

```python
def test_a_single_year_is_new_not_rising() -> None:
    """One point is not a direction."""
```

Deterministic LLM behaviour comes from `ScriptedProvider` in `tests/test_rag.py`,
which returns queued responses in order. It makes control flow testable —
which node runs next, whether the revision cycle triggers, whether the bound
holds — without asserting anything about answer quality, which is a model
property and not this codebase's to guarantee.

## Migrations

```bash
alembic revision --autogenerate -m "describe the change"
alembic upgrade head
alembic downgrade -1    # verify the round trip before committing
```

Autogenerate is a starting point, not an output. Check the generated file: it
routinely misses index renames and server defaults.

## Commits

One logical change per commit. The message says what changed on the first line
and why in the body — the diff already shows the what, so a body that repeats it
adds nothing.

```
feat(retrieval): fuse dense and sparse results by reciprocal rank

Cosine similarity and BM25 scores are on incompatible scales, and normalising
them requires knowing each distribution, which shifts with the corpus. RRF uses
only the ordering, which is the part that is actually comparable.
```

Prefixes: `feat`, `fix`, `refactor`, `test`, `docs`, `chore`, `perf`.

## CI

Three workflows run on every push:

- **ci** — ruff, pytest, the retrieval benchmark, then eslint, tsc, and next build
- **security** — pip-audit, npm audit, secret scan
- **cd** — builds and deploys on `main` and tags

The benchmark gate is the one worth knowing about: it ingests a synthetic corpus
through the real pipeline and fails on retrieval quality, so a regression in
chunking or fusion is caught even when every unit test still passes.
