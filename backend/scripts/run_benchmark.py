"""Run the evaluation benchmark and exit non-zero if it regresses.

Builds a throwaway corpus in SQLite, ingests it through the real pipeline, and
runs the benchmark against it. Using the real pipeline rather than fixtures is
the point: a regression in chunking or fusion should fail this, and it would not
if the contexts were hand-written.

    python scripts/run_benchmark.py [--dataset cases.json] [--json report.json]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402

import app.models  # noqa: E402, F401
from app.db.base import Base  # noqa: E402
from app.eval.runner import Thresholds, run_benchmark  # noqa: E402
from app.models import Paper, PaperStatus, User, Workspace  # noqa: E402
from app.retrieval.embeddings import HashEmbeddingProvider  # noqa: E402
from app.retrieval.indexer import index_paper  # noqa: E402
from app.retrieval.service import rebuild_sparse_index  # noqa: E402
from app.retrieval.sparse import BM25Index  # noqa: E402
from app.retrieval.vector_store import InMemoryVectorStore  # noqa: E402
from app.worker.tasks import _process_paper  # noqa: E402

CORPUS = (
    {
        "title": "Hybrid Retrieval for Scientific Literature",
        "abstract": "We present a hybrid retrieval system combining dense and sparse matching.",
        "body": "Our system reaches an accuracy of 91.2 on the benchmark using a dual encoder.",
    },
    {
        "title": "Cross Encoder Reranking at Scale",
        "abstract": "We rerank candidate passages with a cross encoder for scientific search.",
        "body": "Reranking improves ndcg by 4.1 points over the first-stage retriever.",
    },
)


async def _seed(session) -> User:  # noqa: ANN001
    from app.core.storage import build_storage_key, compute_checksum, get_storage
    from tests.factories import build_pdf

    user = User(email="benchmark@example.com", hashed_password="x")
    workspace = Workspace(name="Benchmark", owner=user)
    session.add_all([user, workspace])
    await session.commit()

    storage = get_storage()
    for index, spec in enumerate(CORPUS):
        pdf = build_pdf(**spec)
        checksum, size = compute_checksum(pdf)
        key = build_storage_key(checksum, ".pdf")
        storage.save(key, pdf)

        paper = Paper(
            workspace_id=workspace.id,
            uploaded_by_id=user.id,
            original_filename=f"paper-{index}.pdf",
            content_type="application/pdf",
            checksum=checksum,
            size_bytes=size,
            storage_key=key,
            status=PaperStatus.PENDING,
        )
        session.add(paper)
        await session.commit()
        await _process_paper(session, paper.id)

    return user


async def main() -> int:
    parser = argparse.ArgumentParser(description="Run the ResearchMind evaluation benchmark")
    parser.add_argument("--dataset", help="Path to a JSON dataset of eval cases")
    parser.add_argument("--json", dest="json_out", help="Write the full report here")
    parser.add_argument("--min-faithfulness", type=float, default=None)
    parser.add_argument("--min-precision", type=float, default=None)
    parser.add_argument(
        "--limit",
        type=int,
        default=3,
        help="Passages retrieved per case. Small by default: on a corpus this "
        "size a larger limit only adds padding, which precision correctly punishes.",
    )
    args = parser.parse_args()

    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    provider = HashEmbeddingProvider(dimensions=256)
    store = InMemoryVectorStore()
    bm25 = BM25Index()

    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    async with session_factory() as session:
        user = await _seed(session)

        from sqlalchemy import select

        for paper_id in (await session.scalars(select(Paper.id))).all():
            await index_paper(session, paper_id, provider=provider, store=store)
        await rebuild_sparse_index(session, bm25=bm25)

        thresholds = Thresholds.from_settings()
        if args.min_faithfulness is not None:
            thresholds.faithfulness = args.min_faithfulness
        if args.min_precision is not None:
            thresholds.context_precision = args.min_precision

        report = await run_benchmark(
            session,
            user,
            dataset_path=args.dataset,
            thresholds=thresholds,
            limit=args.limit,
            provider=provider,
            store=store,
            bm25=bm25,
        )

    await engine.dispose()

    print(report.summary())
    for result in report.results:
        status = "skip" if result.skipped else ("ok  " if result.passed else "FAIL")
        print(f"  {status} {result.case_id}: {'; '.join(result.failures) or 'passed'}")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report.as_dict(), indent=2), encoding="utf-8")

    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
