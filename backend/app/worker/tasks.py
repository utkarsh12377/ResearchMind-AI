"""Celery tasks for the document ingestion pipeline."""

from __future__ import annotations

import asyncio
import io
import uuid
from collections.abc import Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor
from typing import TypeVar

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger
from app.core.storage import get_storage
from app.ingestion.pdf_parser import ParsedDocument, PdfParseError, parse_pdf
from app.models import AssetKind, Paper, PaperAsset, PaperReference, PaperStatus
from app.worker.celery_app import celery_app

configure_logging()
logger = get_logger(__name__)

T = TypeVar("T")


def _run_with_session(operation: Callable[[AsyncSession], Awaitable[T]]) -> T:
    """Run an async DB operation from a synchronous Celery task.

    A fresh engine is created per task rather than reusing the API process's
    engine: Celery workers are separate processes (and prefork children can't
    safely inherit an async connection pool), so sharing one would hand out
    connections bound to another process's event loop.
    """

    async def runner() -> T:
        engine = create_async_engine(get_settings().database_url, pool_pre_ping=True)
        session_factory = async_sessionmaker(engine, expire_on_commit=False)
        try:
            async with session_factory() as session:
                return await operation(session)
        finally:
            await engine.dispose()

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # The normal path: a Celery worker process, no loop running.
        return asyncio.run(runner())

    # task_always_eager runs the task inline inside the caller — which for an
    # upload request is a thread already driving an event loop, and asyncio.run
    # refuses to nest. Hand the coroutine to a worker thread with its own loop.
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, runner()).result()


@celery_app.task(name="ingestion.process_paper", bind=True, max_retries=2)
def process_paper(self, paper_id: str) -> dict[str, object]:  # noqa: ANN001
    """Parse an uploaded paper and record the extracted metadata."""
    return _run_with_session(lambda session: _process_paper(session, uuid.UUID(paper_id)))


async def _process_paper(db: AsyncSession, paper_id: uuid.UUID) -> dict[str, object]:
    paper = await db.get(Paper, paper_id)
    if paper is None:
        logger.warning("ingestion_paper_missing", paper_id=str(paper_id))
        return {"paper_id": str(paper_id), "status": "missing"}

    paper.status = PaperStatus.PROCESSING
    paper.error_message = None
    await db.commit()

    settings = get_settings()
    try:
        with get_storage().open(paper.storage_key) as handle:
            parsed = parse_pdf(
                handle,
                enable_ocr=settings.ocr_enabled,
                enable_tables=settings.extract_tables,
                enable_figures=settings.extract_figures,
                ocr_dpi=settings.ocr_dpi,
            )
    except (PdfParseError, FileNotFoundError, OSError) as exc:
        logger.error("ingestion_failed", paper_id=str(paper_id), error=str(exc))
        paper.status = PaperStatus.FAILED
        paper.error_message = str(exc)[:2000]
        await db.commit()
        return {"paper_id": str(paper_id), "status": PaperStatus.FAILED.value}

    paper.page_count = parsed.page_count
    paper.title = parsed.title or paper.original_filename
    paper.authors = parsed.authors
    paper.abstract = parsed.abstract
    paper.ocr_page_count = parsed.ocr_page_count
    paper.status = PaperStatus.READY

    await _replace_assets(db, paper, parsed)
    await db.commit()

    logger.info(
        "ingestion_completed",
        paper_id=str(paper_id),
        page_count=parsed.page_count,
        scanned=parsed.is_probably_scanned,
    )
    return {
        "paper_id": str(paper_id),
        "status": PaperStatus.READY.value,
        "page_count": parsed.page_count,
        "is_probably_scanned": parsed.is_probably_scanned,
        "ocr_page_count": parsed.ocr_page_count,
        "tables": len(parsed.tables),
        "figures": len(parsed.figures),
        "references": len(parsed.references),
    }


async def _replace_assets(db: AsyncSession, paper: Paper, parsed: ParsedDocument) -> None:
    """Persist extracted tables and figures, replacing any previous run's output.

    Reprocessing a paper must not accumulate duplicates, so existing assets are
    cleared first.
    """
    existing = await db.scalars(select(PaperAsset).where(PaperAsset.paper_id == paper.id))
    for asset in existing:
        await db.delete(asset)

    stale_refs = await db.scalars(
        select(PaperReference).where(PaperReference.paper_id == paper.id)
    )
    for reference in stale_refs:
        await db.delete(reference)

    for reference in parsed.references:
        db.add(
            PaperReference(
                paper_id=paper.id,
                order=reference.order,
                raw_text=reference.raw_text,
                title=reference.title,
                doi=reference.doi,
                arxiv_id=reference.arxiv_id,
                year=reference.year,
            )
        )

    for table in parsed.tables:
        db.add(
            PaperAsset(
                paper_id=paper.id,
                kind=AssetKind.TABLE,
                page_number=table.page_number,
                caption=table.caption,
                content=table.markdown,
            )
        )

    storage = get_storage()
    for index, figure in enumerate(parsed.figures):
        storage_key = None
        if figure.image_png:
            storage_key = f"figures/{paper.id}/p{figure.page_number}-{index}.png"
            storage.save(storage_key, io.BytesIO(figure.image_png))

        db.add(
            PaperAsset(
                paper_id=paper.id,
                kind=AssetKind.FIGURE,
                page_number=figure.page_number,
                caption=figure.caption,
                storage_key=storage_key,
            )
        )
