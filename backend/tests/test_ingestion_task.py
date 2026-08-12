"""Covers the ingestion task body directly, without Celery in the loop.

`process_paper` is a thin synchronous wrapper that opens its own engine; the
real work lives in `_process_paper`, which is what these tests drive against
the shared in-memory test session.
"""

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.storage import build_storage_key, compute_checksum, get_storage
from app.models import Paper, PaperStatus, User, Workspace
from app.worker.tasks import _process_paper
from tests.factories import build_pdf, build_scanned_pdf


async def _store_paper(db: AsyncSession, pdf, filename: str = "paper.pdf") -> Paper:  # noqa: ANN001
    user = User(email="ingest@example.com", hashed_password="x")
    workspace = Workspace(name="Personal", owner=user)
    db.add_all([user, workspace])
    await db.flush()

    checksum, size = compute_checksum(pdf)
    key = build_storage_key(checksum, filename)
    get_storage().save(key, pdf)

    paper = Paper(
        workspace_id=workspace.id,
        uploaded_by_id=user.id,
        original_filename=filename,
        content_type="application/pdf",
        checksum=checksum,
        size_bytes=size,
        storage_key=key,
        status=PaperStatus.PENDING,
    )
    db.add(paper)
    await db.commit()
    await db.refresh(paper)
    return paper


@pytest.mark.asyncio
async def test_processing_extracts_metadata_and_marks_ready(db_session: AsyncSession) -> None:
    paper = await _store_paper(
        db_session,
        build_pdf(title="Hybrid Retrieval for Scientific Corpora", pages=2),
    )

    result = await _process_paper(db_session, paper.id)

    assert result["status"] == PaperStatus.READY.value
    await db_session.refresh(paper)
    assert paper.status == PaperStatus.READY
    assert paper.title == "Hybrid Retrieval for Scientific Corpora"
    assert paper.page_count == 2
    assert paper.abstract is not None
    assert paper.error_message is None


@pytest.mark.asyncio
async def test_processing_reports_scanned_documents(db_session: AsyncSession) -> None:
    paper = await _store_paper(db_session, build_scanned_pdf(pages=2), "scan.pdf")

    result = await _process_paper(db_session, paper.id)

    assert result["is_probably_scanned"] is True


@pytest.mark.asyncio
async def test_unparseable_file_marks_paper_failed(db_session: AsyncSession) -> None:
    import io

    paper = await _store_paper(db_session, io.BytesIO(b"not a pdf at all"), "broken.pdf")

    result = await _process_paper(db_session, paper.id)

    assert result["status"] == PaperStatus.FAILED.value
    await db_session.refresh(paper)
    assert paper.status == PaperStatus.FAILED
    assert paper.error_message


@pytest.mark.asyncio
async def test_falls_back_to_filename_when_no_title_found(db_session: AsyncSession) -> None:
    paper = await _store_paper(db_session, build_scanned_pdf(), "my-uploaded-name.pdf")

    await _process_paper(db_session, paper.id)

    await db_session.refresh(paper)
    assert paper.title == "my-uploaded-name.pdf"


@pytest.mark.asyncio
async def test_missing_paper_is_handled_gracefully(db_session: AsyncSession) -> None:
    import uuid

    result = await _process_paper(db_session, uuid.uuid4())

    assert result["status"] == "missing"


@pytest.mark.asyncio
async def test_run_with_session_works_inside_a_running_event_loop() -> None:
    """Regression: eager Celery runs the task inline inside the API's loop.

    `asyncio.run` refuses to nest, so dispatching an upload with
    CELERY_TASK_ALWAYS_EAGER=true used to raise "asyncio.run() cannot be called
    from a running event loop" and 500 the request after the paper was stored.
    """
    from app.worker.tasks import _run_with_session

    async def operation(session: AsyncSession) -> str:
        assert session is not None
        return "ran"

    # This test function is itself running inside an event loop.
    assert _run_with_session(operation) == "ran"
