import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Paper, PaperStatus, User, Workspace


@pytest.mark.asyncio
async def test_create_user_workspace_paper_relationships(db_session: AsyncSession) -> None:
    user = User(email="researcher@example.com", hashed_password="hashed")
    db_session.add(user)
    await db_session.flush()

    workspace = Workspace(name="Graph RAG survey", owner=user)
    db_session.add(workspace)
    await db_session.flush()

    paper = Paper(workspace=workspace, uploaded_by=user, title="A Survey of Graph RAG")
    db_session.add(paper)
    await db_session.commit()

    fetched = await db_session.scalar(select(Paper).where(Paper.id == paper.id))
    assert fetched is not None
    assert fetched.title == "A Survey of Graph RAG"
    assert fetched.status == PaperStatus.PENDING
    assert fetched.workspace_id == workspace.id
    assert fetched.uploaded_by_id == user.id


@pytest.mark.asyncio
async def test_user_email_must_be_unique(db_session: AsyncSession) -> None:
    db_session.add(User(email="dup@example.com", hashed_password="a"))
    await db_session.commit()

    db_session.add(User(email="dup@example.com", hashed_password="b"))
    with pytest.raises(IntegrityError):
        await db_session.commit()


@pytest.mark.asyncio
async def test_deleting_workspace_cascades_to_papers(db_session: AsyncSession) -> None:
    user = User(email="owner@example.com", hashed_password="hashed")
    workspace = Workspace(name="Temp workspace", owner=user)
    paper = Paper(workspace=workspace, uploaded_by=user, title="Some paper")
    db_session.add_all([user, workspace, paper])
    await db_session.commit()
    paper_id = paper.id

    await db_session.delete(workspace)
    await db_session.commit()

    assert await db_session.scalar(select(Paper).where(Paper.id == paper_id)) is None
