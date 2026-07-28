"""User registration and authentication logic."""

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import ConflictError, UnauthorizedError
from app.core.security import hash_password, verify_password
from app.models import User, Workspace
from app.schemas.user import UserCreate


async def get_user_by_email(db: AsyncSession, email: str) -> User | None:
    return await db.scalar(select(User).where(User.email == email))


async def register_user(db: AsyncSession, data: UserCreate) -> User:
    if await get_user_by_email(db, data.email) is not None:
        raise ConflictError("A user with this email is already registered")

    user = User(
        email=data.email,
        hashed_password=hash_password(data.password),
        full_name=data.full_name,
    )
    db.add(user)
    await db.flush()

    # Every user needs somewhere to put papers; give them a default workspace
    # so paper upload (Milestone 5) has no chicken-and-egg dependency.
    db.add(Workspace(name="Personal", owner=user))

    await db.commit()
    await db.refresh(user)
    return user


async def authenticate_user(db: AsyncSession, email: str, password: str) -> User:
    user = await get_user_by_email(db, email)
    if user is None or not verify_password(password, user.hashed_password):
        raise UnauthorizedError("Incorrect email or password")
    if not user.is_active:
        raise UnauthorizedError("This account is inactive")
    return user
