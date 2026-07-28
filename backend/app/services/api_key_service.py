"""API key issuance, lookup, and revocation."""

import uuid
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import NotFoundError
from app.core.security import generate_api_key, hash_api_key
from app.models import ApiKey, User
from app.schemas.api_key import ApiKeyCreate


async def create_api_key(db: AsyncSession, user: User, data: ApiKeyCreate) -> tuple[ApiKey, str]:
    raw_key, prefix, hashed_key = generate_api_key()
    api_key = ApiKey(user_id=user.id, name=data.name, prefix=prefix, hashed_key=hashed_key)
    db.add(api_key)
    await db.commit()
    await db.refresh(api_key)
    return api_key, raw_key


async def list_api_keys(db: AsyncSession, user: User) -> list[ApiKey]:
    result = await db.scalars(
        select(ApiKey).where(ApiKey.user_id == user.id).order_by(ApiKey.created_at.desc())
    )
    return list(result)


async def revoke_api_key(db: AsyncSession, user: User, api_key_id: uuid.UUID) -> None:
    api_key = await db.scalar(
        select(ApiKey).where(ApiKey.id == api_key_id, ApiKey.user_id == user.id)
    )
    if api_key is None:
        raise NotFoundError("API key not found")

    api_key.revoked_at = datetime.now(timezone.utc)
    await db.commit()


async def get_user_by_api_key(db: AsyncSession, raw_key: str) -> User | None:
    hashed_key = hash_api_key(raw_key)
    api_key = await db.scalar(select(ApiKey).where(ApiKey.hashed_key == hashed_key))

    if api_key is None or api_key.revoked_at is not None:
        return None

    api_key.last_used_at = datetime.now(timezone.utc)
    await db.commit()

    return await db.get(User, api_key.user_id)
