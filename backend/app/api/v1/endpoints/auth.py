import uuid

from fastapi import APIRouter, Depends, Request, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.v1.deps import get_current_active_user
from app.core.rate_limit import limiter
from app.core.security import create_access_token
from app.db.session import get_db
from app.models import ApiKey, User
from app.schemas.api_key import ApiKeyCreate, ApiKeyCreated, ApiKeyRead
from app.schemas.user import Token, UserCreate, UserRead
from app.services.api_key_service import create_api_key, list_api_keys, revoke_api_key
from app.services.user_service import authenticate_user, register_user

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=UserRead, status_code=status.HTTP_201_CREATED)
@limiter.limit("5/minute")
async def register(
    request: Request, data: UserCreate, db: AsyncSession = Depends(get_db)
) -> User:
    return await register_user(db, data)


@router.post("/login", response_model=Token)
@limiter.limit("5/minute")
async def login(
    request: Request,
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: AsyncSession = Depends(get_db),
) -> Token:
    user = await authenticate_user(db, form_data.username, form_data.password)
    return Token(access_token=create_access_token(user.id))


@router.get("/me", response_model=UserRead)
async def read_current_user(user: User = Depends(get_current_active_user)) -> User:
    return user


@router.post("/api-keys", response_model=ApiKeyCreated, status_code=status.HTTP_201_CREATED)
async def create_key(
    data: ApiKeyCreate,
    user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> ApiKeyCreated:
    api_key, raw_key = await create_api_key(db, user, data)
    return ApiKeyCreated(
        id=api_key.id,
        name=api_key.name,
        prefix=api_key.prefix,
        created_at=api_key.created_at,
        last_used_at=api_key.last_used_at,
        revoked_at=api_key.revoked_at,
        api_key=raw_key,
    )


@router.get("/api-keys", response_model=list[ApiKeyRead])
async def list_keys(
    user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> list[ApiKey]:
    return await list_api_keys(db, user)


@router.delete("/api-keys/{key_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_key(
    key_id: uuid.UUID,
    user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> None:
    await revoke_api_key(db, user, key_id)
