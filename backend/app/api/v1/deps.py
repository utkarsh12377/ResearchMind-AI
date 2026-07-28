"""Shared FastAPI dependencies for authentication and DB access."""

from fastapi import Depends, Header
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import ForbiddenError, UnauthorizedError
from app.core.security import InvalidTokenError, decode_access_token
from app.db.session import get_db
from app.models import User
from app.services.api_key_service import get_user_by_api_key

# auto_error=False: a missing bearer token isn't fatal by itself, since an
# X-API-Key header is also a valid way to authenticate.
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/v1/auth/login", auto_error=False)


async def get_current_user(
    db: AsyncSession = Depends(get_db),
    token: str | None = Depends(oauth2_scheme),
    x_api_key: str | None = Header(default=None),
) -> User:
    if token is not None:
        try:
            payload = decode_access_token(token)
        except InvalidTokenError as exc:
            raise UnauthorizedError(str(exc)) from exc

        user = await db.get(User, payload.sub)
        if user is None:
            raise UnauthorizedError("User no longer exists")
        return user

    if x_api_key is not None:
        user = await get_user_by_api_key(db, x_api_key)
        if user is None:
            raise UnauthorizedError("Invalid or revoked API key")
        return user

    raise UnauthorizedError("Not authenticated")


async def get_current_active_user(user: User = Depends(get_current_user)) -> User:
    if not user.is_active:
        raise ForbiddenError("This account is inactive")
    return user


async def get_current_superuser(user: User = Depends(get_current_active_user)) -> User:
    if not user.is_superuser:
        raise ForbiddenError("This action requires superuser privileges")
    return user
