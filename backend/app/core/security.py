"""Password hashing, JWT access tokens, and API key generation/hashing."""

import hashlib
import secrets
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import bcrypt
import jwt

from app.core.config import get_settings

settings = get_settings()

API_KEY_PREFIX = "rma_"


def hash_password(plain_password: str) -> str:
    return bcrypt.hashpw(plain_password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return bcrypt.checkpw(plain_password.encode("utf-8"), hashed_password.encode("utf-8"))


@dataclass
class TokenPayload:
    sub: uuid.UUID
    exp: datetime


class InvalidTokenError(Exception):
    pass


def create_access_token(user_id: uuid.UUID) -> str:
    expires_in = timedelta(minutes=settings.jwt_access_token_expire_minutes)
    expire = datetime.now(timezone.utc) + expires_in
    payload = {"sub": str(user_id), "exp": expire}
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def generate_api_key() -> tuple[str, str, str]:
    """Create a new API key.

    Returns (raw_key, prefix, hashed_key). Only `hashed_key` should ever be
    persisted; `raw_key` is shown to the user once and cannot be recovered.
    `prefix` is a short, non-secret slice stored alongside the hash so users
    can identify a key in a list without re-exposing the secret.
    """
    raw_key = f"{API_KEY_PREFIX}{secrets.token_urlsafe(32)}"
    prefix = raw_key[: len(API_KEY_PREFIX) + 8]
    return raw_key, prefix, hash_api_key(raw_key)


def hash_api_key(raw_key: str) -> str:
    # API keys are high-entropy random tokens (not user-chosen passwords), so
    # a fast, unsalted digest is sufficient for exact-match lookup and avoids
    # needing to scan/verify against every stored key on each request.
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


def decode_access_token(token: str) -> TokenPayload:
    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
    except jwt.PyJWTError as exc:
        raise InvalidTokenError("Invalid or expired token") from exc

    try:
        return TokenPayload(
            sub=uuid.UUID(payload["sub"]),
            exp=datetime.fromtimestamp(payload["exp"], tz=timezone.utc),
        )
    except (KeyError, ValueError) as exc:
        raise InvalidTokenError("Malformed token payload") from exc
