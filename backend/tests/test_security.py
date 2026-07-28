import uuid
from datetime import datetime, timedelta, timezone

import jwt
import pytest

from app.core.config import get_settings
from app.core.security import (
    InvalidTokenError,
    create_access_token,
    decode_access_token,
    generate_api_key,
    hash_api_key,
    hash_password,
    verify_password,
)


def test_password_hash_round_trip() -> None:
    hashed = hash_password("correct horse battery staple")

    assert hashed != "correct horse battery staple"
    assert verify_password("correct horse battery staple", hashed)
    assert not verify_password("wrong password", hashed)


def test_access_token_round_trip() -> None:
    user_id = uuid.uuid4()
    token = create_access_token(user_id)

    payload = decode_access_token(token)

    assert payload.sub == user_id


def test_expired_token_is_rejected() -> None:
    settings = get_settings()
    expired = jwt.encode(
        {"sub": str(uuid.uuid4()), "exp": datetime.now(timezone.utc) - timedelta(minutes=1)},
        settings.jwt_secret,
        algorithm=settings.jwt_algorithm,
    )

    with pytest.raises(InvalidTokenError):
        decode_access_token(expired)


def test_garbage_token_is_rejected() -> None:
    with pytest.raises(InvalidTokenError):
        decode_access_token("not-a-real-token")


def test_generate_api_key_is_unique_and_hash_is_deterministic() -> None:
    raw_1, prefix_1, hashed_1 = generate_api_key()
    raw_2, _prefix_2, hashed_2 = generate_api_key()

    assert raw_1 != raw_2
    assert hashed_1 != hashed_2
    assert raw_1.startswith(prefix_1)
    assert hash_api_key(raw_1) == hashed_1
