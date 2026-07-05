"""Password hashing and JWT helpers."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import bcrypt
from jose import JWTError, jwt

from src.utils.env import get_jwt_secret

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60
REMEMBER_ME_EXPIRE_DAYS = 30


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return bcrypt.checkpw(
        plain_password.encode("utf-8"),
        hashed_password.encode("utf-8"),
    )


def create_access_token(*, subject: str, remember_me: bool = False) -> tuple[str, int]:
    if remember_me:
        expires_delta = timedelta(days=REMEMBER_ME_EXPIRE_DAYS)
    else:
        expires_delta = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    expires_at = datetime.now(UTC) + expires_delta
    payload = {"sub": subject, "exp": expires_at}
    token = jwt.encode(payload, get_jwt_secret(), algorithm=ALGORITHM)
    return token, int(expires_delta.total_seconds())


def decode_access_token(token: str) -> dict[str, Any]:
    return jwt.decode(token, get_jwt_secret(), algorithms=[ALGORITHM])


def token_subject(token: str) -> str:
    try:
        payload = decode_access_token(token)
    except JWTError as exc:
        raise ValueError("Invalid or expired token.") from exc
    subject = payload.get("sub")
    if not subject:
        raise ValueError("Invalid token payload.")
    return str(subject)
