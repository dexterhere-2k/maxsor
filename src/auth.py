from __future__ import annotations

import datetime as dt
from typing import Any

import bcrypt
import jwt
from fastapi import Header, HTTPException, status

from . import config, database

_BCRYPT_MAX_BYTES = 72

def _prepare(password: str) -> bytes:
    return password.encode("utf-8")[:_BCRYPT_MAX_BYTES]

def hash_password(password: str) -> str:
    return bcrypt.hashpw(_prepare(password), bcrypt.gensalt()).decode("ascii")

def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(_prepare(password), password_hash.encode("ascii"))
    except ValueError:
        return False

def create_token(user_id: int, ttl_hours: int | None = None) -> str:
    hours = config.TOKEN_TTL_HOURS if ttl_hours is None else ttl_hours
    now = dt.datetime.now(dt.timezone.utc)
    payload = {
        "sub": str(user_id),
        "iat": int(now.timestamp()),
        "exp": int((now + dt.timedelta(hours=hours)).timestamp()),
    }
    return jwt.encode(payload, config.JWT_SECRET, algorithm=config.JWT_ALGORITHM)

def _unauthorized(detail: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"},
    )

def decode_token(token: str) -> int:
    try:
        payload = jwt.decode(token, config.JWT_SECRET, algorithms=[config.JWT_ALGORITHM])
    except jwt.ExpiredSignatureError as exc:
        raise _unauthorized("Token has expired") from exc
    except jwt.InvalidTokenError as exc:
        raise _unauthorized("Token is invalid") from exc

    try:
        return int(payload["sub"])
    except (KeyError, TypeError, ValueError) as exc:
        raise _unauthorized("Token is missing a usable subject") from exc

def current_user(authorization: str | None = Header(default=None)) -> dict[str, Any]:
    if not authorization:
        raise _unauthorized("Not authenticated")

    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise _unauthorized("Expected an 'Authorization: Bearer <token>' header")

    user = database.get_user_by_id(decode_token(token.strip()))
    if user is None:
        raise _unauthorized("Not authenticated")
    return user

if __name__ == "__main__":
    digest = hash_password("correct-horse-battery")
    assert verify_password("correct-horse-battery", digest)
    assert not verify_password("correct-horse-battery ", digest)
    assert digest != hash_password("correct-horse-battery"), "each hash needs its own salt"
    assert decode_token(create_token(7)) == 7, "a token must round-trip to its user id"
    print("hashing is salted and the token round-trips")
