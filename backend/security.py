from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID

from fastapi import Cookie, Depends, HTTPException, Response, status
from jose import JWTError, jwt

from .config import settings

# Two separate cookies so signing in as admin doesn't log the user out of chat
# (and vice versa). A single browser can hold both sessions at the same time.
USER_COOKIE_NAME = "user_session"
ADMIN_COOKIE_NAME = "admin_session"


def create_jwt(claims: dict[str, Any]) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=settings.JWT_EXPIRE_MINUTES)
    return jwt.encode({**claims, "exp": expire}, settings.JWT_SECRET, algorithm=settings.JWT_ALGORITHM)


def decode_jwt(token: str) -> dict[str, Any]:
    return jwt.decode(token, settings.JWT_SECRET, algorithms=[settings.JWT_ALGORITHM])


def _set_cookie(resp: Response, name: str, token: str) -> None:
    resp.set_cookie(
        key=name,
        value=token,
        httponly=True,
        samesite="lax",
        secure=False,
        max_age=settings.JWT_EXPIRE_MINUTES * 60,
        path="/",
    )


def set_user_session_cookie(resp: Response, token: str) -> None:
    _set_cookie(resp, USER_COOKIE_NAME, token)


def set_admin_session_cookie(resp: Response, token: str) -> None:
    _set_cookie(resp, ADMIN_COOKIE_NAME, token)


def clear_user_session_cookie(resp: Response) -> None:
    resp.delete_cookie(USER_COOKIE_NAME, path="/")


def clear_admin_session_cookie(resp: Response) -> None:
    resp.delete_cookie(ADMIN_COOKIE_NAME, path="/")


def verify_admin_password(password: str) -> bool:
    return password == settings.ADMIN_PASSWORD


def _unauthorized() -> HTTPException:
    return HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="not authenticated")


def _forbidden() -> HTTPException:
    return HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="forbidden")


def _decode_or_none(token: str | None) -> dict | None:
    if not token:
        return None
    try:
        return decode_jwt(token)
    except JWTError:
        return None


def optional_user_claims(
    token: str | None = Cookie(default=None, alias=USER_COOKIE_NAME),
) -> dict | None:
    claims = _decode_or_none(token)
    if claims and claims.get("type") == "user":
        return claims
    return None


def optional_admin_claims(
    token: str | None = Cookie(default=None, alias=ADMIN_COOKIE_NAME),
) -> dict | None:
    claims = _decode_or_none(token)
    if claims and claims.get("type") == "admin":
        return claims
    return None


def require_user(claims: dict | None = Depends(optional_user_claims)) -> dict:
    if claims is None:
        raise _unauthorized()
    return claims


def require_admin(claims: dict | None = Depends(optional_admin_claims)) -> dict:
    if claims is None:
        raise _unauthorized()
    return claims


def user_id_from_claims(claims: dict) -> UUID:
    return UUID(claims["user_id"])
