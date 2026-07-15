from __future__ import annotations

import hashlib
import hmac
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from fastapi import Depends, HTTPException, Request, status

from backend.config import settings
from backend.db.database import connect_database, timestamp_string


SESSION_COOKIE = "leaflight_session"
CSRF_COOKIE = "leaflight_csrf"
OAUTH_STATE_COOKIE = "leaflight_oauth_state"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def isoformat(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="seconds")


def _require_auth_secret() -> str:
    if len(settings.auth_secret) < 32:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication is not configured. Set AUTH_SECRET to at least 32 random characters.",
        )
    return settings.auth_secret


def hash_token(value: str) -> str:
    return hmac.new(
        _require_auth_secret().encode("utf-8"),
        value.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


@dataclass(frozen=True)
class AuthContext:
    user_id: str
    session_id: int
    csrf_token_hash: str
    expires_at: str
    name: str
    email: str
    profile_picture: str | None
    auth_provider: str
    created_at: str
    last_login_at: str

    def user_dict(self) -> dict[str, str | None]:
        return {
            "id": self.user_id,
            "name": self.name,
            "email": self.email,
            "profile_picture": self.profile_picture,
            "auth_provider": self.auth_provider,
            "created_at": self.created_at,
            "last_login_at": self.last_login_at,
        }


def create_session(user_id: str) -> tuple[str, str, str]:
    token = secrets.token_urlsafe(48)
    csrf_token = secrets.token_urlsafe(32)
    expires_at = isoformat(utc_now() + timedelta(hours=settings.session_ttl_hours))
    with connect_database() as connection:
        connection.execute(
            """
            INSERT INTO auth_sessions(user_id, token_hash, csrf_token_hash, expires_at)
            VALUES (?, ?, ?, ?)
            """,
            (user_id, hash_token(token), hash_token(csrf_token), expires_at),
        )
        connection.commit()
    return token, csrf_token, expires_at


def require_user(request: Request) -> AuthContext:
    token = request.cookies.get(SESSION_COOKIE)
    if not token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required.")
    now = isoformat(utc_now())
    with connect_database() as connection:
        row = connection.execute(
            """
            SELECT s.id AS session_id, s.csrf_token_hash, s.expires_at,
                   u.id AS user_id, u.name, u.email, u.profile_picture,
                   u.auth_provider, u.created_at, u.last_login_at
            FROM auth_sessions s
            JOIN users u ON u.id = s.user_id
            WHERE s.token_hash = ? AND s.revoked_at IS NULL AND s.expires_at > ?
            """,
            (hash_token(token), now),
        ).fetchone()
        if row:
            connection.execute(
                "UPDATE auth_sessions SET last_seen_at = ? WHERE id = ?",
                (now, row["session_id"]),
            )
            connection.commit()
    if not row:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Session is invalid or expired.")
    payload = dict(row)
    payload["user_id"] = str(payload["user_id"])
    for field in ("expires_at", "created_at", "last_login_at"):
        payload[field] = timestamp_string(payload[field])
    return AuthContext(**payload)


def require_csrf(
    request: Request,
    auth: AuthContext = Depends(require_user),
) -> AuthContext:
    token = request.headers.get("X-CSRF-Token", "")
    if not token or not hmac.compare_digest(hash_token(token), auth.csrf_token_hash):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="CSRF validation failed.")
    return auth
