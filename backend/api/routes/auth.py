from __future__ import annotations

import logging
import secrets
from datetime import timedelta
from urllib.parse import urlencode
from uuid import uuid4

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, status
from fastapi.responses import RedirectResponse

from backend.api.auth import (
    CSRF_COOKIE,
    OAUTH_STATE_COOKIE,
    SESSION_COOKIE,
    AuthContext,
    create_session,
    hash_token,
    isoformat,
    require_csrf,
    require_user,
    utc_now,
)
from backend.api.schemas import AuthConfigResponse, SessionResponse, UserResponse
from backend.config import settings
from backend.db.database import connect_database, database_error_types, timestamp_string


router = APIRouter(prefix="/auth", tags=["authentication"])
GOOGLE_AUTHORIZE_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_URL = "https://openidconnect.googleapis.com/v1/userinfo"
logger = logging.getLogger(__name__)


def _cookie_options(cookie_name: str) -> dict:
    return {
        "domain": settings.cookie_domain,
        "httponly": cookie_name != CSRF_COOKIE,
        "secure": settings.cookie_secure,
        "samesite": settings.cookie_samesite,
        "path": "/",
    }


def _set_cookie(response: Response, cookie_name: str, value: str, max_age: int) -> None:
    response.set_cookie(
        cookie_name,
        value,
        max_age=max_age,
        **_cookie_options(cookie_name),
    )


def _delete_cookie(response: Response, cookie_name: str) -> None:
    response.delete_cookie(cookie_name, **_cookie_options(cookie_name))


def _frontend_redirect(return_to: str = "/", **query: str) -> RedirectResponse:
    safe_return_to = _safe_return_to(return_to)
    separator = "&" if "?" in safe_return_to else "?"
    location = f"{settings.app_url}{safe_return_to}{separator}{urlencode(query)}"
    return RedirectResponse(location, status_code=302)


def _oauth_error_redirect(error_code: str) -> RedirectResponse:
    redirect = _frontend_redirect(auth_error=error_code)
    _delete_cookie(redirect, OAUTH_STATE_COOKIE)
    return redirect


def _require_google_configuration() -> None:
    if not settings.google_auth_configured or len(settings.auth_secret) < 32:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Google authentication is not configured. Set GOOGLE_CLIENT_ID, "
                "GOOGLE_CLIENT_SECRET, AUTH_SECRET, APP_URL, and OAUTH_CALLBACK_URL."
            ),
        )


def _safe_return_to(value: str) -> str:
    if value.startswith("/") and not value.startswith("//") and "\\" not in value:
        return value[:500]
    return "/"


async def _fetch_google_user(code: str) -> dict:
    timeout = httpx.Timeout(connect=10, read=15, write=10, pool=10)
    async with httpx.AsyncClient(timeout=timeout) as client:
        token_response = await client.post(
            GOOGLE_TOKEN_URL,
            data={
                "client_id": settings.google_client_id,
                "client_secret": settings.google_client_secret,
                "code": code,
                "grant_type": "authorization_code",
                "redirect_uri": settings.oauth_callback_url,
            },
        )
        token_response.raise_for_status()
        access_token = token_response.json().get("access_token")
        if not access_token:
            raise ValueError("Google token response did not contain an access token.")
        user_response = await client.get(
            GOOGLE_USERINFO_URL,
            headers={"Authorization": f"Bearer {access_token}"},
        )
        user_response.raise_for_status()
        return user_response.json()


def _upsert_google_user(profile: dict) -> str:
    provider_id = str(profile.get("sub", "")).strip()
    email = str(profile.get("email", "")).strip().lower()
    name = str(profile.get("name", "")).strip()
    picture = str(profile.get("picture", "")).strip() or None
    if not provider_id or not email or not name or profile.get("email_verified") is not True:
        raise HTTPException(status_code=400, detail="Google did not return a verified identity profile.")
    now = isoformat(utc_now())
    with connect_database() as connection:
        existing = connection.execute(
            "SELECT id, provider_account_id FROM users WHERE LOWER(email) = LOWER(?)",
            (email,),
        ).fetchone()
        if existing and existing["provider_account_id"] != provider_id:
            raise HTTPException(status_code=409, detail="An account already exists for this email.")
        if existing:
            user_id = str(existing["id"])
            connection.execute(
                """
                UPDATE users
                SET name = ?, profile_picture = ?, last_login_at = ?
                WHERE id = ?
                """,
                (name, picture, now, user_id),
            )
        else:
            user_id = str(uuid4())
            connection.execute(
                """
                INSERT INTO users(
                    id, name, email, profile_picture, auth_provider,
                    provider_account_id, created_at, last_login_at
                ) VALUES (?, ?, ?, ?, 'google', ?, ?, ?)
                """,
                (user_id, name, email, picture, provider_id, now, now),
            )
        connection.commit()
    return user_id


@router.get("/config", response_model=AuthConfigResponse)
def auth_config() -> AuthConfigResponse:
    configured = settings.google_auth_configured and len(settings.auth_secret) >= 32
    return AuthConfigResponse(
        configured=configured,
        callback_url=settings.oauth_callback_url if configured else None,
    )


@router.get("/google/login")
def google_login(
    request: Request,
    return_to: str = Query(default="/"),
):
    _require_google_configuration()
    previous_state = request.cookies.get(OAUTH_STATE_COOKIE, "")
    state_value = secrets.token_urlsafe(40)
    now = utc_now()
    expires_at = isoformat(now + timedelta(minutes=10))
    with connect_database() as connection:
        connection.execute("DELETE FROM oauth_states WHERE expires_at <= ?", (isoformat(now),))
        if previous_state:
            connection.execute(
                "DELETE FROM oauth_states WHERE state_hash = ?",
                (hash_token(previous_state),),
            )
        connection.execute(
            "INSERT INTO oauth_states(state_hash, return_to, expires_at) VALUES (?, ?, ?)",
            (hash_token(state_value), _safe_return_to(return_to), expires_at),
        )
        connection.commit()
    logger.info(
        "oauth_state_issue route=/auth/google/login result=created previous_state_cookie_present=%s",
        bool(previous_state),
    )
    parameters = urlencode(
        {
            "client_id": settings.google_client_id,
            "redirect_uri": settings.oauth_callback_url,
            "response_type": "code",
            "scope": "openid email profile",
            "state": state_value,
            "access_type": "online",
            "prompt": "select_account",
        }
    )
    redirect = RedirectResponse(f"{GOOGLE_AUTHORIZE_URL}?{parameters}", status_code=302)
    _set_cookie(redirect, OAUTH_STATE_COOKIE, state_value, max_age=600)
    return redirect


@router.get("/google/callback")
async def google_callback(
    request: Request,
    code: str | None = None,
    state: str | None = None,
    error: str | None = None,
):
    _require_google_configuration()
    cookie_state = request.cookies.get(OAUTH_STATE_COOKIE, "")
    state_matches_cookie = bool(
        state and cookie_state and secrets.compare_digest(state, cookie_state)
    )
    if not state_matches_cookie:
        if cookie_state:
            with connect_database() as connection:
                connection.execute(
                    "DELETE FROM oauth_states WHERE state_hash = ?",
                    (hash_token(cookie_state),),
                )
                connection.commit()
        logger.warning(
            "oauth_state_validation route=/auth/google/callback result=mismatch "
            "state_present=%s oauth_state_cookie_present=%s",
            bool(state),
            bool(cookie_state),
        )
        return _oauth_error_redirect("state")

    state_hash = hash_token(state)
    now = isoformat(utc_now())
    with connect_database() as connection:
        state_row = connection.execute(
            "DELETE FROM oauth_states WHERE state_hash = ? RETURNING return_to, expires_at",
            (state_hash,),
        ).fetchone()
        connection.commit()
    state_is_valid = bool(
        state_row and (timestamp_string(state_row["expires_at"]) or "") > now
    )
    logger.info(
        "oauth_state_validation route=/auth/google/callback result=%s "
        "state_present=true oauth_state_cookie_present=true",
        "accepted" if state_is_valid else "expired_or_consumed",
    )
    if not state_is_valid:
        return _oauth_error_redirect("state")
    if error:
        logger.info("oauth_callback route=/auth/google/callback result=provider_cancelled")
        return _oauth_error_redirect("cancelled")
    if not code:
        logger.warning("oauth_callback route=/auth/google/callback result=missing_code")
        return _oauth_error_redirect("missing_code")
    try:
        profile = await _fetch_google_user(code)
        user_id = _upsert_google_user(profile)
        session_token, csrf_token, _ = create_session(user_id)
    except HTTPException as exc:
        logger.warning(
            "oauth_callback route=/auth/google/callback result=identity_rejected status=%s",
            exc.status_code,
        )
        return _oauth_error_redirect("account")
    except (httpx.HTTPError, ValueError, *database_error_types()):
        logger.warning(
            "oauth_callback route=/auth/google/callback result=provider_or_session_failure"
        )
        return _oauth_error_redirect("provider")
    redirect = _frontend_redirect(state_row["return_to"], auth="success")
    max_age = settings.session_ttl_hours * 3600
    _set_cookie(redirect, SESSION_COOKIE, session_token, max_age=max_age)
    _set_cookie(redirect, CSRF_COOKIE, csrf_token, max_age=max_age)
    _delete_cookie(redirect, OAUTH_STATE_COOKIE)
    logger.info(
        "oauth_callback route=/auth/google/callback result=success session_creation=true"
    )
    return redirect


@router.get("/session", response_model=SessionResponse)
def session(auth: AuthContext = Depends(require_user)) -> SessionResponse:
    return SessionResponse(
        user=UserResponse(**auth.user_dict()),
        expires_at=auth.expires_at,
    )


@router.post("/logout", status_code=204)
def logout(
    request: Request,
    response: Response,
    auth: AuthContext = Depends(require_csrf),
) -> Response:
    oauth_state = request.cookies.get(OAUTH_STATE_COOKIE, "")
    with connect_database() as connection:
        revocation = connection.execute(
            "UPDATE auth_sessions SET revoked_at = ? WHERE id = ? AND revoked_at IS NULL",
            (isoformat(utc_now()), auth.session_id),
        )
        if oauth_state:
            connection.execute(
                "DELETE FROM oauth_states WHERE state_hash = ?",
                (hash_token(oauth_state),),
            )
        connection.commit()
    _delete_cookie(response, SESSION_COOKIE)
    _delete_cookie(response, CSRF_COOKIE)
    _delete_cookie(response, OAUTH_STATE_COOKIE)
    logger.info(
        "session_logout route=/auth/logout result=%s session_cookie_present=%s "
        "csrf_cookie_present=%s oauth_state_cookie_present=%s",
        "revoked" if revocation.rowcount == 1 else "already_revoked",
        SESSION_COOKIE in request.cookies,
        CSRF_COOKIE in request.cookies,
        bool(oauth_state),
    )
    response.status_code = 204
    return response
