import ipaddress
import secrets

from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from app.core.config import AppConfig


http_basic = HTTPBasic(auto_error=False)
TRUSTED_PROXY_IPS = {"127.0.0.1", "::1"}


def get_bearer_token(authorization: str | None) -> str | None:
    if not authorization:
        return None
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        return None
    return token.strip()


async def require_admin_api_key(request: Request):
    """Protect administrative endpoints when ADMIN_API_KEY is configured."""
    if not AppConfig.ADMIN_API_KEY:
        return
    provided_key = request.headers.get("x-api-key") or get_bearer_token(
        request.headers.get("authorization")
    )
    if provided_key != AppConfig.ADMIN_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing admin API key")


def require_admin_login(
    credentials: HTTPBasicCredentials | None = Depends(http_basic),
):
    """Protect the browser admin page with constant-time Basic Auth checks."""
    provided_user = credentials.username if credentials else ""
    provided_password = credentials.password if credentials else ""
    user_matches = secrets.compare_digest(
        provided_user.encode("utf-8"), AppConfig.ADMIN_USER.encode("utf-8")
    )
    password_matches = secrets.compare_digest(
        provided_password.encode("utf-8"), AppConfig.ADMIN_PASSWORD.encode("utf-8")
    )
    valid = bool(AppConfig.ADMIN_USER and AppConfig.ADMIN_PASSWORD)
    valid = valid and user_matches and password_matches
    if not valid:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing administrator credentials",
            headers={"WWW-Authenticate": 'Basic realm="AI Image Tasks"'},
        )
    return credentials.username


def get_request_client_ip(request: Request) -> str:
    """Return the visitor IP without trusting forwarded headers from public clients."""
    peer_ip = request.client.host if request.client else ""
    if peer_ip not in TRUSTED_PROXY_IPS:
        return peer_ip or "unknown"

    candidates = [request.headers.get("cf-connecting-ip", "")]
    candidates.extend(request.headers.get("x-forwarded-for", "").split(","))
    for candidate in candidates:
        value = candidate.strip()
        try:
            return str(ipaddress.ip_address(value))
        except ValueError:
            continue
    return peer_ip or "unknown"
