import ipaddress
import base64
import binascii
import hashlib
import hmac
import json
import secrets
import time
from urllib.parse import urlsplit, urlunsplit

from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from app.core.config import AppConfig


http_basic = HTTPBasic(auto_error=False)
ADMIN_SESSION_COOKIE = "ai_image_admin_session"
ADMIN_SESSION_MAX_AGE = 60 * 60 * 24 * 30
TRUSTED_PROXY_IPS = {"127.0.0.1", "::1"}
CLOUDFLARE_NETWORKS = tuple(
    ipaddress.ip_network(cidr)
    for cidr in (
        "173.245.48.0/20",
        "103.21.244.0/22",
        "103.22.200.0/22",
        "103.31.4.0/22",
        "141.101.64.0/18",
        "108.162.192.0/18",
        "190.93.240.0/20",
        "188.114.96.0/20",
        "197.234.240.0/22",
        "198.41.128.0/17",
        "162.158.0.0/15",
        "104.16.0.0/13",
        "104.24.0.0/14",
        "172.64.0.0/13",
        "131.0.72.0/22",
        "2400:cb00::/32",
        "2606:4700::/32",
        "2803:f800::/32",
        "2405:b500::/32",
        "2405:8100::/32",
        "2a06:98c0::/29",
        "2c0f:f248::/32",
    )
)


def _is_trusted_proxy(peer_ip: str) -> bool:
    if peer_ip in TRUSTED_PROXY_IPS:
        return True
    try:
        address = ipaddress.ip_address(peer_ip)
    except ValueError:
        return False
    return any(address in network for network in CLOUDFLARE_NETWORKS)


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


def validate_admin_credentials(username: str, password: str) -> bool:
    """Compare administrator credentials in constant time."""
    user_matches = secrets.compare_digest(
        username.encode("utf-8"), AppConfig.ADMIN_USER.encode("utf-8")
    )
    password_matches = secrets.compare_digest(
        password.encode("utf-8"), AppConfig.ADMIN_PASSWORD.encode("utf-8")
    )
    return bool(AppConfig.ADMIN_USER and AppConfig.ADMIN_PASSWORD) and user_matches and password_matches


def _admin_session_secret() -> bytes:
    # The fallback keeps existing deployments working; a dedicated secret is preferred.
    configured = AppConfig.ADMIN_SESSION_SECRET
    material = configured or f"{AppConfig.ADMIN_USER}\0{AppConfig.ADMIN_PASSWORD}"
    return hashlib.sha256(material.encode("utf-8")).digest()


def create_admin_session(username: str, max_age: int = ADMIN_SESSION_MAX_AGE) -> str:
    payload = json.dumps(
        {"u": username, "e": int(time.time()) + max_age}, separators=(",", ":")
    ).encode("utf-8")
    encoded = base64.urlsafe_b64encode(payload).rstrip(b"=")
    signature = hmac.new(_admin_session_secret(), encoded, hashlib.sha256).digest()
    return f"{encoded.decode('ascii')}.{base64.urlsafe_b64encode(signature).rstrip(b'=').decode('ascii')}"


def get_admin_session_user(token: str | None) -> str | None:
    if not token or "." not in token:
        return None
    encoded, signature = token.split(".", 1)
    try:
        encoded_bytes = encoded.encode("ascii")
        expected = hmac.new(_admin_session_secret(), encoded_bytes, hashlib.sha256).digest()
        actual = base64.urlsafe_b64decode(signature + "=" * (-len(signature) % 4))
        payload = json.loads(base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4)))
    except (ValueError, UnicodeEncodeError, json.JSONDecodeError, binascii.Error):
        return None
    if not hmac.compare_digest(actual, expected):
        return None
    if payload.get("u") != AppConfig.ADMIN_USER or not isinstance(payload.get("e"), int):
        return None
    return AppConfig.ADMIN_USER if payload["e"] >= int(time.time()) else None


def require_admin_login(
    request: Request,
    credentials: HTTPBasicCredentials | None = Depends(http_basic),
):
    """Protect browser admin pages with a signed session, accepting legacy Basic Auth."""
    session_user = get_admin_session_user(request.cookies.get(ADMIN_SESSION_COOKIE))
    if session_user:
        return session_user
    if credentials and validate_admin_credentials(credentials.username, credentials.password):
        return credentials.username
    target = request.url.path
    if request.url.query:
        target = f"{target}?{request.url.query}"
    raise HTTPException(status_code=303, headers={"Location": f"/admin/login?next={target}"})


def get_request_client_ip(request: Request) -> str:
    """Return the visitor IP without trusting forwarded headers from public clients."""
    peer_ip = request.client.host if request.client else ""
    if not _is_trusted_proxy(peer_ip):
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


def get_request_country(request: Request) -> str:
    """Return Cloudflare's ISO country code only for requests from a trusted proxy."""
    peer_ip = request.client.host if request.client else ""
    if not _is_trusted_proxy(peer_ip):
        return ""
    country = request.headers.get("cf-ipcountry", "").strip().upper()
    if len(country) == 2 and country.isalpha():
        return country
    return ""


def get_request_source_page(request: Request, submitted_url: str | None = None) -> str:
    """Return a safe storefront page URL, preferring the explicit form value."""
    candidates = (submitted_url, request.headers.get("referer"))
    for candidate in candidates:
        value = str(candidate or "").strip()
        if not value or len(value) > 2048:
            continue
        try:
            parsed = urlsplit(value)
            if (
                parsed.scheme.lower() not in {"http", "https"}
                or not parsed.hostname
                or parsed.username
                or parsed.password
            ):
                continue
            # Fragments are browser-local and may contain sensitive application state.
            return urlunsplit(
                (parsed.scheme.lower(), parsed.netloc, parsed.path or "/", parsed.query, "")
            )
        except ValueError:
            continue
    return ""
