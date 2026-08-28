import ipaddress
import secrets

from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from app.core.config import AppConfig


http_basic = HTTPBasic(auto_error=False)
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
