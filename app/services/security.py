from fastapi import HTTPException, Request

from app.core.config import AppConfig


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
