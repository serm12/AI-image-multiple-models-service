"""FastAPI adapters for Shopify SDK verification results."""

from fastapi import Request, Response
from shopify_app.types import RequestInput, Res


async def request_to_shopify(request: Request) -> RequestInput:
    # Do not parse/re-serialize JSON: HMAC verification needs the original body.
    return {
        "method": request.method,
        "headers": dict(request.headers),
        "url": str(request.url),
        "body": (await request.body()).decode("utf-8"),
    }


def shopify_response(response: Res) -> Response:
    """Preserve the SDK's status, body and security headers."""
    return Response(
        content=response.body,
        status_code=response.status,
        headers=response.headers,
    )
