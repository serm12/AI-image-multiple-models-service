"""Shopify app foundation. No store data is exposed by these routes."""

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import HTMLResponse
from shopify_app import ShopifyApp

from app.services.shopify_client import get_shopify_app
from app.services.shopify_http import request_to_shopify, shopify_response

router = APIRouter(prefix="/shopify", tags=["Shopify"])


def shopify_dependency() -> ShopifyApp:
    try:
        return get_shopify_app()
    except RuntimeError:
        raise HTTPException(status_code=503, detail="Shopify integration is not configured") from None


@router.get("", response_class=HTMLResponse, include_in_schema=False)
async def app_home():
    return HTMLResponse(
        '<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>Paintingify AI Backend</title><body>'
        '<h1>Paintingify AI Backend</h1><p>Paintingify 图片服务应用。</p>'
        '</body></html>',
        headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"},
    )


@router.post("/webhooks", include_in_schema=False)
async def webhooks(request: Request, sdk: ShopifyApp = Depends(shopify_dependency)):
    try:
        incoming = await request_to_shopify(request)
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="Invalid UTF-8 body") from None
    result = sdk.verify_webhook_req(incoming)
    if not result.ok:
        return shopify_response(result.response)

    if request.headers.get("x-shopify-topic") == "app/uninstalled":
        # No shop sessions/tokens are persisted yet. Add token cleanup here when
        # implementing storage, before enabling store authorization in production.
        return Response(status_code=200)
    # Never acknowledge unimplemented order/privacy handlers as processed.
    raise HTTPException(status_code=404, detail="Unsupported webhook topic")
