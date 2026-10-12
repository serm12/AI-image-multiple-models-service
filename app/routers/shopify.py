"""Shopify app foundation. No store data is exposed by these routes."""
from urllib.parse import urlencode
import base64
import json
import re

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, RedirectResponse
from shopify_app import ShopifyApp

from app.services.shopify_client import get_shopify_app
from app.services.shopify_http import request_to_shopify, shopify_response
from app.services.shopify_admin import allowed_shop, set_session
from app.services.shopify_orders import sync_orders, receive_order_webhook
from app.services.shopify_order_images import get_order_images

router = APIRouter(prefix="/shopify", tags=["Shopify"])


def shopify_dependency() -> ShopifyApp:
    try:
        return get_shopify_app()
    except RuntimeError:
        raise HTTPException(status_code=503, detail="Shopify integration is not configured") from None


@router.get("", response_class=HTMLResponse, include_in_schema=False)
async def app_home(request: Request, sdk: ShopifyApp = Depends(shopify_dependency)):
    return await verified_session(request, sdk, redirect=True)


async def verified_session(request, sdk, redirect=False):
    result = sdk.verify_app_home_req(await request_to_shopify(request),
                                    app_home_patch_id_token_path="/shopify/patch-id-token")
    if not result.ok:
        return shopify_response(result.response)
    claims = result.id_token.claims
    if (result.shop != allowed_shop() or not result.user_id
            or claims.get("dest") != f"https://{allowed_shop()}.myshopify.com"
            or claims.get("iss") != f"https://{allowed_shop()}.myshopify.com/admin"
            or not isinstance(claims.get("exp"), int)):
        raise HTTPException(403, "Shop is not authorized")
    await sync_orders(sdk, result)
    context = urlencode({"shop": f"{allowed_shop()}.myshopify.com", "embedded": "1",
                         "host": base64.b64encode(b"admin.shopify.com/store/paintingify").decode().rstrip("=")})
    response = RedirectResponse("/admin/tasks?" + context, status_code=303) if redirect else Response(status_code=204)
    response.headers.update(result.response.headers)
    response.headers["Cache-Control"] = "no-store"
    response.headers["Referrer-Policy"] = "no-referrer"
    set_session(response, result.user_id)
    return response


@router.post("/session", include_in_schema=False)
async def refresh_session(request: Request, sdk: ShopifyApp = Depends(shopify_dependency)):
    if not request.headers.get("authorization", "").startswith("Bearer "):
        raise HTTPException(401, "Shopify session token required")
    return await verified_session(request, sdk)


@router.get("/patch-id-token", include_in_schema=False)
async def patch_id_token(request: Request, sdk: ShopifyApp = Depends(shopify_dependency)):
    if request.query_params.get("shop") != f"{allowed_shop()}.myshopify.com":
        raise HTTPException(403, "Shop is not authorized")
    if request.query_params.get("shopify-reload") != "/shopify" and not request.query_params.get("shopify-reload", "").startswith("/shopify?"):
        raise HTTPException(400, "Invalid reload path")
    response = shopify_response(sdk.app_home_patch_id_token(await request_to_shopify(request)).response)
    response.headers["Cache-Control"] = "no-store"
    return response


@router.post("/webhooks", include_in_schema=False)
async def webhooks(request: Request, sdk: ShopifyApp = Depends(shopify_dependency)):
    try:
        incoming = await request_to_shopify(request)
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="Invalid UTF-8 body") from None
    result = sdk.verify_webhook_req(incoming)
    if not result.ok:
        return shopify_response(result.response)

    if request.headers.get('x-shopify-shop-domain') != f'{allowed_shop()}.myshopify.com':
        raise HTTPException(403, 'Shop is not authorized')
    if request.headers.get('x-shopify-topic') in {'orders/create', 'orders/paid', 'orders/updated'}:
        try:
            receive_order_webhook(json.loads(incoming['body']), request.headers.get('x-shopify-event-id') or request.headers.get('x-shopify-webhook-id'))
        except (ValueError, TypeError, KeyError, AttributeError):
            raise HTTPException(400, 'Invalid order payload') from None
        return Response(status_code=200)

    if request.headers.get("x-shopify-topic") == "app/uninstalled":
        # No shop sessions/tokens are persisted yet. Add token cleanup here when
        # implementing storage, before enabling store authorization in production.
        return Response(status_code=200)
    # Never acknowledge unimplemented order/privacy handlers as processed.
    raise HTTPException(status_code=404, detail="Unsupported webhook topic")


@router.get('/orders/{order_id}/images', include_in_schema=False)
async def order_images(order_id: str, request: Request, sdk: ShopifyApp = Depends(shopify_dependency)):
    if not re.fullmatch(r'[0-9]{1,30}', order_id):
        raise HTTPException(400, 'Invalid order ID')
    verified = sdk.verify_admin_ui_ext_req(await request_to_shopify(request))
    if not verified.ok:
        return shopify_response(verified.response)
    claims = verified.id_token.claims
    if (verified.shop != allowed_shop() or not verified.user_id
            or claims.get('dest') != f'https://{allowed_shop()}.myshopify.com'
            or not isinstance(claims.get('exp'), int)):
        raise HTTPException(403, 'Shop is not authorized')
    from fastapi.responses import JSONResponse
    return JSONResponse(await get_order_images(sdk, verified, order_id), headers={'Cache-Control': 'no-store'})
