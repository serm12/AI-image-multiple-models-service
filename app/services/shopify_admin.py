"""Short-lived, partitioned admin sessions issued only after Shopify verification."""
import os
import time
from html import escape

import jwt
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

COOKIE = "__Secure-paintingify-shopify"
MAX_AGE = 900
ORIGIN = "https://image-api.shopupup.com"


def allowed_shop():
    return os.getenv("SHOPIFY_SHOP", "ai-generate-image").removesuffix(".myshopify.com")


def secret():
    return os.getenv("SHOPIFY_API_SECRET") or os.environ["SHOPIFY_CLIENT_SECRET"]


def session_shop(request: Request):
    try:
        claims = jwt.decode(request.cookies.get(COOKIE, ""), secret(), algorithms=["HS256"],
                            audience="paintingify-admin", options={"require": ["exp", "shop", "sub"]})
        return claims["shop"] if claims["shop"] == allowed_shop() else None
    except (jwt.PyJWTError, KeyError):
        return None


def set_session(response, user_id):
    token = jwt.encode({"aud": "paintingify-admin", "shop": allowed_shop(), "sub": str(user_id),
                        "exp": int(time.time()) + MAX_AGE}, secret(), algorithm="HS256")
    response.set_cookie(COOKIE, token, max_age=MAX_AGE, secure=True, httponly=True,
                        samesite="none", path="/")
    # CHIPS isolates this cookie to the Shopify top-level site (Python 3.12).
    response.headers["set-cookie"] += "; Partitioned"


class ShopifyAdminMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        shop = session_shop(request) if request.url.path.startswith("/admin/") else None
        if shop:
            if request.method not in {"GET", "HEAD", "OPTIONS"} and request.headers.get("origin") != ORIGIN:
                return Response("Forbidden", status_code=403)
            request.state.shopify_admin = True
        response = await call_next(request)
        if shop:
            response.headers["Content-Security-Policy"] = f"frame-ancestors https://{shop}.myshopify.com https://admin.shopify.com;"
            response.headers["Cache-Control"] = "no-store"
            response.headers["Referrer-Policy"] = "no-referrer"
            if request.url.path == "/admin/logout":
                response.delete_cookie(COOKIE, path="/", secure=True, httponly=True, samesite="none")
                response.headers["set-cookie"] += "; Partitioned"
            if "text/html" in response.headers.get("content-type", ""):
                body = b"".join([chunk async for chunk in response.body_iterator])
                key = escape(os.getenv("SHOPIFY_API_KEY") or os.getenv("SHOPIFY_CLIENT_ID", ""), quote=True)
                bridge = (f'<meta name="shopify-api-key" content="{key}">'
                          '<script src="https://cdn.shopify.com/shopifycloud/app-bridge.js"></script>'
                          '<script>setInterval(async()=>{try{const token=await shopify.idToken();'
                          'await fetch("/shopify/session",{method:"POST",headers:{Authorization:"Bearer "+token}});'
                          '}catch(e){}},300000);</script>')
                body = body.replace(b"<head>", ("<head>" + bridge).encode(), 1)
                headers = dict(response.headers)
                headers.pop("content-length", None)
                return Response(body, status_code=response.status_code, headers=headers, background=response.background)
        return response
