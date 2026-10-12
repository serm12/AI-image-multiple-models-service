"""Lazy initialization of Shopify's official, framework-independent app SDK."""

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from shopify_app import ShopifyApp


@lru_cache(maxsize=1)
def get_shopify_app() -> ShopifyApp:
    """Load backend credentials only when Shopify functionality is requested.

    Process environment variables take precedence over the local .env file.
    This creates an SDK instance; it does not install the app or obtain a token.
    """
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    # Shopify CLI provides API_KEY/API_SECRET to its child process. Prefer those
    # over local credentials when developing against a CLI-selected app.
    client_id = (os.getenv("SHOPIFY_API_KEY") or os.getenv("SHOPIFY_CLIENT_ID", "")).strip()
    client_secret = (os.getenv("SHOPIFY_API_SECRET") or os.getenv("SHOPIFY_CLIENT_SECRET", "")).strip()
    if not client_id or not client_secret:
        raise RuntimeError(
            "Configure SHOPIFY_CLIENT_ID and SHOPIFY_CLIENT_SECRET in the backend environment."
        )
    return ShopifyApp(client_id=client_id, client_secret=client_secret)
