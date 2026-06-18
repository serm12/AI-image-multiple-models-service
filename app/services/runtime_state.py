import httpx


_http_client: httpx.AsyncClient | None = None
_long_http_client: httpx.AsyncClient | None = None


def set_http_clients(http_client: httpx.AsyncClient, long_http_client: httpx.AsyncClient) -> None:
    global _http_client, _long_http_client
    _http_client = http_client
    _long_http_client = long_http_client


def clear_http_clients() -> None:
    global _http_client, _long_http_client
    _http_client = None
    _long_http_client = None


def get_http_client() -> httpx.AsyncClient:
    if _http_client is None:
        raise RuntimeError("HTTP client is not initialized")
    return _http_client


def get_long_http_client() -> httpx.AsyncClient:
    if _long_http_client is None:
        raise RuntimeError("Long HTTP client is not initialized")
    return _long_http_client
