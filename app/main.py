import asyncio
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi

from app.core.config import APIConfig, AppConfig, initialize_config
from app.core.version import APP_RELEASE_DATE, APP_VERSION
from app.routers.api import router as api_router
from app.services.async_task_manager import task_manager
from app.services.runtime_state import clear_http_clients, set_http_clients


initialize_config()


async def _periodic_task_cleanup():
    """Clean old completed/failed task records to avoid unbounded memory growth."""
    while True:
        try:
            await asyncio.sleep(300)
            removed = task_manager.cleanup_old_tasks(max_age_hours=24)
            if removed:
                print(f"🧹 定期清理：已移除 {removed} 个过期任务记录")
        except Exception as e:
            print(f"⚠️ 任务清理出错: {e}")


@asynccontextmanager
async def app_lifespan(app: FastAPI):
    limits = httpx.Limits(max_keepalive_connections=10, max_connections=20)
    http_client = httpx.AsyncClient(timeout=60.0, limits=limits)
    long_http_client = httpx.AsyncClient(timeout=300.0, limits=limits)
    app.state.http_client = http_client
    app.state.long_http_client = long_http_client
    set_http_clients(http_client, long_http_client)
    cleanup_task = asyncio.create_task(_periodic_task_cleanup())
    try:
        yield
    finally:
        cleanup_task.cancel()
        clear_http_clients()
        await http_client.aclose()
        await long_http_client.aclose()


app = FastAPI(
    title="AI Image Generation API",
    description=(
        "AI图像生成API服务 - 支持多种模型，异步高并发处理，支持20个并发能力\n\n"
        f"版本: {APP_VERSION}\n\n"
        f"发布日期: {APP_RELEASE_DATE}"
    ),
    version=APP_VERSION,
    lifespan=app_lifespan,
)


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    openapi_schema["info"]["x-release-date"] = APP_RELEASE_DATE
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi

app.add_middleware(
    CORSMiddleware,
    allow_origins=AppConfig.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)

if APIConfig.REPLICATE_API_TOKEN:
    import os

    os.environ["REPLICATE_API_TOKEN"] = APIConfig.REPLICATE_API_TOKEN
