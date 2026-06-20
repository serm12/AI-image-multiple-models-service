import os
import shutil
import sys

from fastapi.testclient import TestClient

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.core.config import APIConfig, AppConfig
from app.main import app


client = TestClient(app)
TASKS_DIR = os.path.join(PROJECT_ROOT, "tasks")


def admin_headers() -> dict:
    if not AppConfig.ADMIN_API_KEY:
        return {}
    return {"x-api-key": AppConfig.ADMIN_API_KEY}


def check(method: str, path: str, expected_status: set[int], **kwargs) -> tuple[bool, str]:
    response = getattr(client, method)(path, **kwargs)
    ok = response.status_code in expected_status
    return ok, f"{method.upper()} {path} -> {response.status_code}"


def main() -> int:
    existing_task_dirs = set(os.listdir(TASKS_DIR)) if os.path.exists(TASKS_DIR) else set()
    checks: list[tuple[str, bool, str]] = []

    def add(name: str, method: str, path: str, expected_status: set[int], **kwargs):
        ok, detail = check(method, path, expected_status, **kwargs)
        checks.append((name, ok, detail))

    add("root", "get", "/", {200})
    add("health", "get", "/health", {200})
    add("version", "get", "/version", {200})
    add("providers", "get", "/providers", {200})
    add("provider options", "get", "/provider-options", {200})
    add("styles", "get", "/styles/", {200})
    add("seeds", "get", "/seeds/", {200})
    add("last seed", "get", "/seeds/last", {200})
    add("seed search miss", "get", "/seeds/search/__smoke_missing__", {200})
    add("upscale models", "get", "/upscale/models", {200})
    add("task status missing", "get", "/task-status/__smoke_missing__", {404})
    add("task file missing", "get", "/taskfile/__smoke_missing__/missing.png", {404})
    add("task detail missing", "get", "/task/__smoke_missing__", {404})
    add("system stats", "get", "/system-stats", {200}, headers=admin_headers())
    add("tasks admin", "get", "/tasks/", {200}, headers=admin_headers())
    add("token usage", "get", "/token-usage/", {200}, headers=admin_headers())

    fake_photo = {"file": ("invalid.jpg", b"not an image", "image/jpeg")}
    response = client.post("/check-photo/", files=fake_photo, data={"client_city": "smoke"})
    checks.append(("check photo invalid image", response.status_code == 200, f"POST /check-photo/ -> {response.status_code}"))

    if APIConfig.get_available_providers():
        configured = [p["provider"] for p in APIConfig.get_available_providers() if p["configured"]]
    else:
        configured = []

    if configured:
        response = client.post(
            "/generate-async/",
            data={
                "provider": configured[0],
                "prompt": "smoke test",
                "aspect_ratio": "3:4",
                "output_format": "png",
                "art_style": "flux_realistic",
            },
        )
        checks.append(
            (
                "generate validation without input image",
                response.status_code == 500,
                f"POST /generate-async/ missing input -> {response.status_code}",
            )
        )
    else:
        checks.append(("generate validation without input image", True, "SKIP no configured provider"))

    previous_default_provider = APIConfig.IMAGE_GENERATION_PROVIDER
    previous_bfl_key = APIConfig.BFL_API_KEY
    try:
        APIConfig.IMAGE_GENERATION_PROVIDER = "flux_bfl"
        APIConfig.BFL_API_KEY = "smoke-test-key"
        response = client.post(
            "/generate-async/",
            data={
                "provider": "",
                "prompt": "smoke test default provider",
                "aspect_ratio": "3:4",
                "output_format": "png",
                "art_style": "flux_realistic",
            },
        )
        checks.append(
            (
                "generate empty provider uses default",
                response.status_code == 500 and "必须提供 files 或 input_image_url" in response.text,
                f"POST /generate-async/ empty provider -> {response.status_code}",
            )
        )
    finally:
        APIConfig.IMAGE_GENERATION_PROVIDER = previous_default_provider
        APIConfig.BFL_API_KEY = previous_bfl_key

    previous_google_key = APIConfig.GOOGLE_GEMINI_API_KEY
    try:
        APIConfig.GOOGLE_GEMINI_API_KEY = "smoke-test-key"
        response = client.post(
            "/generate-async/",
            data={
                "provider": "gemini-nanobanana_google",
                "prompt": "smoke test gemini automatic dimension",
                "aspect_ratio": "match_input_image",
                "output_format": "png",
                "art_style": "gemini_realistic",
            },
        )
        checks.append(
            (
                "generate gemini match input dimension",
                response.status_code == 500 and "必须提供 files 或 input_image_url" in response.text,
                f"POST /generate-async/ gemini match_input_image -> {response.status_code}",
            )
        )
    finally:
        APIConfig.GOOGLE_GEMINI_API_KEY = previous_google_key

    checks.append(("upscale external call", True, "SKIP would call external Replicate API"))

    if "flux_bfl" in APIConfig.ALL_PROVIDERS:
        add("resume bfl missing task", "post", "/resume-bfl/__smoke_missing__", {404})

    try:
        failed = [item for item in checks if not item[1]]
        for name, ok, detail in checks:
            status = "PASS" if ok else "FAIL"
            print(f"{status}: {name}: {detail}")

        print(f"\nSummary: {len(checks) - len(failed)}/{len(checks)} checks passed")
        return 1 if failed else 0
    finally:
        cleanup_new_task_dirs(existing_task_dirs)


def cleanup_new_task_dirs(existing_task_dirs: set[str]) -> None:
    if not os.path.exists(TASKS_DIR):
        return
    tasks_root = os.path.abspath(TASKS_DIR)
    for name in os.listdir(TASKS_DIR):
        if name in existing_task_dirs:
            continue
        path = os.path.abspath(os.path.join(TASKS_DIR, name))
        if os.path.isdir(path) and os.path.commonpath([tasks_root, path]) == tasks_root:
            shutil.rmtree(path)


if __name__ == "__main__":
    raise SystemExit(main())
