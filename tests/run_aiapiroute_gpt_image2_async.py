import time
from pathlib import Path

import requests


BASE_URL = "http://localhost:8001"
PROVIDER = "aiapiroute_gpt-image-2"
INPUT_IMAGES = [
    Path("assets/logo_brand.png"),
    Path("assets/logo_brand-white.png"),
    Path("assets/logo_watermark.png"),
]


def submit_task(image_path: Path, index: int) -> str:
    data = {
        "provider": PROVIDER,
        "prompt": (
            f"Async integration test #{index}. Create a polished product-style image "
            "based on the uploaded reference, clean white background, sharp details."
        ),
        "aspect_ratio": "1:1",
        "output_format": "png",
        "art_style": "flux_realistic",
        "auto_upscale": "false",
        "size": "1K",
    }
    with image_path.open("rb") as image_file:
        files = {"files": (image_path.name, image_file, "image/png")}
        response = requests.post(
            f"{BASE_URL}/generate-async/",
            data=data,
            files=files,
            timeout=60,
        )
    print(f"SUBMIT {image_path.name}: {response.status_code} {response.text}")
    response.raise_for_status()
    return response.json()["task_id"]


def poll_task(task_id: str, timeout_seconds: int = 600) -> dict:
    deadline = time.time() + timeout_seconds
    last_status = None
    while time.time() < deadline:
        response = requests.get(f"{BASE_URL}/task-status/{task_id}", timeout=20)
        response.raise_for_status()
        payload = response.json()
        status = payload.get("status")
        progress = payload.get("progress")
        if (status, progress) != last_status:
            print(f"POLL {task_id}: status={status} progress={progress} files={payload.get('files')}")
            last_status = (status, progress)
        if status in {"completed", "failed", "cancelled"}:
            return payload
        time.sleep(5)
    raise TimeoutError(f"Timed out waiting for task {task_id}")


def main() -> int:
    version = requests.get(f"{BASE_URL}/version", timeout=10).json()
    print(f"SERVER VERSION: {version}")

    task_ids = [submit_task(image_path, index + 1) for index, image_path in enumerate(INPUT_IMAGES)]
    print(f"TASK_IDS: {task_ids}")

    results = [poll_task(task_id) for task_id in task_ids]
    print("\nRESULTS:")
    failed = False
    for result in results:
        print(result)
        failed = failed or result.get("status") != "completed"
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
