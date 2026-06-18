import time

import requests


BASE_URL = "http://localhost:8001"
PROVIDER = "aiapiroute_gpt-image-2"
PROMPTS = [
    "A cinematic product photo of a matte black ceramic coffee cup on a clean white table, soft studio lighting, minimal composition.",
    "A cheerful watercolor illustration of a small bakery storefront on a sunny morning, pastel colors, cozy atmosphere.",
    "A futuristic electric scooter concept render, white background, premium industrial design, sharp details.",
]


def submit_task(prompt: str, index: int) -> str:
    data = {
        "provider": PROVIDER,
        "prompt": f"Async text-to-image test #{index}. {prompt}",
        "aspect_ratio": "1:1",
        "output_format": "png",
        "art_style": "flux_realistic",
        "auto_upscale": "false",
        "size": "1K",
    }
    response = requests.post(f"{BASE_URL}/generate-async/", data=data, timeout=60)
    print(f"SUBMIT #{index}: {response.status_code} {response.text}")
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


def verify_files(result: dict) -> None:
    for path in result.get("files") or []:
        url = f"{BASE_URL}{path}"
        response = requests.get(url, timeout=30)
        print(f"FILE {path}: {response.status_code} {response.headers.get('content-type')} {len(response.content)} bytes")
        response.raise_for_status()


def main() -> int:
    version = requests.get(f"{BASE_URL}/version", timeout=10).json()
    print(f"SERVER VERSION: {version}")

    task_ids = [submit_task(prompt, index + 1) for index, prompt in enumerate(PROMPTS)]
    print(f"TASK_IDS: {task_ids}")

    results = [poll_task(task_id) for task_id in task_ids]
    print("\nRESULTS:")
    failed = False
    for result in results:
        print(result)
        verify_files(result)
        failed = failed or result.get("status") != "completed"
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
