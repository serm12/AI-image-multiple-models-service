import json
import os

from app.core.config import DirectoryConfig


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")
API_RESPONSE_FILES = ("bfl_response.json", "replicate_response.json")


def list_task_summaries() -> dict:
    """Return compact task summaries for the admin task list endpoint."""
    tasks = []
    if os.path.exists(DirectoryConfig.TASKS_DIR):
        for task_id in os.listdir(DirectoryConfig.TASKS_DIR):
            task_dir = os.path.join(DirectoryConfig.TASKS_DIR, task_id)
            if not os.path.isdir(task_dir):
                continue

            params = _read_json_if_exists(os.path.join(task_dir, "params.json"))
            response = _read_first_json(task_dir, API_RESPONSE_FILES)
            output_files_count = sum(
                1 for filename in os.listdir(task_dir) if filename.endswith(IMAGE_EXTENSIONS)
            )

            tasks.append(
                {
                    "task_id": task_id,
                    "description": params.get("description", ""),
                    "created_at": params.get("time", ""),
                    "extracted_seed": response.get("extracted_seed"),
                    "output_files_count": output_files_count,
                    "status": response.get("status", "unknown"),
                    "api_provider": params.get("api_provider"),
                }
            )

    tasks.sort(key=lambda x: x["created_at"], reverse=True)
    return {"tasks": tasks, "total": len(tasks)}


def get_task_detail(task_id: str) -> dict | None:
    """Return detailed task data, or None when the task folder does not exist."""
    task_dir = os.path.join(DirectoryConfig.TASKS_DIR, task_id)
    if not os.path.exists(task_dir):
        return None

    main_params = _read_json_if_exists(os.path.join(task_dir, "params.json"))
    api_responses = {}
    for response_file in API_RESPONSE_FILES:
        response_path = os.path.join(task_dir, response_file)
        response = _read_json_if_exists(response_path)
        if response:
            api_responses[response_file] = response

    output_files = [
        {"filename": filename, "url": f"/taskfile/{task_id}/{filename}"}
        for filename in os.listdir(task_dir)
        if filename.endswith(IMAGE_EXTENSIONS)
    ]

    return {
        "task_id": task_id,
        "params": main_params,
        "api_responses": api_responses,
        "output_files": output_files,
    }


def get_token_usage_summary(log_file: str = "gemini_token_usage.json") -> dict:
    """Read Gemini token usage data and calculate summary statistics."""
    if not os.path.exists(log_file):
        return {
            "success": True,
            "message": "暂无Token使用记录",
            "stats": {
                "total_requests": 0,
                "total_tokens": 0,
                "total_cost": 0,
                "avg_tokens_per_request": 0,
            },
            "recent_logs": [],
            "all_logs": [],
        }

    logs = _read_json_if_exists(log_file)
    if not logs:
        return {
            "success": True,
            "stats": {
                "total_requests": 0,
                "total_tokens": 0,
                "total_cost": 0,
                "avg_tokens_per_request": 0,
            },
            "recent_logs": [],
            "all_logs": [],
        }

    total_tokens = sum(log.get("total_tokens", 0) for log in logs)
    total_cost = sum(log.get("estimated_cost", 0) for log in logs)
    total_requests = len(logs)
    avg_tokens = total_tokens / total_requests if total_requests > 0 else 0

    return {
        "success": True,
        "stats": {
            "total_requests": total_requests,
            "total_tokens": total_tokens,
            "total_cost": round(total_cost, 6),
            "avg_tokens_per_request": round(avg_tokens, 1),
        },
        "recent_logs": logs[-10:],
        "all_logs": logs,
    }


def _read_first_json(directory: str, filenames: tuple[str, ...]) -> dict:
    for filename in filenames:
        data = _read_json_if_exists(os.path.join(directory, filename))
        if data:
            return data
    return {}


def _read_json_if_exists(path: str) -> dict | list:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
