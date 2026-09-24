import json
import os
from collections import defaultdict

from app.core.config import DirectoryConfig
from app.services.r2_storage import read_r2_mapping, replace_with_cdn_urls
from app.services.storefront_events import read_storefront_events, summarize_storefront_events
from app.services.task_files import resolve_task_generated_original


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")
API_RESPONSE_FILES = ("bfl_response.json", "replicate_response.json")
EDIT_REQUEST_PREFIX = "Apply only this customer request: "
EDIT_REQUEST_SUFFIX = "\nPreserve the subject identity"


def extract_edit_instructions(params: dict) -> str:
    explicit = str(params.get("edit_instructions") or "").strip()
    if explicit:
        return explicit

    prompt = str(params.get("original_prompt") or params.get("prompt") or "")
    marker_index = prompt.find(EDIT_REQUEST_PREFIX)
    if marker_index < 0:
        return ""
    value_start = marker_index + len(EDIT_REQUEST_PREFIX)
    value_end = prompt.find(EDIT_REQUEST_SUFFIX, value_start)
    if value_end < 0:
        value_end = len(prompt)
    return prompt[value_start:value_end].strip()


def get_source_reference_files(source_task_id: str) -> list[dict]:
    source_path = resolve_task_generated_original(source_task_id)
    if not source_path:
        return []

    filename = os.path.basename(source_path)
    source_dir = os.path.dirname(source_path)
    local_url = f"/taskfile/{source_task_id}/{filename}"
    public_url = read_r2_mapping(source_dir).get(local_url, local_url)
    return [
        {
            "task_id": source_task_id,
            "filename": filename,
            "url": public_url,
        }
    ]


def list_task_summaries(
    page: int | None = None,
    page_size: int | None = None,
    client_ip_query: str | None = None,
    user_query: str | None = None,
    task_id_query: str | None = None,
    provider_query: str | None = None,
) -> dict:
    """Return compact task summaries, optionally filtered and paginated."""
    tasks = []
    task_entries = []
    if os.path.exists(DirectoryConfig.TASKS_DIR):
        with os.scandir(DirectoryConfig.TASKS_DIR) as entries:
            task_entries = [
                entry
                for entry in entries
                if entry.is_dir()
                and os.path.isfile(os.path.join(entry.path, "params.json"))
            ]

    # Task directory names begin with YYYYMMDD_HHMMSS, so lexical order keeps
    # the newest tasks first without opening every task's params.json file.
    task_entries.sort(key=lambda entry: entry.name, reverse=True)
    normalized_ip_query = str(client_ip_query or "").strip().casefold()
    normalized_user_query = str(user_query or "").strip().casefold()
    normalized_task_id_query = str(task_id_query or "").strip().casefold()
    normalized_provider_query = str(provider_query or "").strip().casefold()
    all_params = {
        entry.name: _read_json_if_exists(os.path.join(entry.path, "params.json"))
        for entry in task_entries
    }

    if (
        normalized_ip_query
        or normalized_user_query
        or normalized_task_id_query
        or normalized_provider_query
    ):
        filtered_entries = []
        for entry in task_entries:
            params = all_params[entry.name]
            client_ip = str(params.get("client_ip") or "").casefold()
            provider = str(params.get("api_provider") or "").casefold()
            user_values = (
                params.get("customer_email", ""),
                params.get("customer_id", ""),
                params.get("storefront_visitor_id", ""),
            )
            matches_ip = not normalized_ip_query or normalized_ip_query in client_ip
            matches_user = not normalized_user_query or any(
                normalized_user_query in str(value or "").casefold()
                for value in user_values
            )
            matches_task_id = (
                not normalized_task_id_query
                or normalized_task_id_query in entry.name.casefold()
            )
            matches_provider = (
                not normalized_provider_query
                or normalized_provider_query in provider
            )
            if matches_ip and matches_user and matches_task_id and matches_provider:
                filtered_entries.append(entry)
        task_entries = filtered_entries

    total = len(task_entries)

    if page_size is None:
        selected_entries = task_entries
        current_page = 1
        normalized_page_size = total or 1
        total_pages = 1
    else:
        normalized_page_size = max(10, min(int(page_size), 100))
        total_pages = max(1, (total + normalized_page_size - 1) // normalized_page_size)
        current_page = max(1, min(int(page or 1), total_pages))
        start = (current_page - 1) * normalized_page_size
        selected_entries = task_entries[start : start + normalized_page_size]

    selected_task_ids = {entry.name for entry in selected_entries}
    selected_params = {}
    ip_task_counts = defaultdict(int)
    ip_task_sequences = {}

    # Assign each IP a stable, chronological task number across every page.
    # Directory names start with the task timestamp, so reversing the normal
    # newest-first order gives us oldest-first numbering.
    for entry in reversed(task_entries):
        params = all_params[entry.name]
        if entry.name in selected_task_ids:
            selected_params[entry.name] = params
        client_ip = str(params.get("client_ip") or "").strip()
        if client_ip:
            ip_task_counts[client_ip] += 1
            ip_task_sequences[entry.name] = ip_task_counts[client_ip]

    for entry in selected_entries:
        task_id = entry.name
        task_dir = entry.path
        filenames = os.listdir(task_dir)

        params = selected_params.get(task_id, {})
        source_task_id = str(params.get("source_task_id") or "").strip()
        event_summary = summarize_storefront_events(read_storefront_events(task_dir))
        response = _read_first_api_response(task_dir, filenames)
        output_filenames = [
            filename
            for filename in filenames
            if filename.lower().endswith(IMAGE_EXTENSIONS)
        ]

        local_output_files = [
            f"/taskfile/{task_id}/{filename}" for filename in output_filenames
        ]
        tasks.append(
            {
                "task_id": task_id,
                "description": params.get("description", ""),
                "created_at": params.get("time", ""),
                "time_zone": params.get("time_zone", "UTC"),
                "extracted_seed": response.get("extracted_seed"),
                "output_files_count": len(output_filenames),
                "status": response.get("status", "unknown"),
                "api_provider": params.get("api_provider"),
                "request_url": params.get("request_url", ""),
                "source_page_url": params.get("source_page_url", ""),
                "source_page_title": params.get("source_page_title", ""),
                "source_page_type": params.get("source_page_type", ""),
                "source_product_id": params.get("source_product_id", ""),
                "source_product_handle": params.get("source_product_handle", ""),
                "source_product_title": params.get("source_product_title", ""),
                "shop_domain": params.get("shop_domain", ""),
                "customer_id": params.get("customer_id", ""),
                "customer_email": params.get("customer_email", "")
                or event_summary.get("event_customer_email", ""),
                "customer_logged_in": params.get("customer_logged_in", False),
                "storefront_visitor_id": params.get("storefront_visitor_id", ""),
                "traffic_source": params.get("traffic_source", ""),
                "traffic_referrer": params.get("traffic_referrer", ""),
                "traffic_landing_path": params.get("traffic_landing_path", ""),
                "utm_source": params.get("utm_source", ""),
                "utm_medium": params.get("utm_medium", ""),
                "utm_campaign": params.get("utm_campaign", ""),
                "utm_term": params.get("utm_term", ""),
                "utm_content": params.get("utm_content", ""),
                "client_ip": params.get("client_ip", ""),
                "ip_task_sequence": ip_task_sequences.get(task_id),
                "ip_task_total": ip_task_counts.get(
                    str(params.get("client_ip") or "").strip()
                ),
                "client_country": params.get("client_country", ""),
                "generation_duration_seconds": params.get(
                    "generation_duration_seconds"
                ),
                "user_agent": params.get("user_agent", ""),
                "prompt": params.get("original_prompt", params.get("prompt", "")),
                "edit_instructions": extract_edit_instructions(params),
                "source_task_id": source_task_id,
                "source_reference_files": get_source_reference_files(source_task_id),
                "output_files": replace_with_cdn_urls(
                    local_output_files, read_r2_mapping(task_dir)
                ),
                **event_summary,
            }
        )

    return {
        "tasks": tasks,
        "total": total,
        "page": current_page,
        "page_size": normalized_page_size,
        "total_pages": total_pages,
        "has_previous": current_page > 1,
        "has_next": current_page < total_pages,
    }


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

    local_output_files = [
        {"filename": filename, "url": f"/taskfile/{task_id}/{filename}"}
        for filename in os.listdir(task_dir)
        if filename.endswith(IMAGE_EXTENSIONS)
    ]
    cdn_mapping = read_r2_mapping(task_dir)
    output_files = [
        {**item, "url": cdn_mapping.get(item["url"], item["url"])}
        for item in local_output_files
    ]

    return {
        "task_id": task_id,
        "params": main_params,
        "api_responses": api_responses,
        "output_files": output_files,
        **summarize_storefront_events(read_storefront_events(task_dir)),
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


def _read_first_api_response(directory: str, filenames: list[str] | None = None) -> dict:
    response = _read_first_json(directory, API_RESPONSE_FILES)
    if response:
        return response
    for filename in filenames if filenames is not None else os.listdir(directory):
        if filename.endswith("_response.json"):
            response = _read_json_if_exists(os.path.join(directory, filename))
            if isinstance(response, dict) and response:
                return response
    return {}


def _read_json_if_exists(path: str) -> dict | list:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
