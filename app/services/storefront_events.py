import hashlib
import hmac
import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

from app.core.config import DirectoryConfig


EVENTS_FILENAME = "storefront_events.json"
ALLOWED_EVENT_TYPES = {
    "added_to_cart",
    "checkout_intent",
    "checkout_started",
    "checkout_completed",
}
_event_write_lock = threading.Lock()


def hash_tracking_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _read_json(path: Path, default):
    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return default


def _task_dir(task_id: str) -> Path | None:
    if not task_id or Path(task_id).name != task_id:
        return None
    root = Path(DirectoryConfig.TASKS_DIR).resolve()
    candidate = (root / task_id).resolve()
    if candidate.parent != root or not candidate.is_dir():
        return None
    return candidate


def record_storefront_event(task_id: str, tracking_token: str, event: dict) -> dict:
    task_dir = _task_dir(task_id)
    if task_dir is None:
        raise FileNotFoundError("Task not found")

    params = _read_json(task_dir / "params.json", {})
    expected_hash = str(params.get("storefront_tracking_token_hash") or "")
    supplied_hash = hash_tracking_token(tracking_token)
    if not expected_hash or not hmac.compare_digest(expected_hash, supplied_hash):
        raise PermissionError("Invalid tracking token")

    event_type = str(event.get("event_type") or "")
    if event_type not in ALLOWED_EVENT_TYPES:
        raise ValueError("Unsupported event type")

    event_id = str(event.get("event_id") or "")[:120]
    if not event_id:
        raise ValueError("event_id is required")

    normalized = {
        "event_id": event_id,
        "event_type": event_type,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "occurred_at": str(event.get("occurred_at") or "")[:50],
        "source": str(event.get("source") or "theme")[:50],
        "shop_domain": str(event.get("shop_domain") or "")[:255],
        "customer_id": str(event.get("customer_id") or "")[:100],
        "customer_logged_in": bool(event.get("customer_logged_in")),
        "visitor_id": str(event.get("visitor_id") or "")[:100],
        "cart_token_hash": hash_tracking_token(str(event.get("cart_token") or ""))
        if event.get("cart_token")
        else "",
        "variant_id": str(event.get("variant_id") or "")[:100],
        "checkout_token": str(event.get("checkout_token") or "")[:255],
    }

    events_path = task_dir / EVENTS_FILENAME
    with _event_write_lock:
        events = _read_json(events_path, [])
        if not isinstance(events, list):
            events = []
        existing = next(
            (item for item in events if item.get("event_id") == event_id), None
        )
        if existing:
            return {"event": existing, "created": False}
        events.append(normalized)
        events = events[-100:]
        temp_path = events_path.with_suffix(".tmp")
        with temp_path.open("w", encoding="utf-8") as file:
            json.dump(events, file, ensure_ascii=False, indent=2)
        os.replace(temp_path, events_path)
    return {"event": normalized, "created": True}


def read_storefront_events(task_dir: str) -> list[dict]:
    events = _read_json(Path(task_dir) / EVENTS_FILENAME, [])
    return events if isinstance(events, list) else []


def summarize_storefront_events(events: list[dict]) -> dict:
    event_types = {str(event.get("event_type") or "") for event in events}
    return {
        "storefront_events": events,
        "added_to_cart": "added_to_cart" in event_types,
        "checkout_intent": "checkout_intent" in event_types,
        "checkout_started": "checkout_started" in event_types,
        "checkout_completed": "checkout_completed" in event_types,
    }
