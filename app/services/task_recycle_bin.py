"""Thirty-day task recycling. Durable markers survive partial cleanup/restarts."""
import asyncio
import json
import logging
import os
import re
import shutil
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.core.config import DirectoryConfig

logger = logging.getLogger(__name__)
_lock = threading.RLock()
RETENTION_DAYS = 30
TASK_ID = re.compile(r'[A-Za-z0-9][A-Za-z0-9_-]{0,159}\Z')


def marker_path(task_id):
    if not TASK_ID.fullmatch(task_id):
        raise ValueError('invalid_task_id')
    return Path(DirectoryConfig.TASKS_DIR).resolve() / '.recycle-bin' / (task_id + '.json')


def metadata(task_id):
    try:
        value = json.loads(marker_path(task_id).read_text('utf-8'))
        if not isinstance(value, dict): raise ValueError('invalid_recycle_metadata')
        return value
    except FileNotFoundError:
        return {}


def write_marker(task_id, value):
    path = marker_path(task_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value), 'utf-8')
    os.replace(temporary, path)


def task_folder(task_id):
    marker_path(task_id)  # Reject traversal, hidden folders and separators.
    root = Path(DirectoryConfig.TASKS_DIR).resolve()
    folder = root / task_id
    if folder.is_symlink() or (hasattr(folder, 'is_junction') and folder.is_junction()) or folder.resolve().parent != root:
        raise ValueError('unsafe_task_path')
    return folder


def is_running(task_id):
    from app.services.async_task_manager import task_manager
    item = task_manager.get_task(task_id)
    return task_id in task_manager.active_tasks or bool(item and item.get('status') not in {'completed', 'failed', 'cancelled'})


def change_tasks(task_ids, *, restore=False, now=None):
    from app.services.task_list_index import refresh_task
    now = now or datetime.now(timezone.utc)
    changed, failed = [], []
    for task_id in dict.fromkeys(task_ids):
        try:
            with _lock:
                folder = task_folder(task_id)
                if not (folder / 'params.json').is_file(): raise ValueError('task_not_found')
                value = metadata(task_id)
                if value.get('cleanup_started_at'): raise ValueError('cleanup_in_progress')
                if is_running(task_id): raise ValueError('task_running')
                if restore:
                    marker_path(task_id).unlink(missing_ok=True)
                elif not value:
                    write_marker(task_id, {'trashed_at': now.isoformat(),
                        'purge_after': (now + timedelta(days=RETENTION_DAYS)).isoformat()})
                refresh_task(folder)
                changed.append(task_id)
        except (ValueError, OSError) as error:
            reason = str(error) if isinstance(error, ValueError) else 'storage_error'
            failed.append({'task_id': task_id, 'reason': reason})
    return {'deleted': changed, 'failed': failed}


def clear_task_caches(task_id):
    from app.services.shopify_customers import database
    from app.services import shopify_orders
    with database() as db:
        db.execute('DELETE FROM jobs WHERE task_id=?', (task_id,))
    with shopify_orders._write_lock:
        cache = shopify_orders.read_order_cache()
        if cache:
            cache.get('tasks', {}).pop(task_id, None)
            for order in cache.get('customer_orders', {}).values():
                order['items'] = [item for item in order.get('items', []) if item.get('task_id') != task_id]
            path = shopify_orders.cache_path()
            temporary = path.with_suffix('.tmp')
            temporary.write_text(json.dumps(cache, ensure_ascii=False), 'utf-8')
            os.replace(temporary, path)


def purge_expired(now=None, *, delete_cloud=None):
    from app.services.r2_storage import delete_task_images
    from app.services.task_list_index import refresh_task
    from app.services.storefront_events import _event_write_lock
    from app.services.async_task_manager import task_manager
    now = now or datetime.now(timezone.utc)
    delete_cloud = delete_cloud or delete_task_images
    directory = Path(DirectoryConfig.TASKS_DIR).resolve() / '.recycle-bin'
    removed = 0
    for path in sorted(directory.glob('*.json')):
        task_id = path.stem
        try:
            with _lock, _event_write_lock:
                value = metadata(task_id)
                deadline = datetime.fromisoformat(value['purge_after'])
                if deadline.tzinfo is None or deadline > now or is_running(task_id): continue
                folder = task_folder(task_id)
                # This durable marker prevents restoring partially deleted images.
                if not value.get('cleanup_started_at'):
                    value['cleanup_started_at'] = now.isoformat()
                    write_marker(task_id, value)
                delete_cloud(task_id, folder)
                clear_task_caches(task_id)
                # Recheck the absolute root immediately before recursive removal.
                if task_folder(task_id) != folder: raise ValueError('unsafe_task_path')
                if folder.exists(): shutil.rmtree(folder)
                task_manager.tasks.pop(task_id, None)
                refresh_task(folder)
                path.unlink(missing_ok=True)
                removed += 1
        except Exception as error:
            logger.warning('Recycle cleanup will retry task=%s error=%s', task_id, type(error).__name__)
    return removed


async def worker():
    while True:
        try:
            count = await asyncio.to_thread(purge_expired)
            if count: logger.info('Recycle bin permanently cleaned %s tasks', count)
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            return
        except Exception:
            logger.exception('Recycle cleanup unavailable; retrying in one hour')
            await asyncio.sleep(3600)
