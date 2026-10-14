"""Rebuildable list index. Task folders remain the source of truth."""
import asyncio
import json
import logging
import sqlite3
import threading
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path
from app.core.config import DirectoryConfig

logger = logging.getLogger(__name__)
_lock = threading.RLock()
FILES = ('params.json', 'storefront_events.json', 'admin_metadata.json')
EVENTS = ('added_to_cart', 'checkout_intent', 'checkout_started', 'checkout_completed')


@contextmanager
def database():
    root = Path(DirectoryConfig.TASKS_DIR)
    root.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(root / '.task-list-index.sqlite3', timeout=15)
    db.row_factory = sqlite3.Row
    try:
        db.execute('PRAGMA journal_mode=WAL')
        db.executescript('''
        CREATE TABLE IF NOT EXISTS tasks (
          task_id TEXT PRIMARY KEY, task_search TEXT NOT NULL, created REAL,
          ip TEXT NOT NULL, ip_search TEXT NOT NULL, user_search TEXT NOT NULL,
          provider TEXT NOT NULL, starred INTEGER NOT NULL, conversions INTEGER NOT NULL,
          fingerprint TEXT NOT NULL);
        CREATE INDEX IF NOT EXISTS task_created ON tasks(created DESC, task_id DESC);
        CREATE INDEX IF NOT EXISTS task_starred ON tasks(starred, created DESC, task_id DESC);
        CREATE INDEX IF NOT EXISTS task_ip ON tasks(ip, created, task_id);
        CREATE INDEX IF NOT EXISTS task_provider ON tasks(provider);
        CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY, value TEXT);
        ''')
        if 'trashed_at' not in {row[1] for row in db.execute('PRAGMA table_info(tasks)')}:
            db.execute("ALTER TABLE tasks ADD COLUMN trashed_at TEXT NOT NULL DEFAULT ''")
        yield db
        db.commit()
    finally:
        db.close()


def fingerprint(folder):
    values = []
    for name in FILES:
        try:
            stat = (folder / name).stat()
            values.append((stat.st_mtime_ns, stat.st_size))
        except FileNotFoundError:
            values.append(None)
    from app.services.task_recycle_bin import marker_path
    try:
        stat = marker_path(folder.name).stat()
        values.append((stat.st_mtime_ns, stat.st_size))
    except FileNotFoundError:
        values.append(None)
    return json.dumps(values)


def read_json(path, default):
    try:
        return json.loads(path.read_text('utf-8'))
    except FileNotFoundError:
        return default


def upsert(db, folder):
    from app.services.task_query_service import _task_time_in_china
    stamp = fingerprint(folder)
    params = read_json(folder / 'params.json', None)
    if params is None:
        db.execute('DELETE FROM tasks WHERE task_id=?', (folder.name,))
        return
    if not isinstance(params, dict):
        raise ValueError('Invalid task parameters')
    events = read_json(folder / FILES[1], [])
    metadata = read_json(folder / FILES[2], {})
    if not isinstance(events, list) or not isinstance(metadata, dict):
        raise ValueError('Invalid task metadata')
    events = [event for event in events if isinstance(event, dict)]
    created = _task_time_in_china(params)
    event_types = {event.get('event_type') for event in events}
    user_values = [str(params.get(key) or '').casefold() for key in ('customer_email', 'customer_id', 'storefront_visitor_id')]
    user_values.extend(str(event.get('customer_email') or '').casefold() for event in events)
    ip = str(params.get('client_ip') or '').strip()
    # A concurrent file rewrite must not mark partially read data as current.
    if stamp != fingerprint(folder):
        return
    from app.services.task_recycle_bin import metadata as recycle_metadata
    trashed_at = recycle_metadata(folder.name).get('trashed_at', '')
    db.execute('INSERT OR REPLACE INTO tasks VALUES (?,?,?,?,?,?,?,?,?,?,?)', (
        folder.name, folder.name.casefold(), created.timestamp() if created else None,
        ip, ip.casefold(), '\n'.join(user_values), str(params.get('api_provider') or '').casefold(),
        int(bool(metadata.get('starred'))), sum(1 << i for i, event in enumerate(EVENTS) if event in event_types), stamp, trashed_at))


def refresh_task(task_dir):
    """Write-through hook; an index failure never fails image generation."""
    folder = Path(task_dir).resolve()
    if folder.parent != Path(DirectoryConfig.TASKS_DIR).resolve() or folder.name.startswith('.'):
        return
    try:
        with _lock, database() as db:
            db.execute('BEGIN IMMEDIATE')
            upsert(db, folder)
    except (OSError, ValueError, TypeError, sqlite3.Error):
        logger.exception('Task list index update failed for %s; reconciliation will retry', folder.name)


def reconcile():
    """Initial backfill and recovery for external edits/deletes; only reread changed files."""
    with _lock, database() as db:
        db.execute('BEGIN IMMEDIATE')
        known = {row['task_id']: row['fingerprint'] for row in db.execute('SELECT task_id,fingerprint FROM tasks')}
        seen = set()
        for folder in Path(DirectoryConfig.TASKS_DIR).iterdir():
            if folder.name.startswith('.') or folder.is_symlink() or not folder.is_dir():
                continue
            seen.add(folder.name)
            try:
                if known.get(folder.name) != fingerprint(folder):
                    upsert(db, folder)
            except (OSError, ValueError, TypeError):
                logger.warning('Skipping incomplete task during index recovery: %s', folder.name)
        db.executemany('DELETE FROM tasks WHERE task_id=?', [(key,) for key in known.keys() - seen])
        db.execute("INSERT OR REPLACE INTO metadata VALUES ('ready','1')")


def ensure_ready():
    with _lock, database() as db:
        ready = db.execute("SELECT value FROM metadata WHERE key='ready'").fetchone()
    if not ready:
        reconcile()


def select_tasks(page=None, page_size=None, client_ip_query=None, user_query=None,
                 task_id_query=None, provider_query=None, conversion_statuses=None,
                 time_range=None, starred_only=False, recycle_bin=False):
    from app.services.time_settings import effective_timezone
    from app.services.task_query_service import now_china
    ensure_ready()
    clauses, values = ["trashed_at != ''" if recycle_bin else "trashed_at = ''"], []
    for column, value in [('ip_search', client_ip_query), ('user_search', user_query), ('task_search', task_id_query), ('provider', provider_query)]:
        value = str(value or '').strip().casefold()
        if value:
            # Literal substring matching, including % and _, matches the old list.
            clauses.append(f'instr({column}, ?) > 0'); values.append(value)
    mask = sum(1 << i for i, event in enumerate(EVENTS) if event in (conversion_statuses or []))
    if mask:
        clauses.append('(conversions & ?) != 0'); values.append(mask)
    if starred_only:
        clauses.append('starred=1')
    now = now_china().astimezone(effective_timezone())
    if time_range in ('today', 'yesterday'):
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        if time_range == 'yesterday': start -= timedelta(days=1)
        clauses.append('created >= ? AND created < ?'); values.extend([start.timestamp(), (start + timedelta(days=1)).timestamp()])
    elif time_range in ('this_week', 'this_month'):
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        start = start - timedelta(days=now.weekday()) if time_range == 'this_week' else start.replace(day=1)
        clauses.append('created >= ? AND created <= ?'); values.extend([start.timestamp(), now.timestamp()])
    elif time_range in ('last_7_days', 'last_30_days'):
        clauses.append('created >= ?'); values.append((now - timedelta(days=7 if time_range == 'last_7_days' else 30)).timestamp())
    where = ' WHERE ' + ' AND '.join(clauses) if clauses else ''
    with database() as db:
        db.execute('BEGIN')
        total = db.execute('SELECT COUNT(*) FROM tasks' + where, values).fetchone()[0]
        statistics = dict(db.execute('SELECT COUNT(*) AS total, ' + ', '.join(
            f'COALESCE(SUM((conversions & {1 << i}) != 0), 0) AS {event}'
            for i, event in enumerate(EVENTS)) + ' FROM tasks' + where, values).fetchone())
        size = max(10, min(int(page_size), 100)) if page_size is not None else total or 1
        pages = max(1, (total + size - 1) // size)
        current = max(1, min(int(page or 1), pages)) if page_size is not None else 1
        rows = db.execute('''SELECT task_id, ip,
          ROW_NUMBER() OVER (PARTITION BY ip ORDER BY created, task_id) AS ip_sequence,
          COUNT(*) OVER (PARTITION BY ip) AS ip_total
          FROM tasks''' + where + ' ORDER BY created DESC, task_id DESC LIMIT ? OFFSET ?',
          [*values, size, (current - 1) * size]).fetchall()
    return rows, dict(statistics=statistics, total=total, page=current, page_size=size, total_pages=pages,
                      has_previous=current > 1, has_next=current < pages)


async def worker():
    while True:
        try:
            await asyncio.to_thread(reconcile)
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            return
        except Exception:
            logger.exception('Task list index reconciliation failed')
            await asyncio.sleep(60)
