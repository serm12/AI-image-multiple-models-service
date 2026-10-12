"""Display/filter timezone preferences; stored task timestamps remain immutable."""
import asyncio
import json
import os
import threading
import time
import uuid
from pathlib import Path
from zoneinfo import ZoneInfo, available_timezones
from app.core.config import DirectoryConfig

_lock = threading.Lock()
_sync_lock = asyncio.Lock()
SHOP_TIMEZONE_QUERY = 'query AppShopTimezone { shop { ianaTimezone } }'


def path():
    return Path(DirectoryConfig.TASKS_DIR) / '.time-settings.json'


def read():
    try:
        return json.loads(path().read_text('utf-8'))
    except (OSError, ValueError):
        return {'mode': 'shop', 'custom_timezone': 'UTC', 'shop_timezone': '', 'synced_at': 0}


def update(values):
    with _lock:
        data = {**read(), **values}
        target = path(); target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + '.' + uuid.uuid4().hex + '.tmp')
        temporary.write_text(json.dumps(data), 'utf-8')
        os.replace(temporary, target)
        return data


def effective_timezone():
    data = read()
    return ZoneInfo((data.get('shop_timezone') if data.get('mode', 'shop') == 'shop'
                     else data.get('custom_timezone')) or 'UTC')


async def sync_shop_timezone(force=False):
    if _sync_lock.locked():
        return
    async with _sync_lock:
        if not force and time.time() - read().get('synced_at', 0) < 600:
            return
        from app.services.shopify_customers import query
        async with asyncio.timeout(15):
            result = await query(SHOP_TIMEZONE_QUERY, {})
        name = result['shop']['ianaTimezone']
        ZoneInfo(name)
        update({'shop_timezone': name, 'synced_at': time.time()})


def settings_response():
    data = read()
    return {**data, 'effective_timezone': effective_timezone().key,
            'timezones': sorted(available_timezones())}
