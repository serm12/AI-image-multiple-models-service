"""Read-only order matching by purchased line-item task ID, never by email."""
import asyncio
import json
import logging
import os
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlsplit

from app.core.config import DirectoryConfig
from app.services.shopify_admin import allowed_shop

logger = logging.getLogger(__name__)
_lock = asyncio.Lock()
_last_sync = 0.0
TASK_ID = re.compile(r"\d{8}_\d{6}_[a-f0-9]{8}\Z")
ORDER_QUERY = """query TaskOrders($after: String, $query: String) {
  orders(first: 25, after: $after, query: $query) {
    nodes { id name displayFinancialStatus
      lineItems(first: 100) { nodes { customAttributes { key value } }
        pageInfo { hasNextPage endCursor } }
    }
    pageInfo { hasNextPage endCursor }
  }
}"""
LINE_QUERY = """query OrderLines($id: ID!, $after: String) {
  order(id: $id) { lineItems(first: 100, after: $after) {
    nodes { customAttributes { key value } } pageInfo { hasNextPage endCursor }
  } }
}"""


def cache_path():
    return Path(DirectoryConfig.TASKS_DIR) / '.shopify-order-links.json'


def read_order_cache():
    try:
        data = json.loads(cache_path().read_text(encoding='utf-8'))
        return data if data.get('shop') == allowed_shop() else {}
    except (OSError, ValueError, AttributeError):
        return {}


def matched_task_ids(order):
    found = set()
    for line in order['lineItems']['nodes']:
        attrs = {a['key']: a.get('value') or '' for a in line['customAttributes']}
        task_id = attrs.get('_AI Task ID', '').strip()
        # Old orders may predate the hidden task ID. Use only the purchased
        # output URL, never the reference image or the customer's email.
        if not task_id:
            url = urlsplit(attrs.get('Watermark Image URL', ''))
            parts = url.path.strip('/').split('/')
            if url.hostname in {'image-api.shopupup.com', 'image-api-cdn.shopupup.com'} and len(parts) == 3 and parts[0] == 'taskfile':
                task_id = parts[1]
        if TASK_ID.fullmatch(task_id):
            found.add(task_id)
    return found


def save_orders(orders):
    previous = read_order_cache().get('tasks', {})
    updated_ids = {order['id'] for order in orders}
    mapping = {task: [item for item in items if item['id'] not in updated_ids]
               for task, items in previous.items()}
    for order in orders:
        numeric_id = order['id'].removeprefix('gid://shopify/Order/')
        if not numeric_id.isdigit():
            continue
        item = {'id': order['id'], 'name': order['name'],
                'financial_status': order['displayFinancialStatus'],
                'url': f'https://admin.shopify.com/store/paintingify/orders/{numeric_id}'}
        for task_id in matched_task_ids(order):
            mapping.setdefault(task_id, []).append(item)
    data = {'shop': allowed_shop(), 'synced_at': datetime.now(timezone.utc).isoformat(),
            'tasks': {key: value for key, value in mapping.items() if value}}
    path = cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    with temporary.open('w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False)
    os.replace(temporary, path)
    return len(data['tasks'])


async def sync_orders(sdk, verified):
    """Exchange a verified online session; credentials remain only in memory."""
    global _last_sync
    if _lock.locked() or time.monotonic() - _last_sync < 240:
        return
    async with _lock:
        try:
            async with asyncio.timeout(25):
                exchanged = await sdk.exchange_using_token_exchange_async(
                    access_mode='online', id_token=verified.id_token,
                    invalid_token_response=verified.invalid_token_response)
                if not exchanged.ok:
                    raise RuntimeError('token_exchange_failed')

                async def query(document, variables):
                    result = await sdk.admin_graphql_request_async(
                        query=document, shop=allowed_shop(), access_token=exchanged.access_token.token,
                        api_version=os.getenv('SHOPIFY_API_VERSION', '2026-10'), variables=variables)
                    if not result.ok or result.data is None:
                        raise RuntimeError('orders_query_failed')
                    return result.data

                after = None
                orders = []
                since = (datetime.now(timezone.utc) - timedelta(days=59)).date().isoformat()
                while True:
                    connection = (await query(ORDER_QUERY, {'after': after, 'query': f'created_at:>={since}'}))['orders']
                    for order in connection['nodes']:
                        lines = order['lineItems']
                        while lines['pageInfo']['hasNextPage']:
                            lines = (await query(LINE_QUERY, {'id': order['id'], 'after': lines['pageInfo']['endCursor']}))['order']['lineItems']
                            order['lineItems']['nodes'].extend(lines['nodes'])
                        orders.append(order)
                    if not connection['pageInfo']['hasNextPage']:
                        break
                    after = connection['pageInfo']['endCursor']
                count = save_orders(orders)
                _last_sync = time.monotonic()
                logger.info('Shopify orders synchronized: %s task matches', count)
        except Exception as error:
            # Keep existing links on transient failures; do not log tokens or
            # customer attributes. A future verified session retries the sync.
            logger.warning('Shopify order sync failed (%s)', type(error).__name__)
