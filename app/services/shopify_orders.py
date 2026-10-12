"""Read-only order matching by purchased line-item task ID, never by email."""
import asyncio
import json
import logging
import os
import re
import time
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlsplit

from app.core.config import DirectoryConfig
from app.services.shopify_admin import allowed_shop

logger = logging.getLogger(__name__)
_lock = asyncio.Lock()
_last_sync = 0.0
_write_lock = threading.Lock()
WEBHOOK_URI = 'https://image-api.shopupup.com/shopify/webhooks'
WEBHOOK_TOPICS = ['ORDERS_CREATE', 'ORDERS_PAID', 'ORDERS_UPDATED']
WEBHOOK_QUERY = """query OrderWebhooks($after: String, $uri: String) {
 webhookSubscriptions(first: 100, after: $after, uri: $uri) {
  nodes { id topic uri includeFields } pageInfo { hasNextPage endCursor }
 }
}"""
WEBHOOK_CREATE = """mutation OrderWebhook($topic: WebhookSubscriptionTopic!, $input: WebhookSubscriptionInput!) {
 webhookSubscriptionCreate(topic: $topic, webhookSubscription: $input) {
  webhookSubscription { id topic uri } userErrors { field message }
 }
}"""
TASK_ID = re.compile(r"\d{8}_\d{6}_[a-f0-9]{8}\Z")
WEBHOOK_FIELDS = ['id', 'name', 'updated_at', 'financial_status', 'line_items', 'customer']
WEBHOOK_UPDATE = """mutation UpdateOrderWebhook($id: ID!, $input: WebhookSubscriptionInput!) {
 webhookSubscriptionUpdate(id: $id, webhookSubscription: $input) {
  webhookSubscription { id } userErrors { field message }
 }
}"""
ORDER_QUERY = """query TaskOrders($after: String, $query: String) {
  orders(first: 25, after: $after, query: $query) {
    nodes { id name updatedAt displayFinancialStatus customer { id }
      lineItems(first: 100) { nodes { id title customAttributes { key value } }
        pageInfo { hasNextPage endCursor } }
    }
    pageInfo { hasNextPage endCursor }
  }
}"""
LINE_QUERY = """query OrderLines($id: ID!, $after: String) {
  order(id: $id) { lineItems(first: 100, after: $after) {
    nodes { id title customAttributes { key value } } pageInfo { hasNextPage endCursor }
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


def save_orders(orders, event_id=None, subscriptions_ready=False):
    with _write_lock:
        return _save_orders(orders, event_id, subscriptions_ready)


def _save_orders(orders, event_id=None, subscriptions_ready=False):
    cache = read_order_cache()
    event_ids = cache.get('event_ids', [])
    if event_id and event_id in event_ids:
        return len(cache.get('tasks', {}))
    revisions = cache.get('revisions', {})
    customer_orders = cache.get('customer_orders', {})
    orders = [order for order in orders if not revisions.get(order['id']) or
              (order.get('updatedAt') == revisions[order['id']] and order['id'] not in customer_orders and 'customer' in order) or
              datetime.fromisoformat(order.get('updatedAt', '1970-01-01T00:00:00+00:00').replace('Z', '+00:00')) >
              datetime.fromisoformat(revisions[order['id']].replace('Z', '+00:00'))]
    previous = cache.get('tasks', {})
    updated_ids = {order['id'] for order in orders}
    mapping = {task: [item for item in items if item['id'] not in updated_ids]
               for task, items in previous.items()}
    for order in orders:
        if order.get('updatedAt'):
            revisions[order['id']] = order['updatedAt']
        numeric_id = order['id'].removeprefix('gid://shopify/Order/')
        if not numeric_id.isdigit():
            continue
        item = {'id': order['id'], 'name': order['name'],
                'financial_status': order['displayFinancialStatus'],
                'url': f'https://admin.shopify.com/store/paintingify/orders/{numeric_id}'}
        customer_orders[order['id']] = {
            'customer_id': (order.get('customer') or {}).get('id'),
            'name': order['name'],
            'items': [{'line_item_id': line.get('id', ''), 'title': line.get('title', 'Your artwork'), 'task_id': task}
                      for line in order['lineItems']['nodes']
                      for task in sorted(matched_task_ids({'lineItems': {'nodes': [line]}}))]}
        for task_id in matched_task_ids(order):
            mapping.setdefault(task_id, []).append(item)
    if event_id:
        event_ids = (event_ids + [event_id])[-2000:]
    data = {'shop': allowed_shop(), 'synced_at': datetime.now(timezone.utc).isoformat(),
            'revisions': revisions, 'event_ids': event_ids, 'customer_orders': customer_orders,
            'webhook_topics': WEBHOOK_TOPICS if subscriptions_ready else cache.get('webhook_topics', []),
            'tasks': {key: value for key, value in mapping.items() if value}}
    path = cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    with temporary.open('w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False)
    os.replace(temporary, path)
    return len(data['tasks'])


def receive_order_webhook(payload, event_id):
    """Called only after SDK HMAC verification and the shop allowlist check."""
    numeric_id = str(payload.get('id', ''))
    if (not numeric_id.isdigit() or not payload.get('name') or not payload.get('updated_at')
            or not isinstance(payload.get('line_items'), list)):
        raise ValueError('Invalid order payload')
    timestamp = datetime.fromisoformat(payload['updated_at'].replace('Z', '+00:00'))
    if timestamp.tzinfo is None:
        raise ValueError('Invalid order timestamp')
    order = {'id': f'gid://shopify/Order/{numeric_id}', 'name': str(payload['name']),
             'customer': {'id': f"gid://shopify/Customer/{payload['customer']['id']}"} if (payload.get('customer') or {}).get('id') else None,
             'updatedAt': timestamp.isoformat(),
             'displayFinancialStatus': str(payload.get('financial_status') or '').upper(),
             'lineItems': {'nodes': [{'id': str(line.get('id', '')), 'title': line.get('title', 'Your artwork'), 'customAttributes': [
                 {'key': p.get('name', ''), 'value': p.get('value', '')}
                 for p in (line.get('properties') or [])]}
                 for line in payload.get('line_items', [])]}}
    return save_orders([order], event_id=event_id)


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

                # Shopify pushes orders even when no administrator has the app
                # open. This check repairs missing subscriptions on login.
                existing = set()
                cursor = None
                while True:
                    subscriptions = (await query(WEBHOOK_QUERY, {'after': cursor, 'uri': WEBHOOK_URI}))['webhookSubscriptions']
                    existing.update(item['topic'] for item in subscriptions['nodes'] if item['uri'] == WEBHOOK_URI)
                    for subscription in subscriptions['nodes']:
                        if subscription['topic'] in WEBHOOK_TOPICS and subscription['uri'] == WEBHOOK_URI and set(subscription.get('includeFields', [])) != set(WEBHOOK_FIELDS):
                            updated = (await query(WEBHOOK_UPDATE, {'id': subscription['id'], 'input': {'includeFields': WEBHOOK_FIELDS}}))['webhookSubscriptionUpdate']
                            if updated['userErrors']:
                                raise RuntimeError('webhook_update_failed')
                    if not subscriptions['pageInfo']['hasNextPage']:
                        break
                    cursor = subscriptions['pageInfo']['endCursor']
                for topic in WEBHOOK_TOPICS:
                    if topic not in existing:
                        created = (await query(WEBHOOK_CREATE, {'topic': topic, 'input': {
                            'uri': WEBHOOK_URI, 'format': 'JSON',
                            'includeFields': WEBHOOK_FIELDS}}))['webhookSubscriptionCreate']
                        if created['userErrors'] or not created['webhookSubscription']:
                            raise RuntimeError('webhook_registration_failed')

                # Reconciliation is only a fallback for historical/missed events.
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
                count = save_orders(orders, subscriptions_ready=True)
                _last_sync = time.monotonic()
                logger.info('Shopify orders synchronized: %s task matches', count)
        except Exception as error:
            # Keep existing links on transient failures; do not log tokens or
            # customer attributes. A future verified session retries the sync.
            logger.warning('Shopify order sync failed (%s)', type(error).__name__)
