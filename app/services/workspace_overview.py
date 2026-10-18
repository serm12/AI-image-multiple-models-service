"""Public shop identity and local production notifications for authenticated admins."""
import asyncio
import time
from app.services.shopify_admin import allowed_shop
from app.services.shopify_customers import query
from app.services.order_management import database

SHOP_QUERY = "query AdminWorkspaceShop { shop { name myshopifyDomain primaryDomain { url } currencyCode } }"
_shop_cache = {}
_shop_lock = asyncio.Lock()

async def shop_identity():
    domain = allowed_shop() + '.myshopify.com'
    cached = _shop_cache.get(domain)
    if cached and time.time() - cached[0] < 300:
        return cached[1]
    async with _shop_lock:
        cached = _shop_cache.get(domain)
        if cached and time.time() - cached[0] < 300:
            return cached[1]
        try:
            shop = (await asyncio.wait_for(query(SHOP_QUERY, {}), timeout=10))['shop']
            value = {'name': shop['name'], 'domain': shop['myshopifyDomain'], 'url': shop['primaryDomain']['url'], 'currency': shop['currencyCode'], 'state': 'connected'}
            _shop_cache[domain] = (time.time(), value)
            return value
        except Exception:
            return {**(cached[1] if cached else {'name': domain, 'domain': domain, 'url': 'https://' + domain}), 'state': 'unavailable'}

def notifications():
    with database() as db:
        rows = db.execute('SELECT id, state, started FROM jobs ORDER BY started DESC LIMIT 20').fetchall()
    return [{'id': f'production:{order_id}:{state}:{started}', 'kind': 'production', 'state': state,
             'created_at': started, 'order_id': order_id, 'url': '/admin/orders?order=' + order_id}
            for order_id, state, started in rows if order_id.isdigit() and state in ('running', 'completed', 'failed')]
