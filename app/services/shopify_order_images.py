"""Fetch the selected order using the staff member's online Shopify access."""
import os
from pathlib import Path
from urllib.parse import quote

from fastapi import HTTPException

from app.services.shopify_admin import allowed_shop
from app.services.shopify_orders import matched_task_ids

ORDER_IMAGES_QUERY = """query OrderImages($id: ID!, $after: String) {
  order(id: $id) { id name lineItems(first: 100, after: $after) {
    nodes { id title customAttributes { key value } }
    pageInfo { hasNextPage endCursor }
  } }
}"""


def original_image(task_id):
    from app.services.task_files import resolve_task_generated_original
    from app.services.r2_storage import read_r2_mapping
    original = resolve_task_generated_original(task_id)
    if not original:
        return None
    path = Path(original)
    local_url = f'/taskfile/{task_id}/{quote(path.name)}'
    mapping = read_r2_mapping(str(path.parent))
    return {'url': mapping.get(local_url) or f'https://image-api.shopupup.com{local_url}',
            'filename': path.name}


async def get_order_images(sdk, verified, order_id):
    exchange = await sdk.exchange_using_token_exchange_async(
        access_mode='online', id_token=verified.id_token,
        invalid_token_response=verified.invalid_token_response)
    if not exchange.ok:
        raise HTTPException(401, 'Please reopen the app to refresh Shopify authorization')
    after = None
    images = []
    name = ''
    while True:
        result = await sdk.admin_graphql_request_async(
            query=ORDER_IMAGES_QUERY, shop=allowed_shop(), access_token=exchange.access_token.token,
            api_version=os.getenv('SHOPIFY_API_VERSION', '2026-10'),
            variables={'id': f'gid://shopify/Order/{order_id}', 'after': after})
        if not result.ok:
            raise HTTPException(502, 'Unable to read this order from Shopify')
        order = (result.data or {}).get('order')
        if not order:
            raise HTTPException(404, 'Order not found or not accessible')
        name = order['name']
        for line in order['lineItems']['nodes']:
            for task_id in sorted(matched_task_ids({'lineItems': {'nodes': [line]}})):
                original = original_image(task_id)
                images.append({'line_item_id': line['id'], 'title': line['title'], 'task_id': task_id,
                               'image': original, 'status': 'ready' if original else 'original_unavailable'})
        info = order['lineItems']['pageInfo']
        if not info['hasNextPage']:
            break
        after = info['endCursor']
    return {'order_id': order_id, 'order_name': name, 'items': images}
