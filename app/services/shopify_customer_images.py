"""Customer-owned originals, authorized against verified Shopify order events."""
import time
import jwt
from fastapi import HTTPException
from app.services.shopify_admin import allowed_shop, secret
from app.services.shopify_orders import read_order_cache
from app.services.shopify_order_images import original_image


def owned_order(order_id, customer_id):
    order = read_order_cache().get('customer_orders', {}).get(f'gid://shopify/Order/{order_id}')
    if not order or not customer_id or order.get('customer_id') != customer_id:
        raise HTTPException(404, 'Order not found or not accessible')
    return order


def customer_images(order_id, customer_id):
    order = owned_order(order_id, customer_id)
    items = []
    for item in order['items']:
        image = original_image(item['task_id'])
        if image:
            token = jwt.encode({'aud': 'paintingify-download', 'shop': allowed_shop(),
                                'sub': customer_id, 'order': order_id, 'task': item['task_id'],
                                'exp': int(time.time()) + 900}, secret(), algorithm='HS256')
            image = {**image, 'download_url': f'https://image-api.shopupup.com/shopify/customer/download?token={token}'}
        items.append({**item, 'image': image})
    return {'order_name': order['name'], 'items': items}


def download_original(token):
    from app.services.task_files import resolve_task_generated_original
    try:
        claims = jwt.decode(token, secret(), algorithms=['HS256'], audience='paintingify-download',
                            options={'require': ['exp', 'shop', 'sub', 'order', 'task']})
        if claims['shop'] != allowed_shop():
            raise ValueError('shop')
    except (jwt.PyJWTError, ValueError):
        raise HTTPException(401, 'Download expired. Please refresh your order page.') from None
    order = owned_order(claims['order'], claims['sub'])
    if claims['task'] not in {item['task_id'] for item in order['items']}:
        raise HTTPException(404, 'Image not found')
    path = resolve_task_generated_original(claims['task'])
    if not path:
        raise HTTPException(404, 'Original image is not available yet')
    return path
