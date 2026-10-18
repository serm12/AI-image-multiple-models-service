"""Shopify order views and Printful draft fulfillment. Never confirms production."""
import asyncio
import io
import json
import os
import sqlite3
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import quote

import httpx
from PIL import Image, ImageOps
from fastapi import HTTPException
from app.core.config import DirectoryConfig, APIConfig
from app.services import provider_credentials as credentials
from app.services.shopify_customers import query
from app.services.shopify_orders import matched_task_ids
from app.services.shopify_order_images import original_image

LIST_QUERY = '''query ProductionOrders($after: String, $query: String) {
 orders(first: 25, after: $after, query: $query, sortKey: CREATED_AT, reverse: true) {
  nodes { id name createdAt displayFinancialStatus displayFulfillmentStatus
    email customer { firstName lastName defaultEmailAddress { emailAddress } } shippingAddress { name } }
  pageInfo { hasNextPage endCursor }
 }
}'''
DETAIL_QUERY = '''query ProductionOrder($id: ID!, $after: String) {
 order(id: $id) { id name cancelledAt displayFinancialStatus
 email customer { firstName lastName defaultEmailAddress { emailAddress } }
 shippingAddress { name address1 address2 city zip provinceCode countryCodeV2 phone }
 lineItems(first: 100, after: $after) { nodes {
 id title variantTitle sku unfulfilledQuantity requiresShipping
 customAttributes { key value }
 } pageInfo { hasNextPage endCursor } }
 }
}'''
# White wrap templates from printful-mcp/image-processing.js; never infer a variant.
TEMPLATES = {5: (7424, 9472, 640), 6: (9472, 11520, 640),
             7: (10496, 13568, 640), 19316: (8100, 10500, 450)}

@contextmanager
def database():
    path = Path(DirectoryConfig.TASKS_DIR) / '.production.sqlite3'
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path, timeout=10)
    db.execute('CREATE TABLE IF NOT EXISTS pushes (id TEXT PRIMARY KEY, result TEXT NOT NULL)')
    db.execute('CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, state TEXT NOT NULL, started REAL NOT NULL, detail TEXT NOT NULL)')
    try:
        with db:
            yield db
    finally:
        db.close()

def job(order_id):
    with database() as db:
        row = db.execute('SELECT state, started, detail FROM jobs WHERE id=?', (order_id,)).fetchone()
    if not row:
        return {'state': 'idle'}
    return {'state': row[0], 'started': row[1], 'detail': row[2]}

def start_job(order_id):
    with database() as db:
        db.execute('BEGIN IMMEDIATE')
        row = db.execute('SELECT state, started FROM jobs WHERE id=?', (order_id,)).fetchone()
        if row and row[0] == 'running':
            # A stale job is not automatically retried: upstream submission may have succeeded.
            raise HTTPException(409, '该订单仍在处理，或服务重启导致中断。请先查询 Printful 状态。')
        db.execute('INSERT OR REPLACE INTO jobs VALUES (?,?,?,?)', (order_id, 'running', time.time(), ''))

def finish_job(order_id, state, detail=''):
    with database() as db:
        db.execute('UPDATE jobs SET state=?, detail=? WHERE id=?', (state, detail, order_id))

async def run_job(order_id, selection):
    try:
        await push(order_id, selection)
        finish_job(order_id, 'completed')
    except HTTPException as error:
        finish_job(order_id, 'failed', str(error.detail))
    except Exception:
        finish_job(order_id, 'failed', '推送未完成，请查询 Printful 状态后重试。已有草稿会被复用。')

def history(order_id):
    with database() as db:
        row = db.execute('SELECT result FROM pushes WHERE id=?', (order_id,)).fetchone()
    return json.loads(row[0]) if row else None

def record(order_id, value):
    with database() as db:
        db.execute('INSERT OR REPLACE INTO pushes VALUES (?,?)', (order_id, json.dumps(value)))

def config():
    return {key: credentials.resolve(key, os.getenv(key, '')) for key in ('PRINTFUL_TOKEN', 'PRINTFUL_STORE_ID')}

async def pf(method, path, body=None, missing=False):
    settings = config()
    if not settings['PRINTFUL_TOKEN']:
        raise HTTPException(400, '请先在设置中配置 Printful Private Token')
    headers = {'Authorization': 'Bearer ' + settings['PRINTFUL_TOKEN']}
    if settings['PRINTFUL_STORE_ID']:
        headers['X-PF-Store-Id'] = settings['PRINTFUL_STORE_ID']
    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.request(method, 'https://api.printful.com' + path, headers=headers, json=body)
    if missing and response.status_code == 404:
        return None
    if response.is_error:
        raise HTTPException(502, f'Printful 请求失败 ({response.status_code})，请检查权限、Store ID 和商品规格')
    return response.json()['result']

async def shopify(document, variables):
    try:
        return await query(document, variables)
    except RuntimeError:
        raise HTTPException(502, '无法读取 Shopify 订单，请在 Shopify 中重新打开应用并检查 read_orders 和地址访问权限') from None

async def detail(order_id):
    after, lines = None, []
    while True:
        order = (await shopify(DETAIL_QUERY, {'id': 'gid://shopify/Order/' + order_id, 'after': after})).get('order')
        if not order:
            raise HTTPException(404, '订单不存在或没有访问权限')
        lines.extend(order['lineItems']['nodes'])
        info = order['lineItems']['pageInfo']
        if not info['hasNextPage']:
            break
        after = info['endCursor']
    for line in lines:
        line['images'] = [{'task_id': task, **image} for task in sorted(matched_task_ids({'lineItems': {'nodes': [line]}}))
                          if (image := original_image(task))]
    order['items'] = lines
    del order['lineItems']
    order['production'] = history(order_id)
    order['job'] = job(order_id)
    return order

def recipient(address):
    if not address or any(not address.get(k) for k in ('name', 'address1', 'city', 'zip', 'countryCodeV2')):
        raise HTTPException(400, '订单缺少完整收货地址，先在 Shopify 中补齐')
    return {dest: address.get(src) or '' for src, dest in [('name', 'name'), ('address1', 'address1'),
        ('address2', 'address2'), ('city', 'city'), ('zip', 'zip'), ('provinceCode', 'state_code'),
        ('countryCodeV2', 'country_code'), ('phone', 'phone')]}

def render_art(data, template, bleed):
    with Image.open(io.BytesIO(data)) as source:
        source = ImageOps.exif_transpose(source).convert('RGB')
        if source.width * source.height > 100_000_000:
            raise ValueError('Image too large')
        if template:
            width, height, border = template
            border = border if bleed else 0
            # Contain preserves all supplied artwork without stretching or cropping.
            front = ImageOps.contain(source, (width - border * 2, height - border * 2), Image.Resampling.LANCZOS)
            result = Image.new('RGB', (width, height), 'white')
            result.paste(front, ((width-front.width)//2, (height-front.height)//2))
        else:
            result = source.copy()
        output = io.BytesIO()
        result.save(output, format='JPEG', quality=95)
        return output.getvalue(), result.size

def download_art(image_url):
    with httpx.Client(timeout=120, follow_redirects=True) as client:
        with client.stream('GET', image_url) as response:
            response.raise_for_status()
            data = bytearray()
            for chunk in response.iter_bytes():
                data.extend(chunk)
                if len(data) > 50_000_000:
                    raise ValueError('Image exceeds 50 MB')
    return data

def prepare(image_url, variant_id, upscale, bleed):
    import fal_client
    key = APIConfig.FAL_API_KEY
    if not key:
        raise HTTPException(400, '处理图片需要先配置 FAL API Key')
    fal = fal_client.SyncClient(key=key)
    manifest = {'source_url': image_url, 'upscale': upscale, 'bleed': bleed}
    data = download_art(image_url)
    if upscale:
        # Like the MCP workflow, bound AuraSR input to avoid multiplying huge
        # originals into files that exceed practical print preparation limits.
        with tempfile.TemporaryDirectory() as folder:
            with Image.open(io.BytesIO(data)) as source:
                if source.width * source.height > 100_000_000:
                    raise ValueError('Image too large')
                normalized = ImageOps.exif_transpose(source).convert('RGB')
                normalized.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
                path = Path(folder) / 'aura-input.png'
                normalized.save(path)
                manifest['normalized_input_dimensions'] = normalized.size
            image_url = fal.upload_file(str(path))
        manifest['normalized_input_url'] = image_url
        handle = fal.submit('fal-ai/aura-sr', arguments={'image_url': image_url, 'upscale_factor': 4,
            'checkpoint': 'v2', 'overlapping_tiles': True})
        manifest['request_id'] = handle.request_id
        image_url = handle.get()['image']['url']
        manifest['upscale_url'] = image_url
        data = download_art(image_url)
    data, dimensions = render_art(data, TEMPLATES.get(variant_id), bleed)
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / 'print.jpg'
        path.write_bytes(data)
        manifest['file_url'] = fal.upload_file(str(path))
    manifest['dimensions'] = dimensions
    return manifest

_push_lock = asyncio.Lock()

async def push(order_id, selection):
    # Serializes drafts within this app; external_id also prevents duplicate creation on retries.
    if _push_lock.locked():
        raise HTTPException(409, '已有订单正在处理，请稍后重试')
    async with _push_lock:
        order = await detail(order_id)
        if order.get('cancelledAt'):
            raise HTTPException(400, '已取消的订单不能推送')
        address = recipient(order.get('shippingAddress'))
        existing = await pf('GET', '/orders/@' + quote(order_id, safe=''), missing=True)
        if existing and existing['status'] != 'draft':
            raise HTTPException(409, 'Printful 已存在非草稿订单，请在 Printful 中核对，禁止重复创建')
        lines = {line['id']: line for line in order['items']}
        items, manifests = [], []
        if len({item.line_id for item in selection.items}) != len(selection.items):
            raise HTTPException(400, '商品不能重复选择')
        # Validate every line before starting paid image processing.
        for chosen in selection.items:
            line = lines.get(chosen.line_id)
            if not line or not line['requiresShipping'] or line['unfulfilledQuantity'] < 1:
                raise HTTPException(400, '商品不存在、不需要配送或已履约')
            if not any(i['task_id'] == chosen.task_id for i in line['images']):
                raise HTTPException(400, '所选图片不属于该订单商品')
            if chosen.bleed and chosen.variant_id not in TEMPLATES:
                raise HTTPException(400, '此规格没有已验证的出血模板，请关闭出血或选择支持的画布规格')
            catalog = await pf('GET', '/products/variant/' + str(chosen.variant_id))
            if chosen.bleed and catalog.get('product', {}).get('id') != 3:
                raise HTTPException(400, '此商品与画布出血模板不匹配')
        for chosen in selection.items:
            line = lines[chosen.line_id]
            image = next(i for i in line['images'] if i['task_id'] == chosen.task_id)
            manifest = await asyncio.to_thread(prepare, image['url'], chosen.variant_id, chosen.upscale, chosen.bleed) if chosen.upscale or chosen.bleed else {'file_url': image['url'], 'upscale': False, 'bleed': False}
            manifests.append({'line_id': chosen.line_id, 'variant_id': chosen.variant_id, **manifest})
            items.append({'external_id': chosen.line_id.rsplit('/', 1)[-1], 'variant_id': chosen.variant_id,
                          'quantity': line['unfulfilledQuantity'], 'files': [{'url': manifest['file_url']}]})
        body = {'external_id': order_id, 'recipient': address, 'items': items}
        result = await pf('PUT' if existing else 'POST',
            ('/orders/' + str(existing['id']) if existing else '/orders') + '?confirm=0', body)
        value = {'id': result['id'], 'status': result['status'], 'costs': result.get('costs'), 'files': manifests}
        record(order_id, value)
        # Processing may outlast this poll; expose pending/error states without claiming files are ready.
        for attempt in range(5):
            current = await pf('GET', '/orders/' + str(result['id']))
            states = [{'id': f.get('id'), 'status': f.get('status')} for item in current.get('items', []) for f in item.get('files', [])]
            value.update({'status': current['status'], 'costs': current.get('costs'), 'file_statuses': states})
            record(order_id, value)
            if states and all(f['status'] in ('ok', 'failed') for f in states):
                break
            if attempt < 4:
                await asyncio.sleep(2)
        return value
