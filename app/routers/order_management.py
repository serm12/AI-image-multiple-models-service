"""Authenticated production order actions."""
import asyncio
import hashlib
import json
import time
from typing import Annotated
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, Request, Response
from pydantic import BaseModel, Field
from app.services.security import require_admin_login
from app.services import order_management as service
from app.services import provider_credentials as credentials

def no_cache(response: Response):
    response.headers['Cache-Control'] = 'no-store'

router = APIRouter(prefix='/admin/api', dependencies=[Depends(require_admin_login), Depends(no_cache)])
OrderId = Annotated[str, Field(pattern=r'^\d{1,30}$')]

def origin(request):
    if request.headers.get('origin') != str(request.base_url).rstrip('/'):
        raise HTTPException(403, 'Invalid request origin')

def connection_status(value, state=None):
    fingerprint = hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
    with credentials.database() as db:
        db.execute('CREATE TABLE IF NOT EXISTS production_connection (fingerprint TEXT PRIMARY KEY, state TEXT NOT NULL)')
        if state is not None:
            db.execute('DELETE FROM production_connection')
            db.execute('INSERT INTO production_connection VALUES (?, ?)', (fingerprint, state))
        row = db.execute('SELECT state FROM production_connection WHERE fingerprint=?', (fingerprint,)).fetchone()
    return row[0] if row else 'unverified'


@router.get('/production/settings')
def settings():
    value = service.config()
    return {'configured': bool(value['PRINTFUL_TOKEN']), 'store_id': value['PRINTFUL_STORE_ID'], 'connection_status': connection_status(value)}

class SettingsInput(BaseModel):
    token: str = Field(default='', max_length=4096)
    store_id: str = Field(default='', pattern=r'^\d{0,30}$')

@router.post('/production/settings')
async def save_settings(request: Request):
    origin(request)
    try:
        value = SettingsInput.model_validate(await request.json())
        credentials.save({'PRINTFUL_TOKEN': value.token})
        # Empty store ID intentionally clears an account-level store selection.
        with credentials.database() as db:
            db.execute('INSERT OR REPLACE INTO credentials VALUES (?,?)',
                ('PRINTFUL_STORE_ID', credentials.cipher().encrypt(value.store_id.encode()).decode()))
    except ValueError:
        raise HTTPException(400, '密钥格式无效或未配置服务器加密密钥') from None
    return settings()

@router.post('/production/test')
async def test_connection(request: Request):
    origin(request)
    value = service.config()
    try:
        stores = await service.pf('GET', '/stores')
        configured = value['PRINTFUL_STORE_ID']
        if configured and not any(str(store['id']) == configured for store in stores):
            raise HTTPException(400, '此 Token 无法访问所填 Store ID')
    except Exception:
        connection_status(value, 'failed')
        raise
    if service.config() != value:
        raise HTTPException(409, '配置已更改，请重新测试连接')
    connection_status(value, 'connected')
    return {'stores': stores}

@router.get('/orders')
async def orders(after: str | None = None, search: str = Query('', max_length=120)):
    return await service.list_orders(after, search)

@router.get('/orders/{order_id}')
async def order(order_id: OrderId):
    return await service.detail(order_id)

class ItemInput(BaseModel):
    line_id: str = Field(min_length=1, max_length=100)
    task_id: str = Field(pattern=r'^\d{8}_\d{6}_[a-f0-9]{8}$')
    variant_id: int = Field(gt=0)
    upscale: bool = True
    bleed: bool = True

class PushInput(BaseModel):
    items: list[ItemInput] = Field(min_length=1, max_length=100)

@router.post('/orders/{order_id}/printful')
async def push(order_id: OrderId, value: PushInput, request: Request, background: BackgroundTasks):
    origin(request)
    service.start_job(order_id)
    background.add_task(service.run_job, order_id, value)
    return {'state': 'running'}

@router.get('/orders/{order_id}/production-job')
def production_job(order_id: OrderId):
    return {**service.job(order_id), 'production': service.history(order_id)}

@router.post('/orders/{order_id}/printful/status')
async def refresh(order_id: OrderId, request: Request):
    origin(request)
    order = await service.detail(order_id)
    await service.sync_printful(order, force=True)
    job = service.job(order_id)
    if job['state'] == 'running' and time.time() - job['started'] > 3600 and not service._push_lock.locked():
        service.finish_job(order_id, 'failed', '处理已中断，Printful 状态已核对，可以重新推送草稿。')
    return {'orders': order['printful_orders'], 'sync': order['printful_sync']}
