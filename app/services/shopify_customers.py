"""Durable, event-driven customer synchronization for the preview signup form."""
import asyncio
import base64
import hashlib
import json
import logging
import os
import re
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from cryptography.fernet import Fernet, InvalidToken
from app.core.config import DirectoryConfig
from app.services.shopify_admin import allowed_shop
from app.services.shopify_client import get_shopify_app

logger = logging.getLogger(__name__)
wake = asyncio.Event()
_authorization_lock = asyncio.Lock()
CUSTOMERS = '''query PreviewCustomer($query: String!) {
 customers(first: 10, query: $query) { nodes { id email emailMarketingConsent { marketingState } } }
}'''
CREATE = '''mutation PreviewCustomerCreate($input: CustomerInput!) {
 customerCreate(input: $input) { customer { id email emailMarketingConsent { marketingState } } userErrors { field message } }
}'''
SUBSCRIBE = '''mutation PreviewCustomerSubscribe($input: CustomerEmailMarketingConsentUpdateInput!) {
 customerEmailMarketingConsentUpdate(input: $input) { customer { id email emailMarketingConsent { marketingState } } userErrors { field message } }
}'''


@contextmanager
def database():
    path = Path(DirectoryConfig.TASKS_DIR) / '.shopify-customer-sync.sqlite3'
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path, timeout=10)
    db.row_factory = sqlite3.Row
    try:
        db.execute('CREATE TABLE IF NOT EXISTS config (key TEXT PRIMARY KEY, value TEXT NOT NULL)')
        db.execute('''CREATE TABLE IF NOT EXISTS jobs (
            task_id TEXT PRIMARY KEY, email TEXT NOT NULL, consent_at TEXT NOT NULL,
            state TEXT NOT NULL DEFAULT 'queued', attempts INTEGER NOT NULL DEFAULT 0,
            retry_at REAL NOT NULL DEFAULT 0, customer_id TEXT NOT NULL DEFAULT '',
            detail TEXT NOT NULL DEFAULT '', updated_at TEXT NOT NULL DEFAULT '')''')
        yield db
        db.commit()
    finally:
        db.close()


def cipher():
    secret = os.getenv('SHOPIFY_API_SECRET') or os.getenv('SHOPIFY_CLIENT_SECRET', '')
    if not secret:
        raise RuntimeError('shopify_not_configured')
    return Fernet(base64.urlsafe_b64encode(hashlib.sha256(('customer-sync:' + secret).encode()).digest()))


def saved_token():
    with database() as db:
        row = db.execute("SELECT value FROM config WHERE key='token'").fetchone()
    if not row:
        return None
    value = json.loads(cipher().decrypt(row['value'].encode()))
    return value['token'] if value['shop'] == allowed_shop() else None


def clear_token():
    with database() as db:
        db.execute("DELETE FROM config WHERE key='token'")


async def authorize(sdk, verified):
    """Persist an encrypted offline token after Shopify verifies this store's admin."""
    if _authorization_lock.locked():
        return
    async with _authorization_lock:
        try:
            try:
                if await asyncio.to_thread(saved_token):
                    return
            except InvalidToken:
                await asyncio.to_thread(clear_token)
            result = await sdk.exchange_using_token_exchange_async(
                access_mode='offline', expiring=False, id_token=verified.id_token,
                invalid_token_response=verified.invalid_token_response)
            if not result.ok:
                return
            scopes = result.access_token.scope
            if 'write_customers' not in scopes.split(','):
                return
            encrypted = cipher().encrypt(json.dumps({'shop': allowed_shop(), 'token': result.access_token.token}).encode()).decode()
            with database() as db:
                db.execute("INSERT OR REPLACE INTO config VALUES ('token', ?)", (encrypted,))
            await asyncio.to_thread(backfill)
            wake.set()
        except Exception as error:
            logger.warning('Customer sync authorization unavailable: %s', type(error).__name__)


def enqueue(task_id, event):
    # Only the dedicated newsletter submission is evidence of signup intent.
    if not re.fullmatch(r'\d{8}_\d{6}_[a-f0-9]{8}', task_id):
        return False
    if (event.get('event_type') != 'newsletter_result'
            or event.get('source') != 'theme_newsletter_response'
            or event.get('subscription_status') not in {'accepted', 'failed', 'verification_required', 'unconfirmed'}):
        return False
    email = str(event.get('customer_email') or '').strip().lower()
    if len(email) > 254 or not re.fullmatch(r"[a-z0-9.!#$%&'*+/=?^_`{|}~-]+@[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?\.[a-z]{2,63}", email):
        return False
    task = Path(DirectoryConfig.TASKS_DIR) / task_id
    try:
        params = json.loads((task / 'params.json').read_text('utf-8'))
    except (OSError, ValueError):
        return False
    if email != str(params.get('customer_email') or '').strip().lower():
        return False
    consent_at = event.get('recorded_at')
    try:
        stamp = datetime.fromisoformat(consent_at)
        if stamp.tzinfo is None:
            return False
    except (TypeError, ValueError):
        return False
    with database() as db:
        inserted = db.execute('INSERT OR IGNORE INTO jobs(task_id,email,consent_at) VALUES (?,?,?)',
                             (task_id, email, consent_at)).rowcount
    return bool(inserted)


def backfill():
    with database() as db:
        if db.execute("SELECT 1 FROM config WHERE key='backfilled-v1'").fetchone():
            return
    for path in Path(DirectoryConfig.TASKS_DIR).glob('*/storefront_events.json'):
        try:
            for event in json.loads(path.read_text('utf-8')):
                enqueue(path.parent.name, event)
        except (ValueError, OSError, TypeError):
            continue
    with database() as db:
        db.execute("INSERT OR REPLACE INTO config VALUES ('backfilled-v1', 'yes')")


async def query(document, variables):
    token = await asyncio.to_thread(saved_token)
    if not token:
        raise RuntimeError('authorization_required')
    result = await get_shopify_app().admin_graphql_request_async(query=document,
        variables=variables, shop=allowed_shop(), access_token=token,
        api_version=os.getenv('SHOPIFY_API_VERSION', '2026-10'))
    if not result.ok or result.data is None or getattr(result, 'errors', None):
        raise RuntimeError('shopify_api_unavailable')
    return result.data


async def synchronize(email, consent_at, execute=query):
    async def lookup():
        data = await execute(CUSTOMERS, {'query': 'email:' + json.dumps(email)})
        return next((c for c in data['customers']['nodes'] if str(c.get('email') or '').lower() == email), None)
    customer = await lookup()
    consent = {'marketingState': 'SUBSCRIBED', 'marketingOptInLevel': 'SINGLE_OPT_IN', 'consentUpdatedAt': consent_at}
    if customer is None:
        data = (await execute(CREATE, {'input': {'email': email, 'tags': ['newsletter', 'ai-preview'],
                                               'emailMarketingConsent': consent}}))['customerCreate']
        customer = data.get('customer')
        if not customer:
            # Another request or Shopify's form may have created it concurrently.
            customer = await lookup()
            if not customer:
                raise RuntimeError('customer_create_rejected')
    state = (customer.get('emailMarketingConsent') or {}).get('marketingState')
    if state == 'SUBSCRIBED':
        return 'subscribed', customer['id'], 'Shopify 已确认邮件订阅'
    if state in {'UNSUBSCRIBED', 'REDACTED', 'INVALID', 'PENDING'}:
        return 'preserved', customer['id'], '保留 Shopify 原有退订、无效或待确认状态：' + state
    data = (await execute(SUBSCRIBE, {'input': {'customerId': customer['id'],
                                                'emailMarketingConsent': consent}}))['customerEmailMarketingConsentUpdate']
    if data.get('userErrors') or not data.get('customer'):
        raise RuntimeError('subscription_update_rejected')
    if (data['customer'].get('emailMarketingConsent') or {}).get('marketingState') != 'SUBSCRIBED':
        raise RuntimeError('subscription_not_confirmed')
    return 'subscribed', customer['id'], 'Shopify 已确认邮件订阅'


async def process_pending(limit=20):
    # One worker processes jobs serially, keeping same-email creates idempotent.
    with database() as db:
        jobs = db.execute("SELECT * FROM jobs WHERE state IN ('queued','retry') AND retry_at<=? ORDER BY consent_at LIMIT ?",
                          (time.time(), limit)).fetchall()
    for job in jobs:
        attempts = job['attempts'] + 1
        retry_at = 0
        customer_id = job['customer_id']
        try:
            async with asyncio.timeout(45):
                state, customer_id, detail = await synchronize(job['email'], job['consent_at'])
        except Exception as error:
            state = 'retry' if attempts < 12 else 'failed'
            detail = str(error) if isinstance(error, RuntimeError) else type(error).__name__
            retry_at = time.time() + min(3600, 30 * 2 ** min(attempts, 7))
        with database() as db:
            db.execute('UPDATE jobs SET state=?,attempts=?,retry_at=?,customer_id=?,detail=?,updated_at=? WHERE task_id=?',
                       (state, attempts, retry_at, customer_id, detail[:160], datetime.now(timezone.utc).isoformat(), job['task_id']))


async def worker():
    while True:
        try:
            wake.clear()
            if await asyncio.to_thread(saved_token):
                await asyncio.to_thread(backfill)
                await process_pending()
            await asyncio.wait_for(wake.wait(), timeout=30)
        except asyncio.TimeoutError:
            pass
        except asyncio.CancelledError:
            return
        except Exception as error:
            logger.warning('Customer sync worker: %s', type(error).__name__)
            await asyncio.sleep(30)


def summaries():
    with database() as db:
        return {row['task_id']: dict(row) for row in db.execute(
            'SELECT task_id,state,customer_id,detail,updated_at FROM jobs')}
