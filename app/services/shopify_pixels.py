"""Install/reconcile this app's pixel only through a verified Shopify session."""
import asyncio
import json
import logging
import os
import time

from app.services.shopify_admin import allowed_shop

logger = logging.getLogger(__name__)
_lock = asyncio.Lock()
_last_attempt = 0.0
SCOPES = """query PixelScopes { currentAppInstallation { accessScopes { handle } } }"""
PIXEL = """query ConversionPixel { webPixel { id settings } }"""
CREATE = """mutation CreateConversionPixel($webPixel: WebPixelInput!) {
 webPixelCreate(webPixel: $webPixel) { webPixel { id settings } userErrors { field message } }
}"""
UPDATE = """mutation UpdateConversionPixel($id: ID!, $webPixel: WebPixelInput!) {
 webPixelUpdate(id: $id, webPixel: $webPixel) { webPixel { id settings } userErrors { field message } }
}"""


def settings():
    return {'shopDomain': f'{allowed_shop()}.myshopify.com'}


async def reconcile(execute):
    """An injectable, idempotent reconciliation; missing scopes never mutate."""
    data = await execute(SCOPES, {})
    scopes = {item['handle'] for item in data['currentAppInstallation']['accessScopes']}
    if not {'write_pixels', 'read_customer_events'} <= scopes:
        return {'status': 'authorization_required'}
    pixel = (await execute(PIXEL, {})).get('webPixel')
    desired = settings()
    if pixel:
        current = pixel['settings']
        if isinstance(current, str):
            current = json.loads(current)
        if current == desired:
            return {'status': 'active', 'id': pixel['id']}
    variables = {'webPixel': {'settings': json.dumps(desired)}}
    if pixel:
        variables['id'] = pixel['id']
    operation = UPDATE if pixel else CREATE
    key = 'webPixelUpdate' if pixel else 'webPixelCreate'
    result = (await execute(operation, variables))[key]
    if result.get('userErrors') or not result.get('webPixel'):
        raise RuntimeError('pixel_activation_rejected')
    return {'status': 'active', 'id': result['webPixel']['id']}


async def execute_with_token(sdk, token, document, variables):
    result = await sdk.admin_graphql_request_async(
        query=document, variables=variables, shop=allowed_shop(), access_token=token,
        api_version=os.getenv('SHOPIFY_API_VERSION', '2026-10'))
    if result.ok and result.data is not None:
        return result.data
    # Shopify reports the initial missing pixel as a GraphQL error. Only this
    # specific missing-record response is allowed to trigger creation.
    if document == PIXEL:
        try:
            errors = json.loads(result.response.body).get('errors', [])
            if errors and all(error.get('message', '').rstrip('.') == 'No web pixel was found for this app'
                              for error in errors):
                return {'webPixel': None}
        except (ValueError, TypeError, AttributeError):
            pass
    raise RuntimeError('pixel_query_failed')


async def ensure_pixel(sdk, verified):
    global _last_attempt
    if _lock.locked() or time.monotonic() - _last_attempt < 240:
        return
    async with _lock:
        _last_attempt = time.monotonic()
        try:
            async with asyncio.timeout(15):
                exchanged = await sdk.exchange_using_token_exchange_async(
                    access_mode='online', id_token=verified.id_token,
                    invalid_token_response=verified.invalid_token_response)
                if not exchanged.ok:
                    raise RuntimeError('pixel_token_exchange_failed')
                async def execute(document, variables):
                    return await execute_with_token(sdk, exchanged.access_token.token, document, variables)
                state = await reconcile(execute)
                logger.info('Shopify app pixel: %s', state['status'])
        except Exception as error:
            # Pixel activation must never block access to image tasks. Retry on
            # the next verified session, without logging credentials or PII.
            logger.warning('Shopify app pixel unavailable (%s)', type(error).__name__)
