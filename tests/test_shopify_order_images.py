import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from fastapi import HTTPException
from app.services.shopify_order_images import get_order_images


class OrderOriginalImagesTests(unittest.IsolatedAsyncioTestCase):
    async def test_reads_current_order_and_never_uses_reference_image(self):
        task = '20261011_143719_32424eb5'
        order = {'id': 'gid://shopify/Order/123', 'name': '#123', 'lineItems': {
            'nodes': [{'id': 'line1', 'title': 'Portrait', 'customAttributes': [
                {'key': '_AI Task ID', 'value': task},
                {'key': 'Reference Image', 'value': 'https://example.com/reference.jpg'}]}],
            'pageInfo': {'hasNextPage': False, 'endCursor': None}}}
        sdk = SimpleNamespace(
            exchange_using_token_exchange_async=AsyncMock(return_value=SimpleNamespace(ok=True, access_token=SimpleNamespace(token='private-test-token'))),
            admin_graphql_request_async=AsyncMock(return_value=SimpleNamespace(ok=True, data={'order': order})))
        verified = SimpleNamespace(id_token='verified', invalid_token_response=None)
        with patch('app.services.shopify_order_images.original_image', return_value={'url': 'https://example.com/original.png', 'filename': 'original.png'}) as original:
            response = await get_order_images(sdk, verified, '123')
            original.assert_called_once_with(task)
            self.assertEqual(response['items'][0]['image']['filename'], 'original.png')
            self.assertEqual(sdk.admin_graphql_request_async.call_args.kwargs['variables']['id'], 'gid://shopify/Order/123')
        sdk.admin_graphql_request_async.return_value = SimpleNamespace(ok=True, data={'order': None})
        with self.assertRaises(HTTPException) as error:
            await get_order_images(sdk, verified, '123')
        self.assertEqual(error.exception.status_code, 404)
