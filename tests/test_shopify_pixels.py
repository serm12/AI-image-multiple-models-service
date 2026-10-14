import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock
from app.services.shopify_pixels import reconcile, settings, SCOPES, PIXEL, CREATE, UPDATE, execute_with_token

class PixelTests(unittest.IsolatedAsyncioTestCase):
    def scopes(self, *values):
        return {'currentAppInstallation': {'accessScopes': [{'handle': value} for value in values]}}

    async def test_missing_scopes_never_create(self):
        execute = AsyncMock(return_value=self.scopes('read_orders'))
        self.assertEqual(await reconcile(execute), {'status': 'authorization_required'})
        execute.assert_awaited_once_with(SCOPES, {})

    async def test_create_then_matching_settings_are_noop(self):
        pixel = {'id': 'gid://shopify/WebPixel/1', 'settings': json.dumps(settings())}
        execute = AsyncMock(side_effect=[self.scopes('write_pixels','read_customer_events'),
            {'webPixel': None}, {'webPixelCreate': {'webPixel': pixel, 'userErrors': []}}])
        self.assertEqual((await reconcile(execute))['status'], 'active')
        self.assertEqual(execute.await_args_list[-1].args[0], CREATE)
        execute = AsyncMock(side_effect=[self.scopes('write_pixels','read_customer_events'), {'webPixel': pixel}])
        self.assertEqual((await reconcile(execute))['status'], 'active')
        self.assertEqual(execute.await_count, 2)

    async def test_update_preserves_id_and_rejection_is_not_success(self):
        execute = AsyncMock(side_effect=[self.scopes('write_pixels','read_customer_events'),
            {'webPixel': {'id': 'pixel-id', 'settings': {'shopDomain':'old.myshopify.com'}}},
            {'webPixelUpdate': {'webPixel': None, 'userErrors': [{'message': 'rejected'}]}}])
        with self.assertRaises(RuntimeError): await reconcile(execute)
        self.assertEqual(execute.await_args.args[0], UPDATE)
        self.assertEqual(execute.await_args.args[1]['id'], 'pixel-id')

    async def test_only_missing_pixel_error_allows_create(self):
        for message, missing in [('No web pixel was found for this app.', True), ('Access denied', False)]:
            sdk = SimpleNamespace(admin_graphql_request_async=AsyncMock(return_value=SimpleNamespace(
                ok=False, response=SimpleNamespace(body=json.dumps({'errors':[{'message':message}]})))))
            if missing:
                self.assertEqual(await execute_with_token(sdk,'secret',PIXEL,{}), {'webPixel':None})
            else:
                with self.assertRaises(RuntimeError): await execute_with_token(sdk,'secret',PIXEL,{})
