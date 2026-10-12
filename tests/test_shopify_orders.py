import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from app.services import shopify_orders as orders

TASK = '20261011_143719_32424eb5'


def order(number='1076', attributes=None):
    return {'id': f'gid://shopify/Order/{number}', 'name': f'#{number}',
            'displayFinancialStatus': 'PAID', 'lineItems': {'nodes': [
                {'customAttributes': attributes if attributes is not None else [{'key': '_AI Task ID', 'value': TASK}]}],
                'pageInfo': {'hasNextPage': False, 'endCursor': None}}}


class OrderMatchingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / 'orders.json'
        self.mock_path = patch.object(orders, 'cache_path', return_value=self.path)
        self.mock_path.start()

    def tearDown(self):
        self.mock_path.stop()
        self.temp.cleanup()

    def test_exact_id_matches_but_email_and_reference_do_not(self):
        self.assertEqual(orders.matched_task_ids(order()), {TASK})
        unrelated = order(attributes=[{'key': 'email', 'value': 'same@example.com'},
                                     {'key': 'Reference Image', 'value': f'https://image-api.shopupup.com/taskfile/{TASK}/image.png'}])
        self.assertEqual(orders.matched_task_ids(unrelated), set())

    def test_legacy_purchased_output_and_invalid_path(self):
        self.assertEqual(orders.matched_task_ids(order(attributes=[{
            'key': 'Watermark Image URL', 'value': f'https://image-api-cdn.shopupup.com/taskfile/{TASK}/output.png'}])), {TASK})
        self.assertEqual(orders.matched_task_ids(order(attributes=[{'key': '_AI Task ID', 'value': '../secret'}])), set())

    def test_multiple_orders_idempotence_and_status_update(self):
        orders.save_orders([order(), order('1077')])
        updated = order()
        updated['displayFinancialStatus'] = 'REFUNDED'
        orders.save_orders([updated])
        links = orders.read_order_cache()['tasks'][TASK]
        self.assertEqual(len(links), 2)
        self.assertEqual(next(x for x in links if x['name'] == '#1076')['financial_status'], 'REFUNDED')
        self.assertNotIn('customAttributes', self.path.read_text())


class OrderSyncTests(unittest.IsolatedAsyncioTestCase):
    async def test_paginated_read_and_no_save_on_failure(self):
        sdk = SimpleNamespace(
            exchange_using_token_exchange_async=AsyncMock(return_value=SimpleNamespace(ok=True, access_token=SimpleNamespace(token='test'))),
            admin_graphql_request_async=AsyncMock(side_effect=[
                SimpleNamespace(ok=True, data={'orders': {'nodes': [order()], 'pageInfo': {'hasNextPage': True, 'endCursor': 'next'}}}),
                SimpleNamespace(ok=False, data=None)]))
        verified = SimpleNamespace(id_token='verified', invalid_token_response=None)
        with patch.object(orders, '_last_sync', 0), patch.object(orders, 'save_orders') as save:
            await orders.sync_orders(sdk, verified)
            save.assert_not_called()
            self.assertEqual(sdk.admin_graphql_request_async.await_count, 2)
            self.assertEqual(sdk.admin_graphql_request_async.call_args.kwargs['variables']['after'], 'next')


if __name__ == '__main__':
    unittest.main()
