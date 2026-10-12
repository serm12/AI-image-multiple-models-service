import os
import time
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import jwt
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from shopify_app import ShopifyApp
from app.routers.shopify import router, shopify_dependency
from app.services.shopify_admin import ShopifyAdminMiddleware
from app.services.shopify_customer_images import customer_images, download_original
from app.services import shopify_orders as orders

TASK = '20261011_143719_32424eb5'
CUSTOMER = 'gid://shopify/Customer/123'
SECRET = 'customer-test-secret-not-real'


class CustomerImagesTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        for patcher in [patch.dict(os.environ, {'SHOPIFY_CLIENT_SECRET': SECRET, 'SHOPIFY_API_SECRET': '', 'SHOPIFY_SHOP': 'ai-generate-image'}),
                        patch.object(orders, 'cache_path', return_value=Path(self.temp.name) / 'orders.json')]:
            patcher.start(); self.addCleanup(patcher.stop)
        self.payload = {'id': 1076, 'name': '#1076', 'updated_at': '2026-10-12T09:00:00Z', 'customer': {'id': 123},
                        'financial_status': 'paid', 'line_items': [{'id': 1, 'title': 'Portrait', 'properties': [{'name': '_AI Task ID', 'value': TASK}]}]}
        orders.receive_order_webhook(self.payload, 'created')
        app = FastAPI()
        app.add_middleware(ShopifyAdminMiddleware)
        app.include_router(router)
        app.dependency_overrides[shopify_dependency] = lambda: ShopifyApp(client_id='test-client', client_secret=SECRET)
        self.client = TestClient(app)
        self.addCleanup(self.client.close)

    def token(self, **changes):
        return jwt.encode({'aud': 'test-client', 'dest': 'ai-generate-image.myshopify.com',
                           'sub': CUSTOMER, 'exp': int(time.time()) + 300, **changes}, SECRET, algorithm='HS256')

    def test_customer_auth_rejects_wrong_owner_shop_audience_expiry_and_anonymous(self):
        for claims, status in [({'sub': 'gid://shopify/Customer/456'}, 404), ({'dest': 'other.myshopify.com'}, 403),
                               ({'aud': 'wrong'}, 401), ({'exp': 1}, 401), ({'sub': ''}, 401), ({'sub': '123'}, 401)]:
            with self.subTest(claims=claims):
                response = self.client.get('/shopify/customer/orders/1076/images', headers={'Authorization': f'Bearer {self.token(**claims)}'})
                self.assertEqual(response.status_code, status)

    def test_owner_receives_original_and_attachment_download(self):
        path = Path(self.temp.name) / 'original.png'
        path.write_bytes(b'original-image')
        with patch('app.services.shopify_customer_images.original_image', return_value={'url': 'https://example.com/original.png', 'filename': 'original.png'}), \
             patch('app.services.task_files.resolve_task_generated_original', return_value=str(path)):
            response = self.client.get('/shopify/customer/orders/1076/images', headers={'Authorization': f'Bearer {self.token()}'})
            self.assertEqual(response.status_code, 200)
            image = response.json()['items'][0]['image']
            download = self.client.get(image['download_url'])
            self.assertEqual(download.content, b'original-image')
            self.assertIn('attachment', download.headers['content-disposition'])
            self.assertEqual(response.headers['cache-control'], 'no-store')
            self.assertEqual(response.headers['access-control-allow-origin'], '*')
            # Changing order ownership invalidates even previously-issued links.
            self.payload.update(customer={'id': 456}, updated_at='2026-10-12T10:00:00Z')
            orders.receive_order_webhook(self.payload, 'changed')
            self.assertEqual(self.client.get(image['download_url']).status_code, 404)

    def test_missing_original_and_invalid_download(self):
        with patch('app.services.shopify_customer_images.original_image', return_value=None):
            self.assertIsNone(customer_images('1076', CUSTOMER)['items'][0]['image'])
        self.assertEqual(self.client.get('/shopify/customer/download?token=invalid').status_code, 401)
        preflight = self.client.options('/shopify/customer/orders/1076/images', headers={'Origin': 'null'})
        self.assertEqual(preflight.status_code, 204)

    def test_old_webhook_cannot_replace_customer_ownership(self):
        self.payload.update(customer={'id': 456}, updated_at='2026-10-12T08:00:00Z')
        orders.receive_order_webhook(self.payload, 'stale')
        self.assertEqual(orders.read_order_cache()['customer_orders']['gid://shopify/Order/1076']['customer_id'], CUSTOMER)
