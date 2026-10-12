import base64
import hashlib
import hmac
import unittest
import os
import time
from unittest.mock import patch, AsyncMock
import jwt

from fastapi import FastAPI
from fastapi.testclient import TestClient
from shopify_app import ShopifyApp
from shopify_app.types import Res

from app.routers.shopify import router, shopify_dependency
from app.services.shopify_http import shopify_response
from app.services.shopify_admin import ShopifyAdminMiddleware, COOKIE, session_shop
from fastapi import Request
from fastapi.responses import HTMLResponse


class ShopifyIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.secret = "test-secret-not-a-real-credential"
        self.env = patch.dict(os.environ, {"SHOPIFY_CLIENT_SECRET": self.secret, "SHOPIFY_SHOP": "ai-generate-image", "SHOPIFY_API_SECRET": ""})
        self.env.start()
        self.order_sync = patch('app.routers.shopify.sync_orders', new_callable=AsyncMock)
        self.order_sync.start()
        app = FastAPI()
        app.add_middleware(ShopifyAdminMiddleware)
        @app.api_route('/admin/tasks', methods=['GET', 'POST'])
        def tasks(request: Request):
            if not getattr(request.state, 'shopify_admin', False):
                return HTMLResponse('Login required', status_code=401)
            return HTMLResponse('<html><head></head><body>Tasks</body></html>')
        app.include_router(router)
        app.dependency_overrides[shopify_dependency] = lambda: ShopifyApp(
            client_id="test-client", client_secret=self.secret
        )
        self.client = TestClient(app, base_url='https://image-api.shopupup.com')

    def tearDown(self):
        self.client.close()
        self.env.stop()
        self.order_sync.stop()

    def headers(self, body, topic="app/uninstalled"):
        signature = base64.b64encode(
            hmac.new(self.secret.encode(), body, hashlib.sha256).digest()
        ).decode()
        return {
            "content-type": "application/json",
            "x-shopify-hmac-sha256": signature,
            "x-shopify-shop-domain": "ai-generate-image.myshopify.com",
            "x-shopify-topic": topic,
        }

    def test_signed_uninstall_is_idempotent_and_preserves_unicode_body(self):
        body = '{ "name": "测试店铺", "id": 1 }'.encode()
        for _ in range(2):
            response = self.client.post('/shopify/webhooks', content=body, headers=self.headers(body))
            self.assertEqual(response.status_code, 200)

    def test_tampered_body_is_rejected(self):
        response = self.client.post('/shopify/webhooks', content=b'{"id":2}', headers=self.headers(b'{"id":1}'))
        self.assertEqual(response.status_code, 401)

    def test_signed_order_is_dispatched_but_wrong_shop_and_tampering_are_not(self):
        body = b'{"id":1076}'
        with patch('app.routers.shopify.receive_order_webhook') as receive:
            headers = self.headers(body, 'orders/paid')
            headers['x-shopify-event-id'] = 'event-1'
            self.assertEqual(self.client.post('/shopify/webhooks', content=body, headers=headers).status_code, 200)
            receive.assert_called_once_with({'id': 1076}, 'event-1')
            receive.reset_mock()
            headers['x-shopify-shop-domain'] = 'other.myshopify.com'
            self.assertEqual(self.client.post('/shopify/webhooks', content=body, headers=headers).status_code, 403)
            self.assertEqual(self.client.post('/shopify/webhooks', content=b'{"id":2}', headers=headers).status_code, 401)
            receive.assert_not_called()

    def test_unsigned_request_is_rejected(self):
        response = self.client.post('/shopify/webhooks', json={"id": 1})
        # Official SDK treats missing required headers as a malformed request.
        self.assertEqual(response.status_code, 400)

    def test_unimplemented_topic_is_not_acknowledged(self):
        body = b'{}'
        response = self.client.post('/shopify/webhooks', content=body, headers=self.headers(body, 'products/create'))
        self.assertEqual(response.status_code, 404)

    def test_response_adapter_preserves_security_headers(self):
        response = shopify_response(Res(status=403, body="denied", headers={"Content-Security-Policy": "frame-ancestors 'none'"}))
        self.assertEqual(response.status_code, 403)
        self.assertEqual(response.body, b'denied')
        self.assertEqual(response.headers['Content-Security-Policy'], "frame-ancestors 'none'")

    def test_untrusted_shop_cannot_login(self):
        response = self.client.get('/shopify?shop=attacker.example')
        self.assertEqual(response.status_code, 403)
        self.assertNotIn('attacker.example', response.text)

    def token(self, **changes):
        claims = {'aud': 'test-client', 'dest': 'https://ai-generate-image.myshopify.com',
                  'iss': 'https://ai-generate-image.myshopify.com/admin', 'sub': '123',
                  'exp': int(time.time()) + 60, 'nbf': int(time.time()) - 1}
        claims.update(changes)
        return jwt.encode(claims, self.secret, algorithm='HS256')

    def test_verified_shop_gets_partitioned_session_and_tasks(self):
        response = self.client.get('/shopify', headers={'Authorization': 'Bearer ' + self.token()}, follow_redirects=False)
        self.assertEqual(response.status_code, 303)
        self.assertTrue(response.headers['location'].startswith('/admin/tasks?'))
        for flag in ['HttpOnly', 'Secure', 'SameSite=none', 'Partitioned']:
            self.assertIn(flag, response.headers['set-cookie'])
        page = self.client.get('/admin/tasks')
        self.assertEqual(page.status_code, 200)
        self.assertIn('app-bridge.js', page.text)
        self.assertIn('admin.shopify.com', page.headers['content-security-policy'])
        self.assertEqual(self.client.post('/admin/tasks', headers={'Origin': 'https://evil.example'}).status_code, 403)
        self.assertEqual(self.client.post('/admin/tasks', headers={'Origin': 'https://image-api.shopupup.com'}).status_code, 200)

    def test_wrong_shop_audience_expired_and_tampered_tokens_rejected(self):
        for token in [self.token(dest='https://other.myshopify.com'), self.token(aud='other'),
                      self.token(exp=int(time.time()) - 60), self.token() + 'x']:
            self.client.cookies.clear()
            response = self.client.post('/shopify/session', headers={'Authorization': 'Bearer ' + token})
            self.assertIn(response.status_code, (401, 403))
            self.assertNotIn('set-cookie', response.headers)

    def test_direct_access_and_forged_cookie_require_login(self):
        self.assertEqual(self.client.get('/admin/tasks?shop=ai-generate-image.myshopify.com').status_code, 401)
        self.client.cookies.set(COOKIE, 'forged')
        self.assertEqual(self.client.get('/admin/tasks').status_code, 401)

    def test_patch_rejects_external_redirect(self):
        response = self.client.get('/shopify/patch-id-token', params={'shop':'ai-generate-image.myshopify.com', 'shopify-reload':'https://evil.example'})
        self.assertEqual(response.status_code, 400)

    def test_order_originals_require_verified_staff_and_shop(self):
        with patch('app.routers.shopify.get_order_images', new_callable=AsyncMock, return_value={'items': []}) as images:
            self.assertEqual(self.client.get('/shopify/orders/123/images').status_code, 401)
            self.assertEqual(self.client.get('/shopify/orders/123/images', headers={'Authorization': 'Bearer ' + self.token(dest='https://other.myshopify.com')}).status_code, 403)
            images.assert_not_called()
            response = self.client.get('/shopify/orders/123/images', headers={'Authorization': 'Bearer ' + self.token()})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.headers['cache-control'], 'no-store')


if __name__ == '__main__':
    unittest.main()
