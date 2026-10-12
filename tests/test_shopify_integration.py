import base64
import hashlib
import hmac
import unittest

from fastapi import FastAPI
from fastapi.testclient import TestClient
from shopify_app import ShopifyApp
from shopify_app.types import Res

from app.routers.shopify import router, shopify_dependency
from app.services.shopify_http import shopify_response


class ShopifyIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.secret = "test-secret-not-a-real-credential"
        app = FastAPI()
        app.include_router(router)
        app.dependency_overrides[shopify_dependency] = lambda: ShopifyApp(
            client_id="test-client", client_secret=self.secret
        )
        self.client = TestClient(app)

    def tearDown(self):
        self.client.close()

    def headers(self, body, topic="app/uninstalled"):
        signature = base64.b64encode(
            hmac.new(self.secret.encode(), body, hashlib.sha256).digest()
        ).decode()
        return {
            "content-type": "application/json",
            "x-shopify-hmac-sha256": signature,
            "x-shopify-shop-domain": "test-shop.myshopify.com",
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

    def test_unsigned_request_is_rejected(self):
        response = self.client.post('/shopify/webhooks', json={"id": 1})
        # Official SDK treats missing required headers as a malformed request.
        self.assertEqual(response.status_code, 400)

    def test_unimplemented_topic_is_not_acknowledged(self):
        body = b'{}'
        response = self.client.post('/shopify/webhooks', content=body, headers=self.headers(body, 'orders/create'))
        self.assertEqual(response.status_code, 404)

    def test_response_adapter_preserves_security_headers(self):
        response = shopify_response(Res(status=403, body="denied", headers={"Content-Security-Policy": "frame-ancestors 'none'"}))
        self.assertEqual(response.status_code, 403)
        self.assertEqual(response.body, b'denied')
        self.assertEqual(response.headers['Content-Security-Policy'], "frame-ancestors 'none'")

    def test_public_home_is_static(self):
        response = self.client.get('/shopify?shop=attacker.example')
        self.assertEqual(response.status_code, 200)
        self.assertNotIn('attacker.example', response.text)


if __name__ == '__main__':
    unittest.main()
