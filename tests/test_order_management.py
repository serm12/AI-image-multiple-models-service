import io
import tempfile
import unittest
from unittest.mock import AsyncMock, patch
from PIL import Image
from fastapi import HTTPException
from fastapi.testclient import TestClient
from app.core.config import AppConfig, DirectoryConfig
from app.main import app
from app.services import order_management as service
from app.routers.order_management import PushInput


class ProductionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        patcher = patch.object(DirectoryConfig, 'TASKS_DIR', self.folder.name)
        patcher.start(); self.addCleanup(patcher.stop)
        self.order = {'id': 'gid://shopify/Order/42', 'name': '#42', 'cancelledAt': None,
            'shippingAddress': {'name': 'Test', 'address1': '1 Street', 'city': 'City', 'zip': '12345', 'countryCodeV2': 'US'},
            'items': [{'id': 'line1', 'requiresShipping': True, 'unfulfilledQuantity': 2,
                       'images': [{'task_id': '20261018_120000_123456ab', 'url': 'https://example.com/art.jpg'}]}]}
        self.input = PushInput(items=[{'line_id': 'line1', 'task_id': '20261018_120000_123456ab',
                                     'variant_id': 5, 'upscale': False, 'bleed': False}])

    async def test_draft_update_keeps_quantity_and_never_confirms(self):
        result = {'id': 8, 'status': 'draft', 'costs': {'total': '20'}, 'items': [{'files': [{'id': 3, 'status': 'ok'}]}]}
        remote = AsyncMock(side_effect=[result, {'product': {'id': 3}}, result, result])
        with patch.object(service, 'detail', AsyncMock(return_value=self.order)), patch.object(service, 'pf', remote), patch.object(service, 'prepare') as prepare:
            value = await service.push('42', self.input)
        self.assertEqual(value['id'], 8)
        prepare.assert_not_called()
        call = remote.call_args_list[2]
        self.assertEqual(call.args[:2], ('PUT', '/orders/8?confirm=0'))
        self.assertEqual(call.args[2]['items'][0]['quantity'], 2)
        self.assertEqual(service.history('42')['file_statuses'][0]['status'], 'ok')

    async def test_existing_production_order_is_never_overwritten(self):
        remote = AsyncMock(return_value={'id': 8, 'status': 'fulfilled'})
        with patch.object(service, 'detail', AsyncMock(return_value=self.order)), patch.object(service, 'pf', remote):
            with self.assertRaises(HTTPException) as error:
                await service.push('42', self.input)
        self.assertEqual(error.exception.status_code, 409)
        self.assertEqual(remote.await_count, 1)

    async def test_foreign_artwork_rejected_before_processing(self):
        self.input.items[0].task_id = '20261018_120000_876543ab'
        with patch.object(service, 'detail', AsyncMock(return_value=self.order)), patch.object(service, 'pf', AsyncMock(return_value=None)), patch.object(service, 'prepare') as prepare:
            with self.assertRaises(HTTPException):
                await service.push('42', self.input)
            prepare.assert_not_called()

    async def test_unsupported_bleed_rejected(self):
        self.input.items[0].bleed = True
        self.input.items[0].variant_id = 123
        with patch.object(service, 'detail', AsyncMock(return_value=self.order)), patch.object(service, 'pf', AsyncMock(return_value=None)):
            with self.assertRaises(HTTPException):
                await service.push('42', self.input)

    def test_job_claim_is_atomic_and_retriable_after_failure(self):
        service.start_job('42')
        with self.assertRaises(HTTPException): service.start_job('42')
        service.finish_job('42', 'failed', 'test')
        service.start_job('42')
        self.assertEqual(service.job('42')['state'], 'running')

    def test_bleed_preserves_center_art_and_dimensions(self):
        source = io.BytesIO()
        Image.new('RGB', (30, 40), 'red').save(source, format='PNG')
        data, dimensions = service.render_art(source.getvalue(), (50, 60, 10), True)
        self.assertEqual(dimensions, (50, 60))
        with Image.open(io.BytesIO(data)) as result:
            self.assertGreater(result.getpixel((25, 30))[0], 240)
            self.assertLess(result.getpixel((25, 30))[1], 20)
            self.assertTrue(all(c > 240 for c in result.getpixel((1, 1))))

    def test_upscale_normalizes_input_and_hosts_final_file(self):
        source = io.BytesIO()
        Image.new('RGB', (2048, 1024), 'blue').save(source, format='PNG')
        with patch.object(service, 'APIConfig') as config, patch('fal_client.SyncClient') as client, patch.object(service, 'download_art', return_value=source.getvalue()):
            config.FAL_API_KEY = 'test-only'
            fal = client.return_value
            fal.upload_file.side_effect = ['https://example.com/normalized.png', 'https://example.com/final.jpg']
            fal.submit.return_value.request_id = 'request-123'
            fal.submit.return_value.get.return_value = {'image': {'url': 'https://example.com/upscaled.png'}}
            result = service.prepare('https://example.com/source.png', 123, True, False)
            self.assertEqual(result['normalized_input_dimensions'], (1024, 512))
            self.assertEqual(result['request_id'], 'request-123')
            self.assertEqual(result['file_url'], 'https://example.com/final.jpg')
            self.assertEqual(fal.submit.call_args.kwargs['arguments'], {
                'image_url': 'https://example.com/normalized.png', 'upscale_factor': 4,
                'checkpoint': 'v2', 'overlapping_tiles': True})

    def test_credentials_auth_origin_and_no_secret_echo(self):
        with patch.object(AppConfig, 'ADMIN_USER', 'admin'), patch.object(AppConfig, 'ADMIN_PASSWORD', 'test-secret'), patch.dict('os.environ', {'ADMIN_SESSION_SECRET': 'test-encryption-key', 'SHOPIFY_API_SECRET': 'test-encryption-key'}):
            with TestClient(app, base_url='https://testserver') as client:
                self.assertEqual(client.get('/admin/api/production/settings', follow_redirects=False).status_code, 303)
                client.auth = ('admin', 'test-secret')
                self.assertEqual(client.post('/admin/api/production/settings', json={'token': 'secret-value'}).status_code, 403)
                result = client.post('/admin/api/production/settings', headers={'Origin': 'https://testserver'}, json={'token': 'secret-value', 'store_id': '123'})
                self.assertEqual(result.status_code, 200, result.text)
                self.assertNotIn('secret-value', result.text)
                self.assertEqual(result.json(), {'configured': True, 'store_id': '123'})
                self.assertEqual(result.headers['cache-control'], 'no-store')
                bad = client.post('/admin/api/production/settings', headers={'Origin': 'https://testserver'}, json={'token': 'secret-value', 'store_id': 'invalid'})
                self.assertEqual(bad.status_code, 400)
                self.assertNotIn('secret-value', bad.text)

if __name__ == '__main__':
    unittest.main()
