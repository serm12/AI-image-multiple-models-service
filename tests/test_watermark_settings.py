import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from PIL import Image, ImageChops
from fastapi import FastAPI
from fastapi.testclient import TestClient
from app.services import watermark_settings as settings
from app.routers.watermark_settings import router, render_preview, settings_page
from app.services.security import require_admin_login


class WatermarkSettingsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        patcher = patch.object(settings.DirectoryConfig, 'TASKS_DIR', self.temp.name)
        patcher.start(); self.addCleanup(patcher.stop)
        app = FastAPI(); app.include_router(router)
        app.dependency_overrides[require_admin_login] = lambda: 'test-admin'
        self.client = TestClient(app, base_url='https://image-api.shopupup.com')
        self.addCleanup(self.client.close)

    def png(self):
        buffer = io.BytesIO()
        Image.new('RGBA', (100, 30), (255, 0, 0, 160)).save(buffer, 'PNG')
        return buffer.getvalue()

    def post(self, values, path='settings', files=None):
        return self.client.post('/admin/watermark/' + path,
            headers={'Origin': 'https://image-api.shopupup.com'},
            data={'settings': json.dumps(values)}, files=files)

    def test_save_reload_upload_and_runtime_snapshot(self):
        values = settings.default_settings().model_dump(exclude={'watermark_asset', 'brand_asset'})
        values.update(layout='grid', corner_position='top_right', opacity=0.2)
        response = self.post(values, files={'watermark': ('logo.png', self.png(), 'image/png')})
        self.assertEqual(response.status_code, 200, response.text)
        saved = settings.load_settings()
        self.assertTrue(settings.asset_path(saved, 'watermark').exists())
        before = settings.runtime_config()
        saved.opacity = 0.8; settings.save_settings(saved)
        self.assertEqual(before.TILED_LOGO_OPACITY, 0.2)
        self.assertEqual(settings.runtime_config().TILED_LOGO_OPACITY, 0.8)
        self.assertEqual(before.CORNER_POSITION, 'top_right')
        self.assertEqual(before.TILED_ANGLE, 0)

    def test_preview_does_not_save_and_cleans_upload(self):
        values = settings.default_settings().model_dump(exclude={'watermark_asset', 'brand_asset'})
        response = self.post(values, 'preview', {'watermark': ('logo.png', self.png(), 'image/png')})
        self.assertEqual(response.status_code, 200, response.text[:100])
        self.assertEqual(response.headers['content-type'], 'image/png')
        self.assertFalse(settings.settings_path().exists())
        self.assertFalse(list(settings.assets_dir().glob('*.png')))

    def test_rejects_external_origin_invalid_range_and_path_injection(self):
        self.assertEqual(self.client.post('/admin/watermark/settings', headers={'Origin':'https://evil.example'}).status_code, 403)
        for values in ({'opacity': 2}, {'layout':'invalid'}, {'watermark_asset':'../../secret'}, {'opacity': float('nan')}):
            self.assertEqual(self.post(values).status_code, 400)
        self.assertEqual(self.post({}, files={'watermark':('x.svg', b'<svg/>', 'image/svg+xml')}).status_code, 400)
        self.assertFalse(settings.settings_path().exists())

    def test_layouts_render_and_change_pixels_without_changing_source(self):
        outputs = []
        for layout in ['grid','uniform_diagonal','random','center']:
            value = settings.default_settings().model_copy(update={'layout':layout,'corner_position':'top_right','corner_text':'PREVIEW ONLY'})
            output = Image.open(io.BytesIO(render_preview(value))).convert('RGB')
            self.assertEqual(output.size, (540,660))
            outputs.append(output)
        self.assertIsNotNone(ImageChops.difference(outputs[0],outputs[1]).getbbox())

    def test_settings_page_does_not_expose_environment_fields(self):
        page = settings_page().body.decode()
        self.assertIn('水印设置', page)
        self.assertNotIn('当前环境配置', page)
        self.assertNotIn('os.environ', page)
