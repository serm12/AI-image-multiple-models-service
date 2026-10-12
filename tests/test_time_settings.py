import tempfile
import unittest
from datetime import datetime, timezone
from unittest.mock import patch, AsyncMock
from zoneinfo import ZoneInfo
from fastapi import FastAPI
from fastapi.testclient import TestClient
from app.services import time_settings as settings
from app.routers.time_settings import router
from app.services.security import require_admin_login
from app.services.task_query_service import _matches_time_range


class TimeSettingsTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory(); self.addCleanup(folder.cleanup)
        p = patch.object(settings.DirectoryConfig, 'TASKS_DIR', folder.name)
        p.start(); self.addCleanup(p.stop)

    async def test_shop_default_and_custom_survives_refresh(self):
        with patch('app.services.shopify_customers.query', AsyncMock(return_value={'shop': {'ianaTimezone': 'America/New_York'}})):
            await settings.sync_shop_timezone()
            self.assertEqual(settings.read()['mode'], 'shop')
            self.assertEqual(settings.effective_timezone().key, 'America/New_York')
            settings.update({'mode': 'custom', 'custom_timezone': 'Asia/Shanghai'})
            await settings.sync_shop_timezone(force=True)
            self.assertEqual(settings.effective_timezone().key, 'Asia/Shanghai')

    def test_dates_use_selected_timezone_and_dst(self):
        now = datetime(2026, 10, 12, 2, tzinfo=timezone.utc)
        task = {'time': '2026-10-11T15:00:00Z'}
        with patch('app.services.task_query_service.now_china', return_value=now):
            self.assertTrue(_matches_time_range(task, 'today', ZoneInfo('America/New_York')))
            self.assertFalse(_matches_time_range(task, 'today', ZoneInfo('Asia/Shanghai')))
        zone = ZoneInfo('America/New_York')
        self.assertNotEqual(datetime(2026, 1, 1, tzinfo=zone).utcoffset(), datetime(2026, 7, 1, tzinfo=zone).utcoffset())

    def test_save_validates_zone_and_origin(self):
        app = FastAPI(); app.include_router(router)
        app.dependency_overrides[require_admin_login] = lambda: 'admin'
        with TestClient(app, base_url='https://image-api.shopupup.com') as client:
            value = {'mode': 'custom', 'custom_timezone': 'Europe/London'}
            self.assertEqual(client.post('/admin/time-settings', json=value).status_code, 403)
            headers = {'Origin': 'https://image-api.shopupup.com'}
            self.assertEqual(client.post('/admin/time-settings', json={**value, 'custom_timezone': 'invalid'}, headers=headers).status_code, 422)
            self.assertEqual(client.post('/admin/time-settings', json=value, headers=headers).status_code, 200)
            self.assertEqual(settings.effective_timezone().key, 'Europe/London')
