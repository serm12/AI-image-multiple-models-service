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

    def test_direct_timezone_and_comparisons(self):
        settings.update({'shop_timezone': 'Asia/Shanghai'})
        app = FastAPI(); app.include_router(router)
        app.dependency_overrides[require_admin_login] = lambda: 'admin'
        with TestClient(app, base_url='https://image-api.shopupup.com') as client:
            headers = {'Origin': 'https://image-api.shopupup.com'}
            value = {'timezone': 'Asia/Shanghai', 'comparison_timezones': ['Europe/London', 'Asia/Tokyo', 'Europe/London']}
            response = client.post('/admin/time-settings', json=value, headers=headers)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()['mode'], 'shop')
            self.assertEqual(settings.display_timezones(), ['Asia/Shanghai', 'Europe/London', 'Asia/Tokyo'])
            self.assertEqual(client.post('/admin/time-settings', json={**value, 'comparison_timezones': ['bad']}, headers=headers).status_code, 422)
            self.assertEqual(client.post('/admin/time-settings', json={**value, 'comparison_timezones': ['UTC'] * 6}, headers=headers).status_code, 422)
            client.post('/admin/time-settings', json={'timezone': 'America/New_York', 'comparison_timezones': []}, headers=headers)
            self.assertEqual(settings.display_timezones(), ['America/New_York'])

    def test_legacy_and_utc_timestamps_sort_by_instant(self):
        import json
        from pathlib import Path
        from app.services.task_query_service import list_task_summaries
        records = [('20261012_180000_old', '20261012_180000', 'Asia/Shanghai'),
                   ('20261012_110000_new', '2026-10-12T11:00:00+00:00', 'UTC')]
        for task_id, timestamp, zone in records:
            folder = Path(settings.DirectoryConfig.TASKS_DIR) / task_id; folder.mkdir()
            (folder / 'params.json').write_text(json.dumps({'task_id': task_id, 'time': timestamp, 'time_zone': zone}))
        self.assertEqual(list_task_summaries()['tasks'][0]['task_id'], records[1][0])
