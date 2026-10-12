import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from app.services import i18n
from app.services.security import require_admin_login
from app.routers.language_settings import router, LanguageMiddleware
from app.routers import admin
from app.routers.watermark_settings import settings_page


class I18nTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        patcher = patch.object(i18n.DirectoryConfig, 'TASKS_DIR', folder.name)
        patcher.start()
        self.addCleanup(patcher.stop)
        token = i18n.current_locale.set(None)
        self.addCleanup(i18n.current_locale.reset, token)

    def test_default_persistence_and_invalid_preferences(self):
        self.assertEqual(i18n.selected_locale(), 'zh-CN')
        i18n.save_locale('en')
        self.assertEqual(i18n.selected_locale(), 'en')
        with self.assertRaises(ValueError):
            i18n.save_locale('../secret')
        self.assertEqual(i18n.selected_locale(), 'en')
        i18n.preference_path().write_text('{', 'utf-8')
        self.assertEqual(i18n.selected_locale(), 'zh-CN')

    def test_catalog_completeness_and_escaping(self):
        self.assertEqual(set(i18n.catalog('en')), set(i18n.catalog('zh-CN')))
        for parts in i18n.templates().values():
            for part in parts:
                if isinstance(part, dict):
                    self.assertIn(part['key'], i18n.catalog('en'))
        with patch.object(i18n, 't', return_value="</script>'\"&"):
            self.assertEqual(i18n.render_page('{{i18n:test:html}}'), '&lt;/script&gt;&#x27;&quot;&amp;')
            self.assertNotIn('</script>', i18n.render_page('{{i18n:test:js}}'))
            self.assertNotIn("'", i18n.render_page('{{i18n:test:js}}'))

    def test_request_language_changes_and_business_data_preserved(self):
        app = FastAPI()
        app.include_router(router)
        app.include_router(admin.router)
        app.add_middleware(LanguageMiddleware)
        app.dependency_overrides[require_admin_login] = lambda: 'admin'
        data = dict(tasks=[], total=0, page=1, page_size=25, total_pages=1, has_previous=False, has_next=False)
        with TestClient(app, base_url='https://image-api.shopupup.com') as client, patch.object(admin, 'list_task_summaries', return_value=data):
            self.assertEqual(client.post('/admin/language-settings', json={'locale': 'en'}).status_code, 403)
            headers = {'Origin': 'https://image-api.shopupup.com'}
            self.assertEqual(client.post('/admin/language-settings', json={'locale': 'unknown'}, headers=headers).status_code, 422)
            for locale, heading, today in [('en', 'Language settings', 'Today'), ('zh-CN', '语言设置', '今天')]:
                self.assertEqual(client.post('/admin/language-settings', json={'locale': locale}, headers=headers).status_code, 200)
                settings = client.get('/admin/config')
                self.assertIn(heading, settings.text)
                self.assertNotIn('{{i18n:', settings.text)
                self.assertEqual(settings.headers['Content-Language'], locale)
                tasks = client.get('/admin/tasks', params={'user': '任务记录@example.com'})
                self.assertEqual(tasks.status_code, 200)
                self.assertIn(today, tasks.text)
                self.assertIn('任务记录@example.com', tasks.text)
        self.assertIsNone(i18n.current_locale.get())
