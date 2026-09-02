import base64
import json
import os
import tempfile
import unittest
from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image
from starlette.requests import Request

from app.core.config import AppConfig, DirectoryConfig
from app.core.version import APP_RELEASE_DATE, APP_VERSION
from app.main import app
from app.routers.admin import _display_time
from app.services.security import get_request_client_ip, get_request_country


class AdminTasksTests(unittest.TestCase):
    def setUp(self):
        self.old_user = AppConfig.ADMIN_USER
        self.old_password = AppConfig.ADMIN_PASSWORD
        self.old_tasks_dir = DirectoryConfig.TASKS_DIR
        self.temp_dir = tempfile.TemporaryDirectory()
        AppConfig.ADMIN_USER = "admin"
        AppConfig.ADMIN_PASSWORD = "secret"
        DirectoryConfig.TASKS_DIR = self.temp_dir.name

    def tearDown(self):
        AppConfig.ADMIN_USER = self.old_user
        AppConfig.ADMIN_PASSWORD = self.old_password
        DirectoryConfig.TASKS_DIR = self.old_tasks_dir
        self.temp_dir.cleanup()

    def test_admin_page_requires_login_and_escapes_task_content(self):
        task_dir = os.path.join(self.temp_dir.name, "task-1")
        os.makedirs(task_dir)
        with open(os.path.join(task_dir, "params.json"), "w", encoding="utf-8") as file:
            json.dump(
                {
                    "time": "2026-08-28T10:00:00",
                    "original_prompt": "<script>alert(1)</script>",
                    "request_url": "https://image-api.example/generate-async/",
                    "client_ip": "203.0.113.5",
                    "client_country": "US",
                    "generation_duration_seconds": 12.34,
                    "api_provider": "test-provider",
                },
                file,
            )
        Image.new("RGB", (1200, 800), "red").save(
            os.path.join(task_dir, "output_reference.png")
        )

        client = TestClient(app)
        self.assertEqual(client.get("/admin/tasks").status_code, 401)
        token = base64.b64encode(b"admin:secret").decode("ascii")
        response = client.get(
            "/admin/tasks", headers={"Authorization": f"Basic {token}"}
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn("203.0.113.5", response.text)
        self.assertIn("US", response.text)
        self.assertIn("12.3 秒", response.text)
        self.assertIn(f"v{APP_VERSION} · {APP_RELEASE_DATE}", response.text)
        self.assertIn(
            "/admin/tasks/task-1/thumbnail/output_reference.png", response.text
        )
        self.assertIn("&lt;script&gt;alert(1)&lt;/script&gt;", response.text)
        self.assertNotIn("<script>alert(1)</script>", response.text)

        thumbnail = client.get(
            "/admin/tasks/task-1/thumbnail/output_reference.png",
            headers={"Authorization": f"Basic {token}"},
        )
        self.assertEqual(thumbnail.status_code, 200)
        self.assertEqual(thumbnail.headers["content-type"], "image/jpeg")
        with Image.open(BytesIO(thumbnail.content)) as preview:
            self.assertLessEqual(preview.width, 160)
            self.assertLessEqual(preview.height, 160)

    def test_forwarded_ip_is_only_trusted_from_local_proxy(self):
        trusted_request = Request(
            {
                "type": "http",
                "method": "POST",
                "path": "/generate-async/",
                "headers": [(b"cf-connecting-ip", b"203.0.113.7")],
                "client": ("127.0.0.1", 1234),
                "scheme": "https",
                "server": ("image-api.example", 443),
            }
        )
        public_request = Request(
            {
                "type": "http",
                "method": "POST",
                "path": "/generate-async/",
                "headers": [(b"x-forwarded-for", b"203.0.113.8")],
                "client": ("198.51.100.4", 1234),
                "scheme": "http",
                "server": ("server", 8002),
            }
        )

        self.assertEqual(get_request_client_ip(trusted_request), "203.0.113.7")
        self.assertEqual(get_request_client_ip(public_request), "198.51.100.4")

    def test_legacy_utc_time_is_displayed_in_beijing_time(self):
        self.assertEqual(
            _display_time("20260902_205212"),
            "2026-09-03 04:52:12",
        )
        self.assertEqual(
            _display_time("20260903_045212", "Asia/Shanghai"),
            "2026-09-03 04:52:12",
        )

    def test_cf_connecting_ip_is_trusted_from_cloudflare_proxy(self):
        request = Request(
            {
                "type": "http",
                "method": "POST",
                "path": "/generate-async/",
                "headers": [
                    (b"cf-connecting-ip", b"203.0.113.9"),
                    (b"cf-ipcountry", b"US"),
                    (b"x-forwarded-for", b"203.0.113.9, 172.71.152.93"),
                ],
                "client": ("172.71.152.93", 1234),
                "scheme": "https",
                "server": ("image-api.example", 443),
            }
        )

        self.assertEqual(get_request_client_ip(request), "203.0.113.9")
        self.assertEqual(get_request_country(request), "US")

    def test_cf_header_is_ignored_from_untrusted_public_client(self):
        request = Request(
            {
                "type": "http",
                "method": "POST",
                "path": "/generate-async/",
                "headers": [(b"cf-connecting-ip", b"203.0.113.10")],
                "client": ("198.51.100.5", 1234),
                "scheme": "http",
                "server": ("server", 8002),
            }
        )

        self.assertEqual(get_request_client_ip(request), "198.51.100.5")
        self.assertEqual(get_request_country(request), "")


if __name__ == "__main__":
    unittest.main()
