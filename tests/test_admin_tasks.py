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
from app.routers.admin import US_EASTERN_TIMEZONE, _display_time
from app.services.security import (
    get_request_client_ip,
    get_request_country,
    get_request_source_page,
)


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
                    "source_page_url": "https://shop.example/products/custom-portrait",
                    "source_page_title": "Custom Portrait – Example Shop",
                    "source_page_type": "product",
                    "source_product_id": "123456789",
                    "source_product_handle": "custom-portrait",
                    "source_product_title": "Custom Portrait",
                    "customer_id": "123456",
                    "customer_email": "buyer@example.com",
                    "client_ip": "203.0.113.5",
                    "client_country": "US",
                    "generation_duration_seconds": 12.34,
                    "api_provider": "test-provider",
                },
                file,
            )
        with open(
            os.path.join(task_dir, "storefront_events.json"), "w", encoding="utf-8"
        ) as file:
            json.dump(
                [
                    {
                        "event_id": "add-1",
                        "event_type": "added_to_cart",
                        "occurred_at": "2026-09-18T00:10:00Z",
                    },
                    {
                        "event_id": "checkout-1",
                        "event_type": "checkout_started",
                        "occurred_at": "2026-09-18T00:12:00Z",
                    },
                ],
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
        self.assertIn("来源页面", response.text)
        self.assertIn("Custom Portrait", response.text)
        self.assertIn("shop.example/products/custom-portrait", response.text)
        self.assertIn("12.3 秒", response.text)
        self.assertIn("buyer@example.com", response.text)
        self.assertIn("Customer ID: 123456", response.text)
        self.assertIn("✓</span> 加购", response.text)
        self.assertIn("○</span> 结账意图", response.text)
        self.assertIn("✓</span> 开始结账", response.text)
        self.assertIn("加购：已触发 · 2026-09-18T00:10:00Z", response.text)
        self.assertIn("时间（美国东部）", response.text)
        self.assertIn('id="current-beijing-time"', response.text)
        self.assertIn('id="current-us-eastern-time"', response.text)
        self.assertIn("Asia/Shanghai", response.text)
        self.assertIn("America/New_York", response.text)
        self.assertIn('aria-label="任务分页"', response.text)
        self.assertIn('id="page-size-select"', response.text)
        self.assertIn('第 1 / 1 页', response.text)
        self.assertIn(f"v{APP_VERSION} · {APP_RELEASE_DATE}", response.text)
        self.assertIn(
            "/admin/tasks/task-1/thumbnail/output_reference.png", response.text
        )
        self.assertIn('data-image-strip', response.text)
        self.assertIn('class="image-scroll image-scroll-prev"', response.text)
        self.assertIn('class="image-scroll image-scroll-next"', response.text)
        self.assertIn('class="images" tabindex="0"', response.text)
        self.assertIn('class="image-item glightbox"', response.text)
        self.assertIn('data-gallery="task-task-1"', response.text)
        self.assertIn('glightbox@3.3.1/dist/css/glightbox.min.css', response.text)
        self.assertIn('glightbox@3.3.1/dist/js/glightbox.min.js', response.text)
        self.assertIn("selector:'.glightbox'", response.text)
        self.assertIn("touchNavigation:true", response.text)
        self.assertIn("zoomable:true", response.text)
        self.assertIn("updateLightboxCounter", response.text)
        self.assertNotIn('id="image-viewer"', response.text)
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

    def test_admin_tasks_uses_server_side_pagination(self):
        for index in range(30):
            task_id = f"20260913_{index:06d}_task"
            task_dir = os.path.join(self.temp_dir.name, task_id)
            os.makedirs(task_dir)
            with open(os.path.join(task_dir, "params.json"), "w", encoding="utf-8") as file:
                json.dump(
                    {
                        "time": f"20260913_{index:06d}",
                        "original_prompt": f"prompt-{index}",
                    },
                    file,
                )

        empty_intermediate_dir = (
            "20260913_999999_task_aiapiroute_gpt_image_20260913_160000_na"
        )
        os.makedirs(os.path.join(self.temp_dir.name, empty_intermediate_dir))

        client = TestClient(app)
        token = base64.b64encode(b"admin:secret").decode("ascii")
        headers = {"Authorization": f"Basic {token}"}
        first_page = client.get("/admin/tasks", headers=headers)
        second_page = client.get("/admin/tasks?page=2&page_size=25", headers=headers)

        self.assertEqual(first_page.status_code, 200)
        self.assertEqual(first_page.text.count("<tr>"), 26)
        self.assertIn("prompt-29", first_page.text)
        self.assertNotIn("prompt-0</div>", first_page.text)
        self.assertIn("第 1 / 2 页", first_page.text)
        self.assertIn("共 30 条任务", first_page.text)
        self.assertIn("/admin/tasks?page=2&amp;page_size=25", first_page.text)
        self.assertNotIn(empty_intermediate_dir, first_page.text)

        self.assertEqual(second_page.status_code, 200)
        self.assertEqual(second_page.text.count("<tr>"), 6)
        self.assertIn("prompt-0", second_page.text)
        self.assertNotIn("prompt-29", second_page.text)
        self.assertIn("第 2 / 2 页", second_page.text)

    def test_same_ip_has_chronological_sequence_across_pages(self):
        for index in range(12):
            task_id = f"20260914_{index:06d}_task"
            task_dir = os.path.join(self.temp_dir.name, task_id)
            os.makedirs(task_dir)
            with open(os.path.join(task_dir, "params.json"), "w", encoding="utf-8") as file:
                json.dump(
                    {
                        "time": f"20260914_{index:06d}",
                        "client_ip": (
                            "203.0.113.20"
                            if index != 5
                            else "2603:6011:e600:46:4999:2:fbba:a378"
                        ),
                    },
                    file,
                )

        client = TestClient(app)
        token = base64.b64encode(b"admin:secret").decode("ascii")
        headers = {"Authorization": f"Basic {token}"}
        first_page = client.get("/admin/tasks?page=1&page_size=10", headers=headers)
        second_page = client.get("/admin/tasks?page=2&page_size=10", headers=headers)

        self.assertEqual(first_page.status_code, 200)
        self.assertIn(
            '<span class="client-ip__address">203.0.113.20</span> '
            '<span class="ip-task-sequence"',
            first_page.text,
        )
        self.assertIn(
            '<span class="client-ip__address">2603:6011:e600:46:4999:2:fbba:a378</span>',
            first_page.text,
        )
        self.assertIn(".client-ip__address{overflow-wrap:anywhere;word-break:break-all}", first_page.text)
        self.assertIn("#11</span>", first_page.text)
        self.assertIn("#6</span>", first_page.text)
        self.assertIn("#3</span>", first_page.text)
        self.assertIn("#2</span>", second_page.text)
        self.assertIn("#1</span>", second_page.text)
        self.assertEqual(first_page.text.count("共 11 次</span>"), 9)
        self.assertEqual(second_page.text.count("共 11 次</span>"), 2)
        self.assertIn('class="ip-task-total"', first_page.text)
        self.assertIn("该 IP 发起的第 11 个任务", first_page.text)
        self.assertIn("该 IP 累计发起 11 个任务", first_page.text)

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

    def test_china_time_is_displayed_in_us_eastern_time(self):
        self.assertEqual(
            _display_time(
                "20260904_042745",
                "Asia/Shanghai",
                US_EASTERN_TIMEZONE,
            ),
            "2026-09-03 16:27:45",
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

    def test_explicit_source_page_is_preferred_and_fragment_is_removed(self):
        request = Request(
            {
                "type": "http",
                "method": "POST",
                "path": "/generate-async/",
                "headers": [(b"referer", b"https://fallback.example/")],
                "client": ("198.51.100.5", 1234),
                "scheme": "https",
                "server": ("image-api.example", 443),
            }
        )

        self.assertEqual(
            get_request_source_page(
                request,
                "https://shop.example/products/portrait?variant=1#reviews",
            ),
            "https://shop.example/products/portrait?variant=1",
        )

    def test_source_page_rejects_unsafe_scheme_and_falls_back_to_referer(self):
        request = Request(
            {
                "type": "http",
                "method": "POST",
                "path": "/generate-async/",
                "headers": [(b"referer", b"https://shop.example/collections/all")],
                "client": ("198.51.100.5", 1234),
                "scheme": "https",
                "server": ("image-api.example", 443),
            }
        )

        self.assertEqual(
            get_request_source_page(request, "javascript:alert(1)"),
            "https://shop.example/collections/all",
        )


if __name__ == "__main__":
    unittest.main()
