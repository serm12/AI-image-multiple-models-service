import base64
import json
import os
import tempfile
import unittest
from datetime import timedelta
from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image
from starlette.requests import Request

from app.core.config import AppConfig, DirectoryConfig
from app.core.version import APP_RELEASE_DATE, APP_VERSION
from app.utils.time_utils import CHINA_TIMEZONE_NAME, now_china
from app.main import app
from app.routers.admin import (
    US_EASTERN_TIMEZONE,
    US_PACIFIC_TIMEZONE,
    _display_time,
    _traffic_source_icon,
)
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
        from app.services.time_settings import update
        update({"shop_timezone": "Asia/Shanghai"})

    def tearDown(self):
        AppConfig.ADMIN_USER = self.old_user
        AppConfig.ADMIN_PASSWORD = self.old_password
        DirectoryConfig.TASKS_DIR = self.old_tasks_dir
        self.temp_dir.cleanup()

    def test_entered_email_is_not_login_and_subscription_is_frontend_report(self):
        from app.routers.admin import _task_row
        from app.services.task_query_service import list_task_summaries
        task_dir = os.path.join(self.temp_dir.name, "newsletter-task")
        os.makedirs(task_dir)
        with open(os.path.join(task_dir, "params.json"), "w", encoding="utf-8") as file:
            json.dump({"customer_email": "visitor@example.com"}, file)
        task = list_task_summaries()["tasks"][0]
        html = _task_row(task, "/admin/tasks")
        self.assertIn("已填写邮箱", html)
        self.assertNotIn(">已登录<", html)
        self.assertIn("订阅：未记录", html)
        task["subscription_status"] = "verification_required"
        html = _task_row(task, "/admin/tasks")
        self.assertIn("订阅：需要验证", html)
        self.assertIn("前端回报", html)

    def test_traffic_source_icons_include_bing_and_duckduckgo(self):
        self.assertIn("www.bing.com", _traffic_source_icon("referrer:www.bing.com"))
        self.assertIn("duckduckgo.com", _traffic_source_icon("https://duckduckgo.com"))

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
                    "traffic_source": "utm:google",
                    "traffic_referrer": "https://www.google.com",
                    "traffic_landing_path": "/products/custom-portrait",
                    "utm_source": "google",
                    "utm_medium": "cpc",
                    "utm_campaign": "fall-portraits",
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
        unauthenticated = client.get("/admin/api/tasks", follow_redirects=False)
        self.assertEqual(unauthenticated.status_code, 303)
        self.assertTrue(unauthenticated.headers["location"].startswith("/admin/login"))
        token = base64.b64encode(b"admin:secret").decode("ascii")
        response = client.get(
            "/admin/api/tasks", headers={"Authorization": f"Basic {token}"}
        )

        self.assertEqual(response.status_code, 200)
        item = response.json()['tasks'][0]
        self.assertEqual(item['client_ip'], '203.0.113.5')
        self.assertEqual(item['client_country'], 'US')
        self.assertEqual(item['customer_email'], 'buyer@example.com')
        self.assertEqual(item['source_product_title'], 'Custom Portrait')
        self.assertEqual(item['utm_campaign'], 'fall-portraits')
        self.assertEqual(item['generation_duration_seconds'], 12.34)
        self.assertTrue(item['added_to_cart'])
        self.assertTrue(item['checkout_started'])
        self.assertFalse(item['checkout_intent'])
        self.assertEqual(item['prompt'], '<script>alert(1)</script>')
        self.assertEqual(item['display_times']['Asia/Shanghai'], '2026-08-28 18:00:00')
        self.assertEqual(item['display_times']['America/New_York'], '2026-08-28 06:00:00')
        for query in ['client_ip=203.0.113', 'user=buyer%40example.com']:
            filtered = client.get('/admin/api/tasks?' + query, headers={'Authorization': f'Basic {token}'})
            self.assertEqual(filtered.status_code, 200)
            self.assertEqual(filtered.json()['total'], 1)

        thumbnail = client.get(
            "/admin/tasks/task-1/thumbnail/output_reference.png",
            headers={"Authorization": f"Basic {token}"},
        )
        self.assertEqual(thumbnail.status_code, 200)
        self.assertEqual(thumbnail.headers["content-type"], "image/jpeg")
        with Image.open(BytesIO(thumbnail.content)) as preview:
            self.assertLessEqual(preview.width, 160)
            self.assertLessEqual(preview.height, 160)

    def test_login_page_creates_a_remembered_admin_session(self):
        client = TestClient(app, base_url="https://testserver")
        login_page = client.get("/admin/login")
        self.assertEqual(login_page.status_code, 200)
        self.assertIn('id="root"', login_page.text)

        response = client.post(
            "/admin/login",
            data={"username": "admin", "password": "secret", "remember": "true"},
            follow_redirects=False,
        )
        self.assertEqual(response.status_code, 303)
        self.assertIn("ai_image_admin_session=", response.headers["set-cookie"])
        self.assertIn("HttpOnly", response.headers["set-cookie"])
        self.assertIn("Secure", response.headers["set-cookie"])
        self.assertEqual(client.get("/admin/tasks").status_code, 200)

    def test_edit_task_shows_customer_request_and_source_reference(self):
        source_task_id = "20260922_000000_abcdef12"
        source_dir = os.path.join(self.temp_dir.name, source_task_id)
        os.makedirs(source_dir)
        with open(os.path.join(source_dir, "params.json"), "w", encoding="utf-8") as file:
            json.dump({"time": "20260922_000000"}, file)
        source_filename = "output_cropped_original_14155662.png"
        Image.new("RGB", (800, 1200), "purple").save(
            os.path.join(source_dir, source_filename)
        )

        edit_task_id = "20260922_000100_12345678"
        edit_dir = os.path.join(self.temp_dir.name, edit_task_id)
        os.makedirs(edit_dir)
        customer_request = "Make the dress blue & keep the face <recognizable>."
        merged_prompt = (
            "Base portrait prompt.\n\n"
            "Edit the supplied previously generated preview. "
            f"Apply only this customer request: {customer_request}\n"
            "Preserve the subject identity, composition, clothing, lighting, and background."
        )
        with open(os.path.join(edit_dir, "params.json"), "w", encoding="utf-8") as file:
            json.dump(
                {
                    "time": "20260922_000100",
                    "source_task_id": source_task_id,
                    "original_prompt": merged_prompt,
                },
                file,
            )
        Image.new("RGB", (800, 1200), "blue").save(
            os.path.join(edit_dir, "output_cropped_watermark.png")
        )

        client = TestClient(app)
        token = base64.b64encode(b"admin:secret").decode("ascii")
        response = client.get(
            "/admin/api/tasks", headers={"Authorization": f"Basic {token}"}
        )

        self.assertEqual(response.status_code, 200)
        edited = next(task for task in response.json()['tasks'] if task['source_task_id'])
        self.assertEqual(edited['edit_instructions'], 'Make the dress blue & keep the face <recognizable>.')
        reference = next(image for image in edited['images'] if image['reference'])
        self.assertEqual(reference['thumbnail'], f'/admin/tasks/{source_task_id}/thumbnail/{source_filename}')


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
        first_page = client.get("/admin/api/tasks", headers=headers)
        second_page = client.get("/admin/api/tasks?page=2&page_size=25", headers=headers)

        self.assertEqual(first_page.status_code, 200)
        first, second = first_page.json(), second_page.json()
        self.assertEqual(first['total'], 30)
        self.assertEqual(first['total_pages'], 2)
        self.assertEqual(len(first['tasks']), 25)
        self.assertEqual(len(second['tasks']), 5)
        self.assertTrue(first['has_next'])
        self.assertTrue(second['has_previous'])
        self.assertEqual(second['page'], 2)
        self.assertIn('prompt-29', [task['prompt'] for task in first['tasks']])
        self.assertNotIn('prompt-0', [task['prompt'] for task in first['tasks']])
        self.assertIn('prompt-0', [task['prompt'] for task in second['tasks']])
        self.assertNotIn(empty_intermediate_dir, first_page.text)


    def test_admin_tasks_filters_by_task_id_and_provider(self):
        task_specs = (
            ("20260915_120000_recraft", "fal-ai/recraft/upscale/crisp"),
            ("20260915_120100_aura", "fal-ai/aura-sr"),
            ("20260915_120200_gpt", "gpt-image-2_aiapiroute"),
        )
        for task_id, provider in task_specs:
            task_dir = os.path.join(self.temp_dir.name, task_id)
            os.makedirs(task_dir)
            with open(os.path.join(task_dir, "params.json"), "w", encoding="utf-8") as file:
                json.dump(
                    {
                        "time": "20260915_120000",
                        "original_prompt": f"prompt-{task_id}",
                        "api_provider": provider,
                    },
                    file,
                )

        client = TestClient(app)
        token = base64.b64encode(b"admin:secret").decode("ascii")
        headers = {"Authorization": f"Basic {token}"}

        for query in ['task_id=120100', 'provider=aura-sr', 'task_id=120100&provider=fal-ai']:
            response = client.get('/admin/api/tasks?' + query, headers=headers)
            self.assertEqual(response.status_code, 200)
            self.assertEqual([task['task_id'] for task in response.json()['tasks']], ['20260915_120100_aura'])


    def test_admin_can_star_tasks_and_filter_to_starred(self):
        task_ids = ("20260915_120000_first", "20260915_120100_second")
        for task_id in task_ids:
            task_dir = os.path.join(self.temp_dir.name, task_id)
            os.makedirs(task_dir)
            with open(os.path.join(task_dir, "params.json"), "w", encoding="utf-8") as file:
                json.dump({"time": "20260915_120000", "original_prompt": task_id}, file)

        client = TestClient(app)
        token = base64.b64encode(b"admin:secret").decode("ascii")
        headers = {"Authorization": f"Basic {token}"}
        response = client.post(
            f"/admin/tasks/{task_ids[0]}/star",
            data={"starred": "false", "next": "/admin/tasks?starred=true"},
            headers=headers,
            follow_redirects=False,
        )

        self.assertEqual(response.status_code, 303)
        self.assertEqual(response.headers["location"], "/admin/tasks?starred=true")
        with open(
            os.path.join(self.temp_dir.name, task_ids[0], "admin_metadata.json"),
            encoding="utf-8",
        ) as file:
            self.assertTrue(json.load(file)["starred"])

        starred = client.get('/admin/api/tasks?starred=true', headers=headers)
        self.assertEqual(starred.status_code, 200)
        self.assertEqual([task['task_id'] for task in starred.json()['tasks']], [task_ids[0]])
        self.assertTrue(starred.json()['tasks'][0]['starred'])


    def test_admin_tasks_filters_by_multiple_conversion_statuses(self):
        task_specs = (
            ("20260916_120000_cart", "prompt-cart", "added_to_cart"),
            ("20260916_120100_paid", "prompt-paid", "checkout_completed"),
            ("20260916_120200_none", "prompt-none", None),
        )
        for task_id, prompt, event_type in task_specs:
            task_dir = os.path.join(self.temp_dir.name, task_id)
            os.makedirs(task_dir)
            with open(
                os.path.join(task_dir, "params.json"), "w", encoding="utf-8"
            ) as file:
                json.dump(
                    {"time": "20260916_120000", "original_prompt": prompt}, file
                )
            if event_type:
                with open(
                    os.path.join(task_dir, "storefront_events.json"),
                    "w",
                    encoding="utf-8",
                ) as file:
                    json.dump([{"event_id": task_id, "event_type": event_type}], file)

        client = TestClient(app)
        token = base64.b64encode(b"admin:secret").decode("ascii")
        response = client.get(
            "/admin/api/tasks?conversion_status=added_to_cart"
            "&conversion_status=checkout_completed",
            headers={"Authorization": f"Basic {token}"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual({task['prompt'] for task in response.json()['tasks']}, {'prompt-cart', 'prompt-paid'})


    def test_admin_tasks_filters_by_time_range(self):
        now = now_china()
        task_specs = (
            ("20260928_120000_today", "prompt-today", now),
            ("20260927_120000_yesterday", "prompt-yesterday", now - timedelta(days=1)),
            ("20260801_120000_old", "prompt-old", now - timedelta(days=40)),
        )
        for task_id, prompt, task_time in task_specs:
            task_dir = os.path.join(self.temp_dir.name, task_id)
            os.makedirs(task_dir)
            with open(
                os.path.join(task_dir, "params.json"), "w", encoding="utf-8"
            ) as file:
                json.dump(
                    {
                        "time": task_time.isoformat(),
                        "time_zone": CHINA_TIMEZONE_NAME,
                        "original_prompt": prompt,
                    },
                    file,
                )

        client = TestClient(app)
        token = base64.b64encode(b"admin:secret").decode("ascii")
        response = client.get(
            "/admin/api/tasks?time_range=today",
            headers={"Authorization": f"Basic {token}"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual([task['prompt'] for task in response.json()['tasks']], ['prompt-today'])


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
        first_page = client.get("/admin/api/tasks?page=1&page_size=10", headers=headers)
        second_page = client.get("/admin/api/tasks?page=2&page_size=10", headers=headers)

        self.assertEqual(first_page.status_code, 200)
        first = [task for task in first_page.json()['tasks'] if task['client_ip'] == '203.0.113.20']
        second = second_page.json()['tasks']
        self.assertEqual([task['ip_task_sequence'] for task in first], list(range(11, 2, -1)))
        self.assertEqual([task['ip_task_sequence'] for task in second], [2, 1])
        self.assertTrue(all(task['ip_task_total'] == 11 for task in first + second))


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

    def test_china_time_is_displayed_in_us_pacific_time(self):
        self.assertEqual(
            _display_time(
                "20260904_042745",
                "Asia/Shanghai",
                US_PACIFIC_TIMEZONE,
            ),
            "2026-09-03 13:27:45",
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
