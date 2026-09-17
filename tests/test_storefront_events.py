import json
import os
import tempfile
import unittest

from fastapi.testclient import TestClient

from app.core.config import DirectoryConfig
from app.main import app
from app.services.storefront_events import hash_tracking_token


class StorefrontEventsTests(unittest.TestCase):
    def setUp(self):
        self.old_tasks_dir = DirectoryConfig.TASKS_DIR
        self.temp_dir = tempfile.TemporaryDirectory()
        DirectoryConfig.TASKS_DIR = self.temp_dir.name
        self.task_id = "20260917_120000_test"
        self.task_dir = os.path.join(self.temp_dir.name, self.task_id)
        os.makedirs(self.task_dir)
        self.token = "tracking-token-with-enough-characters"
        with open(os.path.join(self.task_dir, "params.json"), "w", encoding="utf-8") as file:
            json.dump(
                {
                    "task_id": self.task_id,
                    "storefront_tracking_token_hash": hash_tracking_token(self.token),
                },
                file,
            )

    def tearDown(self):
        DirectoryConfig.TASKS_DIR = self.old_tasks_dir
        self.temp_dir.cleanup()

    def payload(self):
        return {
            "task_id": self.task_id,
            "tracking_token": self.token,
            "event_id": "cart-add-1",
            "event_type": "added_to_cart",
            "occurred_at": "2026-09-17T12:00:00Z",
            "source": "test",
            "customer_id": "12345",
            "customer_email": "buyer@example.com",
            "customer_logged_in": True,
            "visitor_id": "visitor-1",
            "cart_token": "secret-cart-token",
            "variant_id": "987",
        }

    def test_records_idempotent_event_without_storing_raw_tokens(self):
        client = TestClient(app)
        first = client.post("/storefront-events", json=self.payload())
        second = client.post("/storefront-events", json=self.payload())

        self.assertEqual(first.status_code, 200)
        self.assertTrue(first.json()["created"])
        self.assertFalse(second.json()["created"])
        with open(
            os.path.join(self.task_dir, "storefront_events.json"),
            "r",
            encoding="utf-8",
        ) as file:
            events = json.load(file)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["customer_id"], "12345")
        self.assertEqual(events[0]["customer_email"], "buyer@example.com")
        self.assertNotEqual(events[0]["cart_token_hash"], "secret-cart-token")
        self.assertNotIn("tracking_token", events[0])

    def test_rejects_invalid_tracking_token_and_event_type(self):
        client = TestClient(app)
        invalid_token = self.payload()
        invalid_token["tracking_token"] = "wrong-token-with-enough-characters"
        self.assertEqual(client.post("/storefront-events", json=invalid_token).status_code, 403)

        invalid_type = self.payload()
        invalid_type["event_type"] = "made_up"
        self.assertEqual(client.post("/storefront-events", json=invalid_type).status_code, 422)


if __name__ == "__main__":
    unittest.main()
