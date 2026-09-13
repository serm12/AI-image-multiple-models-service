import json
import os
import tempfile
import unittest

from fastapi.testclient import TestClient

from app.main import app
from app.services.async_task_manager import task_manager
from app.services.r2_storage import (
    R2StorageConfig,
    read_r2_mapping,
    replace_with_cdn_urls,
    upload_public_task_images,
)


class FakeS3Client:
    def __init__(self):
        self.uploads = []

    def upload_file(self, filename, bucket, key, ExtraArgs):
        self.uploads.append((filename, bucket, key, ExtraArgs))


class R2StorageTests(unittest.TestCase):
    def test_uploads_only_public_watermark_and_persists_mapping(self):
        with tempfile.TemporaryDirectory() as task_dir:
            watermark = os.path.join(task_dir, "output portrait_watermark.png")
            original = os.path.join(task_dir, "output portrait_original.png")
            with open(watermark, "wb") as file:
                file.write(b"preview")
            with open(original, "wb") as file:
                file.write(b"original")

            local_preview = "/taskfile/task-1/output%20portrait_watermark.png"
            local_original = "/taskfile/task-1/output%20portrait_original.png"
            config = R2StorageConfig(
                account_id="account",
                access_key_id="access",
                secret_access_key="secret",
                bucket="previews",
                public_base_url="https://images.example.com",
            )
            client = FakeS3Client()

            mapping = upload_public_task_images(
                "task-1",
                task_dir,
                [local_preview, local_original],
                config=config,
                s3_client=client,
            )

            self.assertEqual(len(client.uploads), 1)
            self.assertEqual(
                client.uploads[0][2],
                "ai-image-tasks/task-1/output portrait_watermark.png",
            )
            self.assertEqual(
                mapping[local_preview],
                "https://images.example.com/ai-image-tasks/task-1/"
                "output%20portrait_watermark.png",
            )
            self.assertNotIn(local_original, mapping)
            self.assertEqual(read_r2_mapping(task_dir), mapping)
            self.assertEqual(
                replace_with_cdn_urls([local_preview, local_original], mapping),
                [mapping[local_preview], local_original],
            )
            with open(os.path.join(task_dir, "cdn_uploads.json"), encoding="utf-8") as file:
                self.assertEqual(json.load(file)["version"], 1)

    def test_missing_configuration_keeps_local_urls(self):
        config = R2StorageConfig("", "", "", "", "")
        with tempfile.TemporaryDirectory() as task_dir:
            mapping = upload_public_task_images(
                "task-1", task_dir, ["/taskfile/task-1/a_watermark.png"], config=config
            )
        self.assertEqual(mapping, {})

    def test_task_status_serves_local_once_before_switching_to_cdn(self):
        task_id = "r2-delivery-test"
        local_url = f"/taskfile/{task_id}/output_watermark.png"
        cdn_url = (
            "https://images.example.com/ai-image-tasks/"
            f"{task_id}/output_watermark.png"
        )
        task_manager.create_task(task_id)
        task_manager.set_task_completed(
            task_id,
            {
                "output_files": [local_url],
                "cdn_output_files": [cdn_url],
                "cdn_ready": True,
            },
        )
        try:
            client = TestClient(app)
            first = client.get(f"/task-status/{task_id}")
            second = client.get(f"/task-status/{task_id}")

            self.assertEqual(first.json()["files"], [local_url])
            self.assertEqual(first.json()["file_source"], "local")
            self.assertEqual(second.json()["files"], [cdn_url])
            self.assertEqual(second.json()["file_source"], "cdn")
        finally:
            task_manager.tasks.pop(task_id, None)


if __name__ == "__main__":
    unittest.main()
