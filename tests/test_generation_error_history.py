import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from app.core.config import DirectoryConfig
from app.routers import api
from app.services import task_list_index, task_query_service
from app.services.task_files import resolve_task_file_path


class GenerationErrorHistoryTests(unittest.IsolatedAsyncioTestCase):
    async def test_provider_failure_is_persisted_and_shown_after_memory_is_cleared(self):
        task_id = "20260101_120000_aabbccdd"
        with tempfile.TemporaryDirectory() as folder, patch.object(DirectoryConfig, "TASKS_DIR", folder):
            task_folder = Path(folder, task_id)
            task_folder.mkdir()
            params = {"task_id": task_id, "api_provider": "gpt-image-2_aiapiroute"}
            (task_folder / "params.json").write_text(json.dumps(params), encoding="utf-8")
            api.task_manager.create_task(task_id, api_provider=params["api_provider"])
            try:
                with patch.object(task_list_index, "refresh_task"), patch.object(
                    api.api_client, "generate_image", AsyncMock(side_effect=ValueError("429 Too Many Requests"))
                ):
                    await api.process_generation_background(
                        task_id, str(task_folder), "test", [], None, None, None, None, params,
                        params["api_provider"],
                    )
            finally:
                api.task_manager.tasks.pop(task_id, None)

            task = task_query_service.list_task_summaries()["tasks"][0]
            self.assertEqual(task["status"], "failed")
            self.assertIn("429 Too Many Requests", task["generation_error"])
            self.assertIsNone(resolve_task_file_path(task_id, ".generation-error.json"))
