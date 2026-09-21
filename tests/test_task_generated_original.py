import os
import tempfile
import time
from pathlib import Path

from app.core.config import DirectoryConfig
from app.services.task_files import resolve_task_generated_original


def test_resolves_newest_generated_original() -> None:
    previous_tasks_dir = DirectoryConfig.TASKS_DIR
    with tempfile.TemporaryDirectory() as tasks_dir:
        DirectoryConfig.TASKS_DIR = tasks_dir
        try:
            task_dir = Path(tasks_dir, "20260922_120000_abcdef12")
            task_dir.mkdir()
            older = task_dir / "output_cropped_original_1.png"
            newer = task_dir / "output_cropped_original_2.png"
            watermark = task_dir / "output_cropped_watermark.png"
            older.write_bytes(b"older")
            newer.write_bytes(b"newer")
            watermark.write_bytes(b"watermark")
            now = time.time()
            os.utime(older, (now - 10, now - 10))
            os.utime(newer, (now, now))

            assert resolve_task_generated_original(task_dir.name) == str(newer)
        finally:
            DirectoryConfig.TASKS_DIR = previous_tasks_dir


def test_rejects_invalid_task_id_and_missing_original() -> None:
    previous_tasks_dir = DirectoryConfig.TASKS_DIR
    with tempfile.TemporaryDirectory() as tasks_dir:
        DirectoryConfig.TASKS_DIR = tasks_dir
        try:
            empty_task = Path(tasks_dir, "20260922_120000_abcdef12")
            empty_task.mkdir()

            assert resolve_task_generated_original("../outside") is None
            assert resolve_task_generated_original(empty_task.name) is None
        finally:
            DirectoryConfig.TASKS_DIR = previous_tasks_dir
