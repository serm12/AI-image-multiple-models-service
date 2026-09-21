import os
import re
from pathlib import Path

import aiofiles
from fastapi import UploadFile

from app.core.config import AppConfig, DirectoryConfig


UPLOAD_CHUNK_SIZE = 1024 * 1024
TASK_ID_PATTERN = re.compile(r"^\d{8}_\d{6}_[0-9a-fA-F]{8}$")
GENERATED_ORIGINAL_PATTERN = "output_cropped_original_*"
GENERATED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


def safe_upload_filename(filename: str | None) -> str:
    safe_name = os.path.basename(filename or "upload")
    return safe_name or "upload"


async def save_validated_upload(file: UploadFile, destination: str) -> int:
    """Validate and save an uploaded image without loading the whole file into memory."""
    content_type = (file.content_type or "").split(";", 1)[0].strip().lower()
    if content_type not in AppConfig.ALLOWED_UPLOAD_CONTENT_TYPES:
        raise ValueError(f"不支持的文件类型: {file.content_type or 'unknown'}")

    max_bytes = AppConfig.MAX_UPLOAD_FILE_MB * 1024 * 1024
    total_bytes = 0
    async with aiofiles.open(destination, "wb") as f:
        while True:
            chunk = await file.read(UPLOAD_CHUNK_SIZE)
            if not chunk:
                break
            total_bytes += len(chunk)
            if total_bytes > max_bytes:
                raise ValueError(f"文件过大，单文件不能超过 {AppConfig.MAX_UPLOAD_FILE_MB}MB")
            await f.write(chunk)
    return total_bytes


def resolve_task_file_path(task_id: str, filename: str) -> str | None:
    task_dir = Path(DirectoryConfig.TASKS_DIR, task_id).resolve()
    file_path = Path(task_dir, filename).resolve()
    try:
        file_path.relative_to(task_dir)
    except ValueError:
        return None
    return str(file_path)


def resolve_task_generated_original(task_id: str) -> str | None:
    """Resolve the newest generated, watermark-free image for a known task."""
    if not TASK_ID_PATTERN.fullmatch(str(task_id or "")):
        return None

    tasks_root = Path(DirectoryConfig.TASKS_DIR).resolve()
    task_dir = Path(tasks_root, task_id).resolve()
    try:
        task_dir.relative_to(tasks_root)
    except ValueError:
        return None
    if not task_dir.is_dir():
        return None

    candidates = [
        path
        for path in task_dir.glob(GENERATED_ORIGINAL_PATTERN)
        if path.is_file() and path.suffix.lower() in GENERATED_IMAGE_SUFFIXES
    ]
    if not candidates:
        return None
    return str(max(candidates, key=lambda path: path.stat().st_mtime_ns))
