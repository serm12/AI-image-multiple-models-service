import os
from pathlib import Path

import aiofiles
from fastapi import UploadFile

from app.core.config import AppConfig, DirectoryConfig


UPLOAD_CHUNK_SIZE = 1024 * 1024


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
