import json
import mimetypes
import os
import posixpath
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote, unquote, urlparse


R2_METADATA_FILENAME = "cdn_uploads.json"
PUBLIC_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class R2StorageConfig:
    account_id: str
    access_key_id: str
    secret_access_key: str
    bucket: str
    public_base_url: str
    # Keep the CDN URL structure identical to the backend's local public URL:
    # /taskfile/{task_id}/{filename}.  This makes both sources interchangeable.
    prefix: str = "taskfile"
    enabled: bool = True

    @classmethod
    def from_env(cls) -> "R2StorageConfig":
        return cls(
            account_id=os.getenv("CLOUDFLARE_R2_ACCOUNT_ID", "").strip(),
            access_key_id=os.getenv("CLOUDFLARE_R2_ACCESS_KEY_ID", "").strip(),
            secret_access_key=os.getenv(
                "CLOUDFLARE_R2_SECRET_ACCESS_KEY", ""
            ).strip(),
            bucket=os.getenv("CLOUDFLARE_R2_BUCKET", "").strip(),
            public_base_url=os.getenv(
                "CLOUDFLARE_R2_PUBLIC_BASE_URL", ""
            ).strip().rstrip("/"),
            prefix=os.getenv("CLOUDFLARE_R2_PREFIX", "taskfile")
            .strip()
            .strip("/"),
            enabled=_env_bool("CLOUDFLARE_R2_ENABLED", True),
        )

    @property
    def is_configured(self) -> bool:
        return self.enabled and all(
            (
                self.account_id,
                self.access_key_id,
                self.secret_access_key,
                self.bucket,
                self.public_base_url,
            )
        )

    @property
    def endpoint_url(self) -> str:
        return f"https://{self.account_id}.r2.cloudflarestorage.com"


def is_public_preview_url(url: str) -> bool:
    filename = unquote(os.path.basename(urlparse(str(url)).path)).lower()
    stem, extension = os.path.splitext(filename)
    return stem.endswith("_watermark") and extension in PUBLIC_IMAGE_EXTENSIONS


def upload_public_task_images(
    task_id: str,
    task_dir: str,
    local_urls: list[str],
    *,
    config: R2StorageConfig | None = None,
    s3_client=None,
) -> dict[str, str]:
    """Upload public watermarked previews and persist local-to-CDN mappings."""
    config = config or R2StorageConfig.from_env()
    if not config.is_configured:
        return {}

    if s3_client is None:
        import boto3

        s3_client = boto3.client(
            "s3",
            endpoint_url=config.endpoint_url,
            aws_access_key_id=config.access_key_id,
            aws_secret_access_key=config.secret_access_key,
            region_name="auto",
        )

    resolved_task_dir = Path(task_dir).resolve()
    mapping: dict[str, str] = {}
    for local_url in local_urls:
        if not is_public_preview_url(local_url):
            continue
        filename = unquote(os.path.basename(urlparse(local_url).path))
        source = Path(resolved_task_dir, filename).resolve()
        try:
            source.relative_to(resolved_task_dir)
        except ValueError:
            continue
        if not source.is_file():
            continue

        key = posixpath.join(config.prefix, task_id, filename)
        content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
        s3_client.upload_file(
            str(source),
            config.bucket,
            key,
            ExtraArgs={
                "ContentType": content_type,
                "CacheControl": "public, max-age=31536000, immutable",
            },
        )
        mapping[local_url] = (
            f"{config.public_base_url}/{quote(key, safe='/')}"
        )

    if mapping:
        _write_r2_mapping(resolved_task_dir, mapping)
    return mapping


def read_r2_mapping(task_dir: str | Path) -> dict[str, str]:
    metadata_path = Path(task_dir, R2_METADATA_FILENAME)
    try:
        with metadata_path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
        files = payload.get("files", {}) if isinstance(payload, dict) else {}
        return files if isinstance(files, dict) else {}
    except (OSError, ValueError, TypeError):
        return {}


def replace_with_cdn_urls(
    local_urls: list[str], mapping: dict[str, str]
) -> list[str]:
    return [mapping.get(url, url) for url in local_urls]


def _write_r2_mapping(task_dir: Path, mapping: dict[str, str]) -> None:
    metadata_path = Path(task_dir, R2_METADATA_FILENAME)
    existing = read_r2_mapping(task_dir)
    existing.update(mapping)
    payload = {
        "version": 1,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "files": existing,
    }
    temporary_path = metadata_path.with_suffix(".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    os.replace(temporary_path, metadata_path)
