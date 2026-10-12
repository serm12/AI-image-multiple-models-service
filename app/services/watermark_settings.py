from app.services.i18n import t
"""Persist editable watermark settings on the existing durable task volume."""
import io
import json
import os
import uuid
from pathlib import Path
from typing import Literal

from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field
from app.core.config import DirectoryConfig, WatermarkConfig


class WatermarkSettings(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    layout: Literal['random', 'uniform_diagonal', 'grid', 'center'] = 'uniform_diagonal'
    opacity: float = Field(default=0.35, ge=0, le=1)
    size: int = Field(default=200, ge=40, le=1000)
    angle: float = Field(default=45, ge=-180, le=180)
    gap_x: int = Field(default=80, ge=10, le=1000)
    gap_y: int = Field(default=80, ge=10, le=1000)
    corner_text: str = Field(default=WatermarkConfig.CORNER_LABEL_TEXT, max_length=200)
    corner_position: Literal['top_right', 'top_left'] = 'top_right'
    corner_enabled: bool = True
    brand_enabled: bool = True
    brand_scale: float = Field(default=0.30, ge=0.05, le=0.6)
    brand_opacity: float = Field(default=0.8, ge=0, le=1)
    watermark_asset: str = ''
    brand_asset: str = ''


def settings_path():
    return Path(DirectoryConfig.TASKS_DIR) / '.watermark-settings.json'


def assets_dir():
    return Path(DirectoryConfig.TASKS_DIR) / '.watermark-assets'


def default_settings():
    c = WatermarkConfig
    return WatermarkSettings(layout='center' if c.STYLE == 'center' else c.TILED_LAYOUT,
        opacity=c.CENTER_LOGO_OPACITY if c.STYLE == 'center' else c.TILED_LOGO_OPACITY,
        size=round(c.TILED_LOGO_WIDTH * c.TILED_LOGO_SCALE), angle=c.TILED_ANGLE,
        gap_x=max(10, c.TILED_GAP_X), gap_y=max(10, c.TILED_GAP_Y),
        brand_scale=c.BRAND_LOGO_SCALE, brand_opacity=c.BRAND_LOGO_OPACITY,
        corner_position='top_left')


def load_settings():
    if not settings_path().exists():
        return default_settings()
    # Do not silently discard saved settings on corruption.
    return WatermarkSettings.model_validate_json(settings_path().read_text('utf-8'))


def save_settings(settings):
    path = settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'{path.name}.{uuid.uuid4().hex}.tmp')
    temporary.write_text(settings.model_dump_json(indent=2), encoding='utf-8')
    os.replace(temporary, path)


def asset_path(settings, kind):
    name = settings.watermark_asset if kind == 'watermark' else settings.brand_asset
    if name:
        if len(name) != 36 or not name.endswith('.png') or any(c not in '0123456789abcdef' for c in name[:-4]):
            raise ValueError('Invalid watermark asset')
        return assets_dir() / name
    return Path(WatermarkConfig.LOGO_PATH if kind == 'watermark' else WatermarkConfig.BRAND_LOGO_PATH)


def store_asset(content):
    if not content or len(content) > 5 * 1024 * 1024:
        raise ValueError(t('watermark.file_too_large'))
    try:
        with Image.open(io.BytesIO(content)) as image:
            if image.format not in {'PNG', 'JPEG', 'WEBP'} or image.width * image.height > 16000000:
                raise ValueError(t('watermark.invalid_dimensions'))
            image.load()
            name = uuid.uuid4().hex + '.png'
            assets_dir().mkdir(parents=True, exist_ok=True)
            image.convert('RGBA').save(assets_dir() / name, 'PNG')
            return name
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
        raise ValueError(t('watermark.invalid_image')) from error


def runtime_config(settings=None):
    if settings is None and not settings_path().exists():
        class Existing(WatermarkConfig):
            CORNER_ENABLED = True
            BRAND_ENABLED = True
            CORNER_POSITION = 'top_left'
        return Existing
    s = settings or load_settings()
    # A fresh class per render avoids changing shared state during concurrent jobs.
    class Snapshot(WatermarkConfig):
        pass
    c = Snapshot
    c.LOGO_PATH = str(asset_path(s, 'watermark'))
    c.BRAND_LOGO_PATH = str(asset_path(s, 'brand'))
    c.STYLE = 'center' if s.layout == 'center' else 'tiled'
    c.TILED_LAYOUT = 'uniform_diagonal' if s.layout in {'grid', 'uniform_diagonal'} else 'random'
    c.TILED_LOGO_WIDTH, c.TILED_LOGO_SCALE = s.size, 1.0
    c.TILED_LOGO_OPACITY = c.CENTER_LOGO_OPACITY = s.opacity
    c.TILED_ANGLE = 0 if s.layout == 'grid' else s.angle
    c.DEFAULT_ANGLE = s.angle
    c.DEFAULT_STEP = s.size + s.gap_x
    c.TILED_GAP_X, c.TILED_GAP_Y = s.gap_x, s.gap_y
    c.TILED_ROW_OFFSET = 0 if s.layout == 'grid' else 0.5
    c.CORNER_LABEL_TEXT, c.CORNER_POSITION = s.corner_text, s.corner_position
    c.CORNER_ENABLED, c.BRAND_ENABLED = s.corner_enabled, s.brand_enabled
    c.BRAND_LOGO_SCALE, c.BRAND_LOGO_OPACITY = s.brand_scale, s.brand_opacity
    return c
