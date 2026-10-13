#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
项目配置文件
集中管理所有配置项、枚举类型和常量
"""

import os
import json
from enum import Enum
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


def env_float(name: str, default: float, minimum: float, maximum: float) -> float:
    """Read a bounded float without making the service fail on an invalid env value."""
    try:
        value = float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(maximum, value))

# 应用配置
class AppConfig:
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"
    PORT = int(os.getenv("PORT", "8001"))
    CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")
    _CORS_ORIGIN_REGEX = os.getenv(
        "CORS_ORIGIN_REGEX", r"https://([a-zA-Z0-9-]+\.)*shopifypreview\.com"
    ).strip()
    CORS_ORIGIN_REGEX = (
        rf"(?:{_CORS_ORIGIN_REGEX})|(?:https?://(?:localhost|127\.0\.0\.1)(?::\d+)?)"
        if _CORS_ORIGIN_REGEX
        else r"https?://(?:localhost|127\.0\.0\.1)(?::\d+)?"
    )
    MAX_CONCURRENT_TASKS = int(os.getenv("MAX_CONCURRENT_TASKS", "5"))
    ADMIN_API_KEY = os.getenv("ADMIN_API_KEY", "").strip()
    ADMIN_USER = os.getenv("ADMIN_USER", "").strip()
    ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "")
    # Used to sign browser-admin sessions. Set this to a long random value in production.
    ADMIN_SESSION_SECRET = os.getenv("ADMIN_SESSION_SECRET", "")
    MAX_UPLOAD_FILES = int(os.getenv("MAX_UPLOAD_FILES", "4"))
    MAX_UPLOAD_FILE_MB = int(os.getenv("MAX_UPLOAD_FILE_MB", "20"))
    ALLOWED_UPLOAD_CONTENT_TYPES = {
        item.strip().lower()
        for item in os.getenv("ALLOWED_UPLOAD_CONTENT_TYPES", "image/jpeg,image/png,image/webp").split(",")
        if item.strip()
    }

# 算法配置
class AlgorithmConfig:
    # 文件名算法因数：当前年份 * ALGORITHM_FACTOR
    ALGORITHM_FACTOR = int(os.getenv("ALGORITHM_FACTOR", "6987"))
    
    @classmethod
    def get_algorithm_value(cls):
        """获取算法值：当前年份 * 算法因数"""
        from datetime import datetime
        current_year = datetime.now().year
        from app.services.algorithm_settings import factor
        return current_year * factor()
class _StoredProviderKey:
    def __init__(self, name, fallback):
        self.name, self.fallback = name, fallback

    def __get__(self, instance, owner):
        from app.services.provider_credentials import resolve
        return resolve(self.name, self.fallback)

# API配置
class APIConfig:
    IMAGE_GENERATION_PROVIDER = os.getenv("IMAGE_GENERATION_PROVIDER", "seedream-4_replicate").strip() or "seedream-4_replicate"
    PROVIDER_FALLBACKS_ENABLED = os.getenv("PROVIDER_FALLBACKS_ENABLED", "true").lower() == "true"

    # Replicate 配置（用于生图流程或者放大流程...等其他流程）
    REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
    # Fireworks AI API配置
    FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY")
    # BFL 配置（用于生图流程）
    BFL_API_KEY = os.getenv("BFL_API_KEY")
    BFL_BASE_URL = os.getenv("BFL_BASE_URL", "https://api.bfl.ai/v1")
    
    # Gemini 配置（用于生图流程）
    GOOGLE_GEMINI_API_KEY = os.getenv("GOOGLE_GEMINI_API_KEY")
    
    # OpenRouter 配置（用于生图流程）
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    
    # Fal.ai 配置
    FAL_API_KEY = _StoredProviderKey("FAL_API_KEY", os.getenv("FAL_API_KEY"))
    FAL_GPT_IMAGE2_MODEL_ID = os.getenv("FAL_GPT_IMAGE2_MODEL_ID", "openai/gpt-image-2/edit").strip()
    FAL_GPT_IMAGE2_QUALITY = os.getenv("FAL_GPT_IMAGE2_QUALITY", "medium").strip() or "medium"
    FAL_SEEDREAM5_PRO_MODEL_ID = os.getenv(
        "FAL_SEEDREAM5_PRO_MODEL_ID", "bytedance/seedream/v5/pro/edit"
    ).strip() or "bytedance/seedream/v5/pro/edit"

    # aiapiroute/Sub2API GPT-image 配置（OpenAI 兼容 Images API）
    AIAPIROUTE_API_KEY = _StoredProviderKey("AIAPIROUTE_API_KEY", os.getenv("AIAPIROUTE_API_KEY") or os.getenv("SUB2API_API_KEY"))
    AIAPIROUTE_BASE_URL = os.getenv("AIAPIROUTE_BASE_URL") or os.getenv("SUB2API_BASE_URL", "https://aiapiroute.com")
    AIAPIROUTE_GPT_IMAGE1_MODEL = os.getenv("AIAPIROUTE_GPT_IMAGE1_MODEL", "gpt-image-1")
    AIAPIROUTE_GPT_IMAGE15_MODEL = os.getenv("AIAPIROUTE_GPT_IMAGE15_MODEL", "gpt-image-1.5")
    AIAPIROUTE_GPT_IMAGE2_MODEL = os.getenv("AIAPIROUTE_GPT_IMAGE2_MODEL", "gpt-image-2")
    AIAPIROUTE_GPT_IMAGE25_FLARE_MODEL = os.getenv(
        "AIAPIROUTE_GPT_IMAGE25_FLARE_MODEL",
        os.getenv("AIAPIROUTE_GPT_IMAGE25_MODEL", "gpt-image-2.5-flare"),
    )
    AIAPIROUTE_GPT_IMAGE25_SUNBURST_MODEL = os.getenv(
        "AIAPIROUTE_GPT_IMAGE25_SUNBURST_MODEL",
        "gpt-image-2.5-sunburst",
    )
    AIAPIROUTE_IMAGE_RESOLUTION = os.getenv("AIAPIROUTE_IMAGE_RESOLUTION", os.getenv("AIAPIROUTE_GPT_IMAGE2_RESOLUTION", "1K"))
    AIAPIROUTE_IMAGE_QUALITY = os.getenv("AIAPIROUTE_IMAGE_QUALITY", os.getenv("AIAPIROUTE_GPT_IMAGE2_QUALITY", ""))
    # Sub2API v0.2.5 image edits are most reliable with non-streaming JSON.
    AIAPIROUTE_IMAGE_STREAM = os.getenv("AIAPIROUTE_IMAGE_STREAM", os.getenv("AIAPIROUTE_GPT_IMAGE2_STREAM", "false")).lower() == "true"
    AIAPIROUTE_TIMEOUT_SECONDS = int(os.getenv("AIAPIROUTE_TIMEOUT_SECONDS", os.getenv("SUB2API_TIMEOUT_SECONDS", "300")))
    AIAPIROUTE_GPT_IMAGE2_FORCE_REFERENCE_RATIO = os.getenv(
        "AIAPIROUTE_GPT_IMAGE2_FORCE_REFERENCE_RATIO", "false"
    ).lower() == "true"
    AIAPIROUTE_GPT_IMAGE2_REFERENCE_RATIO = os.getenv(
        "AIAPIROUTE_GPT_IMAGE2_REFERENCE_RATIO", "1:1"
    ).strip()
    
    # 宽高比从宽到窄，横竖配对紧邻
    STANDARD_ASPECT_RATIOS = [
        "1:1",
        "21:9", "9:21",
        "16:9", "9:16",
        "2:1", "1:2",
        "3:2", "2:3",
        "4:3", "3:4",
        "4:5", "5:4",
    ]
    AIAPIROUTE_IMAGE_SIZES = ["1K", "2K", "4K", "auto"]

    PROVIDERS = {
        "flux_bfl": {
            "label": "BFL API (Flux)",
            "key": "BFL_API_KEY",
            "aspect_ratios": STANDARD_ASPECT_RATIOS,
            # Source: https://docs.bfl.ai/api-reference/models/edit-or-create-an-image-with-flux1-kontext-[pro]
        },
        "flux_replicate": {
            "label": "Replicate API (Flux)",
            "key": "REPLICATE_API_TOKEN",
            "aspect_ratios": ["match_input_image", *STANDARD_ASPECT_RATIOS],  # match_input_image: 输出图片尺寸自动匹配输入图片尺寸，仅支持图生图场景
            # Source: https://replicate.com/black-forest-labs/flux-kontext-pro/api/schema
        },
        "flux_fireworks": {
            "label": "Fireworks API (Flux)",
            "key": "FIREWORKS_API_KEY",
            "aspect_ratios": STANDARD_ASPECT_RATIOS,
            # Source: https://docs.fireworks.ai/api-reference/generate-or-edit-image-using-flux-kontext.md
        },
        "gemini-nanobanana_google": {
            "label": "Google Gemini API (Nano-Banana)",
            "key": "GOOGLE_GEMINI_API_KEY",
            "aspect_ratios": ["match_input_image", "1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"],
            # Source: https://ai.google.dev/gemini-api/docs/image-generation
        },
        "gemini-nanobanana_replicate": {
            "label": "Replicate API (Nano-Banana)",
            "key": "REPLICATE_API_TOKEN",
            "aspect_ratios": ["match_input_image", "1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"],
            # Source: https://replicate.com/google/nano-banana/api/schema
        },
        "gemini-nanobanana_openrouter": {
            "label": "OpenRouter API (Nano-Banana)",
            "key": "OPENROUTER_API_KEY",
            "aspect_ratios": ["match_input_image", "1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"],
            # Source: https://openrouter.ai/docs/features/multimodal/images
        },
        "seedream-4_replicate": {
            "label": "Replicate API (Seedream-4)",
            "key": "REPLICATE_API_TOKEN",
            "aspect_ratios": ["match_input_image", "1:1", "4:3", "3:4", "16:9", "9:16", "3:2", "2:3", "21:9"],
            # Source: https://replicate.com/bytedance/seedream-4/api/schema
        },
        "seedream-4_fal": {
            "label": "fal.ai API (Seedream-4)",
            "key": "FAL_API_KEY",
            "aspect_ratios": ["1:1", "4:3", "3:4", "16:9", "9:16"],
            # Source: https://fal.ai/models/fal-ai/bytedance/seedream/v4/edit/api
        },
        "seedream-5-pro_fal": {
            "label": "fal.ai API (Seedream 5 Pro)",
            "key": "FAL_API_KEY",
            "aspect_ratios": ["match_input_image", "1:1", "4:3", "3:4", "16:9", "9:16"],
            # Source: https://fal.ai/models/bytedance/seedream/v5/pro/edit/api
        },
        "gpt-image-2_fal": {
            "label": "fal.ai API (GPT Image 2)",
            "key": "FAL_API_KEY",
            "aspect_ratios": ["match_input_image", *STANDARD_ASPECT_RATIOS],
            "sizes": AIAPIROUTE_IMAGE_SIZES,
        },
        "aiapiroute_gpt-image-1": {
            "label": "aiapiroute/Sub2API gpt-image-1",
            "key": "AIAPIROUTE_API_KEY",
            "model": AIAPIROUTE_GPT_IMAGE1_MODEL,
            "aspect_ratios": STANDARD_ASPECT_RATIOS,
            "sizes": AIAPIROUTE_IMAGE_SIZES,
        },
        "aiapiroute_gpt-image-1.5": {
            "label": "aiapiroute/Sub2API gpt-image-1.5",
            "key": "AIAPIROUTE_API_KEY",
            "model": AIAPIROUTE_GPT_IMAGE15_MODEL,
            "aspect_ratios": STANDARD_ASPECT_RATIOS,
            "sizes": AIAPIROUTE_IMAGE_SIZES,
        },
        "gpt-image-2_aiapiroute": {
            "label": "aiapiroute/Sub2API gpt-image-2",
            "key": "AIAPIROUTE_API_KEY",
            "model": AIAPIROUTE_GPT_IMAGE2_MODEL,
            "aspect_ratios": STANDARD_ASPECT_RATIOS,
            "sizes": AIAPIROUTE_IMAGE_SIZES,
        },
        "gpt-image-2.5-flare_aiapiroute": {
            "label": "aiapiroute/Sub2API gpt-image-2.5 Flare",
            "key": "AIAPIROUTE_API_KEY",
            "model": AIAPIROUTE_GPT_IMAGE25_FLARE_MODEL,
            "aspect_ratios": STANDARD_ASPECT_RATIOS,
            "sizes": AIAPIROUTE_IMAGE_SIZES,
        },
        "gpt-image-2.5-sunburst_aiapiroute": {
            "label": "aiapiroute/Sub2API gpt-image-2.5 Sunburst",
            "key": "AIAPIROUTE_API_KEY",
            "model": AIAPIROUTE_GPT_IMAGE25_SUNBURST_MODEL,
            "aspect_ratios": STANDARD_ASPECT_RATIOS,
            "sizes": AIAPIROUTE_IMAGE_SIZES,
        },
    }

    ALL_PROVIDERS = list(PROVIDERS.keys())
    DEFAULT_PROVIDER_FALLBACK_GROUPS = {
        "flux": ["flux_bfl", "flux_replicate", "flux_fireworks"],
        "gpt-image-2": ["gpt-image-2_aiapiroute", "gpt-image-2_fal"],
        "gpt-image-2.5": [
            "gpt-image-2.5-flare_aiapiroute",
            "gpt-image-2.5-sunburst_aiapiroute",
        ],
        "seedream-4": ["seedream-4_replicate", "seedream-4_fal"],
        "seedream-5-pro": ["seedream-5-pro_fal"],
        "gemini-nanobanana": [
            "gemini-nanobanana_google",
            "gemini-nanobanana_replicate",
            "gemini-nanobanana_openrouter",
        ],
    }
    PROVIDER_FALLBACK_GROUPS = DEFAULT_PROVIDER_FALLBACK_GROUPS
    AIAPIROUTE_PROVIDER_MODEL_MAP = {
        provider: config["model"]
        for provider, config in PROVIDERS.items()
        if "model" in config
    }
    PROVIDER_ASPECT_RATIO_MAP = {
        provider: config.get("aspect_ratios", [])
        for provider, config in PROVIDERS.items()
    }

    @classmethod
    def _load_provider_fallback_groups(cls) -> dict:
        raw = os.getenv("PROVIDER_FALLBACK_GROUPS_JSON", "").strip()
        if not raw:
            return cls.DEFAULT_PROVIDER_FALLBACK_GROUPS
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"PROVIDER_FALLBACK_GROUPS_JSON 不是有效 JSON: {exc}") from exc
        if not isinstance(parsed, dict):
            raise ValueError("PROVIDER_FALLBACK_GROUPS_JSON 必须是对象，例如 {\"flux\":[\"flux_bfl\"]}")
        groups = {}
        for group_name, providers in parsed.items():
            if not isinstance(providers, list) or not all(isinstance(item, str) for item in providers):
                raise ValueError(f"兜底组 {group_name} 必须是 provider 字符串数组")
            unknown = [item for item in providers if item not in cls.ALL_PROVIDERS]
            if unknown:
                raise ValueError(f"兜底组 {group_name} 包含不支持的 provider: {unknown}")
            groups[str(group_name)] = providers
        return groups

    @classmethod
    def get_provider_fallback_groups(cls) -> dict:
        cls.PROVIDER_FALLBACK_GROUPS = cls._load_provider_fallback_groups()
        return cls.PROVIDER_FALLBACK_GROUPS

    @classmethod
    def find_fallback_group(cls, provider: str) -> str | None:
        for group_name, providers in cls.get_provider_fallback_groups().items():
            if provider == group_name or provider in providers:
                return group_name
        return None

    @classmethod
    def validate_provider(cls, provider: str):
        """运行时校验指定服务商是否已配置 API Key，未配置时抛出 ValueError"""
        if provider not in cls.ALL_PROVIDERS:
            raise ValueError(f"不支持的服务提供商: {provider}，可选: {cls.ALL_PROVIDERS}")
        key_name = cls.PROVIDERS[provider].get("key")
        if key_name and not getattr(cls, key_name, None):
            raise ValueError(f"使用 {provider} 需要设置环境变量 {key_name}")

    @classmethod
    def resolve_provider(cls, provider: str | None, validate_config: bool = True) -> str:
        """解析本次请求使用的 primary provider；空值回退到 IMAGE_GENERATION_PROVIDER。"""
        provider_value = getattr(provider, "value", provider)
        effective_provider = (provider_value or cls.IMAGE_GENERATION_PROVIDER).strip()
        fallback_group = cls.find_fallback_group(effective_provider)
        if fallback_group and effective_provider == fallback_group:
            effective_provider = cls.get_provider_fallback_groups()[fallback_group][0]
        if validate_config:
            cls.validate_provider(effective_provider)
        elif effective_provider not in cls.ALL_PROVIDERS:
            raise ValueError(f"不支持的默认服务提供商: {effective_provider}，可选: {cls.ALL_PROVIDERS}")
        return effective_provider

    @classmethod
    def resolve_provider_chain(cls, provider: str | None, validate_config: bool = True) -> list[str]:
        """返回本次请求可尝试的 provider 顺序，用于同模型多服务商兜底。"""
        primary_provider = cls.resolve_provider(provider, validate_config=False)
        if not cls.PROVIDER_FALLBACKS_ENABLED:
            chain = [primary_provider]
        else:
            group_name = cls.find_fallback_group(primary_provider)
            configured_group = cls.get_provider_fallback_groups()
            chain = list(configured_group.get(group_name, [primary_provider])) if group_name else [primary_provider]
            if primary_provider in chain:
                chain = [primary_provider, *[item for item in chain if item != primary_provider]]
        if validate_config:
            configured = []
            missing = []
            for candidate in chain:
                try:
                    cls.validate_provider(candidate)
                    configured.append(candidate)
                except ValueError as exc:
                    missing.append(str(exc))
            if configured:
                return configured
            raise ValueError("兜底链里没有可用 provider: " + "; ".join(missing))
        return chain

    @classmethod
    def validate_aspect_ratio(cls, provider: str, aspect_ratio: str):
        """校验 provider 是否支持指定 aspect_ratio；空列表表示该 provider 不使用此参数。"""
        ratio_value = getattr(aspect_ratio, "value", aspect_ratio)
        supported_ratios = cls.PROVIDER_ASPECT_RATIO_MAP.get(provider, [])
        if not supported_ratios:
            return
        if ratio_value not in supported_ratios:
            raise ValueError(f"{provider} 不支持 aspect_ratio={ratio_value}，可选: {supported_ratios}")

    @classmethod
    def get_provider_options(cls) -> dict:
        """返回每个 provider 支持的测试参数，用于 Swagger/前端选择联动。"""
        return {
            provider: {
                "label": config.get("label", provider),
                "configured": bool(getattr(cls, config.get("key"), None)) if config.get("key") else True,
                "is_default": provider == cls.IMAGE_GENERATION_PROVIDER,
                "aspect_ratios": config.get("aspect_ratios", []),
                "uses_aspect_ratio": bool(config.get("aspect_ratios", [])),
                "sizes": config.get("sizes", []),
                "uses_size": bool(config.get("sizes", [])),
                "fallback_group": cls.find_fallback_group(provider),
                "fallback_chain": cls.resolve_provider_chain(provider, validate_config=False),
                **({"model": config["model"]} if "model" in config else {}),
            }
            for provider, config in cls.PROVIDERS.items()
        }

    @classmethod
    def get_available_providers(cls) -> list:
        """返回所有已配置 API Key 的服务提供商列表"""
        result = []
        for p, config in cls.PROVIDERS.items():
            key_name = config.get("key")
            configured = bool(getattr(cls, key_name, None)) if key_name else True
            result.append({
                "provider": p,
                "label": config.get("label", p),
                "configured": configured,
                "is_default": p == cls.IMAGE_GENERATION_PROVIDER,
                "fallback_group": cls.find_fallback_group(p),
            })
        return result


class ProviderEnum(str, Enum):
    flux_bfl = "flux_bfl"
    flux_replicate = "flux_replicate"
    flux_fireworks = "flux_fireworks"
    gemini_nanobanana_google = "gemini-nanobanana_google"
    gemini_nanobanana_replicate = "gemini-nanobanana_replicate"
    gemini_nanobanana_openrouter = "gemini-nanobanana_openrouter"
    seedream_4_replicate = "seedream-4_replicate"
    seedream_4_fal = "seedream-4_fal"
    gpt_image_2_fal = "gpt-image-2_fal"
    aiapiroute_gpt_image_1 = "aiapiroute_gpt-image-1"
    aiapiroute_gpt_image_1_5 = "aiapiroute_gpt-image-1.5"
    gpt_image_2_aiapiroute = "gpt-image-2_aiapiroute"
    gpt_image_2_5_flare_aiapiroute = "gpt-image-2.5-flare_aiapiroute"
    gpt_image_2_5_sunburst_aiapiroute = "gpt-image-2.5-sunburst_aiapiroute"

# 目录配置
class DirectoryConfig:
    # 生产环境目录
    TASKS_DIR = "tasks"  # 生产API调用的输出目录
    ASSETS_DIR = "assets"
    
    # 测试相关目录
    TEST_INPUT_DIR = "input_image"      # 测试输入图片目录
    TEST_OUTPUT_DIR = "generated_images"  # 测试输出图片目录
    
    @classmethod
    def ensure_directories(cls):
        """确保所有必要目录存在"""
        os.makedirs(cls.TASKS_DIR, exist_ok=True)
        os.makedirs(cls.ASSETS_DIR, exist_ok=True)
    
    @classmethod
    def ensure_test_directories(cls):
        """确保测试相关目录存在"""
        os.makedirs(cls.TEST_INPUT_DIR, exist_ok=True)
        os.makedirs(cls.TEST_OUTPUT_DIR, exist_ok=True)
        print(f"📁 测试目录已准备:")
        print(f"   📷 输入目录: {cls.TEST_INPUT_DIR} (测试图片)")
        print(f"   📄 输出目录: {cls.TEST_OUTPUT_DIR} (生成的图片)")
        print(f"   🔧 生产目录: {cls.TASKS_DIR} (API调用输出)")


# 支持的模型版本
class FluxModelEnum(str, Enum):
    max = "black-forest-labs/flux-kontext-max"
    pro = "black-forest-labs/flux-kontext-pro"
    dev = "black-forest-labs/flux-kontext-dev"


# 官方支持的 aspect_ratio 选项
class AspectRatioEnum(str, Enum):
    match_input_image = "match_input_image"
    ratio_1_1 = "1:1"
    ratio_16_9 = "16:9"
    ratio_9_16 = "9:16"
    ratio_4_3 = "4:3"
    ratio_3_4 = "3:4"
    ratio_3_2 = "3:2"
    ratio_2_3 = "2:3"
    ratio_4_5 = "4:5"
    ratio_5_4 = "5:4"
    ratio_21_9 = "21:9"
    ratio_9_21 = "9:21"
    ratio_2_1 = "2:1"
    ratio_1_2 = "1:2"


# 水印配置
class WatermarkConfig:
    # 水印铺设样式：tiled 为原有全屏平铺，center 为画面中央单个水印。
    STYLE = os.getenv("WATERMARK_STYLE", "tiled").strip().lower() or "tiled"
    DEFAULT_TEXT = "Paintingify"
    DEFAULT_FONT_PATH = "assets/holidayvibesfreeregular-wppxv.ttf"
    DEFAULT_FONT_SIZE = 48
    # 铺满画面的 Paintingify 文字水印颜色；最后一位是 alpha 值（0=完全透明，255=完全不透明）。
    # 153 等于 60% 不透明度，强化防裁剪效果，同时保留成品细节。
    DEFAULT_COLOR = (255, 255, 255, 153)
    DEFAULT_ANGLE = 30
    DEFAULT_STEP = 165
    DEFAULT_PADDING = 10
    # PNG Logo 水印需保留原始黑白颜色；screen 会将黑色几乎完全滤掉。
    DEFAULT_BLEND_MODE = "normal"
    DEFAULT_RESIZE_SCALE = env_float("WATERMARK_OUTPUT_SCALE", 0.5, 0.1, 1.0)

    @classmethod
    def get_resize_scale(cls):
        return cls.DEFAULT_RESIZE_SCALE

    ENABLE_RANDOM_PATTERNS = True
    RANDOM_ANGLE_RANGE = 25
    RANDOM_SIZE_RANGE = 0.2
    RANDOM_OPACITY_RANGE = 30
    MULTIPLE_BLEND_MODES = ["normal"]

    # The source artwork used by both tiled and centered logo watermarks.
    LOGO_PATH = (
        os.getenv("WATERMARK_LOGO_PATH", "assets/logo_watermark-big_black.png").strip()
        or "assets/logo_watermark-big_black.png"
    )
    # Keep tiled marks at a readable density even when the source artwork is large.
    TILED_LOGO_WIDTH = round(
        env_float("WATERMARK_TILED_LOGO_BASE_WIDTH", 200.0, 1.0, 4000.0)
    )
    TILED_LOGO_SCALE = env_float("WATERMARK_TILED_LOGO_SCALE", 1.0, 0.05, 3.0)
    TILED_LOGO_OPACITY = env_float("WATERMARK_TILED_LOGO_OPACITY", 0.35, 0.0, 1.0)
    TILED_LAYOUT = (
        os.getenv("WATERMARK_TILED_LAYOUT", "random").strip().lower() or "random"
    )
    if TILED_LAYOUT not in {"random", "uniform_diagonal"}:
        TILED_LAYOUT = "random"
    TILED_ANGLE = env_float("WATERMARK_TILED_ANGLE", 45.0, -180.0, 180.0)
    TILED_GAP_X = round(env_float("WATERMARK_TILED_GAP_X", 80.0, 0.0, 2000.0))
    TILED_GAP_Y = round(env_float("WATERMARK_TILED_GAP_Y", 80.0, 0.0, 2000.0))
    TILED_ROW_OFFSET = env_float("WATERMARK_TILED_ROW_OFFSET", 0.5, 0.0, 1.0)
    CENTER_LOGO_COUNT = 1
    CENTER_LOGO_WIDTH_RATIO = 0.88
    CENTER_LOGO_SCALE = env_float("WATERMARK_CENTER_LOGO_SCALE", 1.0, 0.05, 1.0)
    CENTER_LOGO_OPACITY = env_float("WATERMARK_CENTER_LOGO_OPACITY", 0.55, 0.0, 1.0)
    CENTER_LOGO_VERTICAL_OFFSET_RATIO = 0.10

    CORNER_LABEL_TEXT = "Purchase to get high-resolution watermark-free image by Paintingify.com"
    CORNER_LABEL_FONT_PATH = "assets/ARIAL.TTF"
    CORNER_LABEL_FONT_SIZE = 28
    CORNER_LABEL_PADDING = 12
    CORNER_LABEL_MARGIN = 12
    CORNER_LABEL_TEXT_COLOR = (255, 255, 255, 500)
    CORNER_LABEL_BG_COLOR = None
    CORNER_LABEL_BORDER_COLOR = (255, 255, 255, 500)
    CORNER_LABEL_BORDER_WIDTH = 2
    CORNER_LABEL_CORNER_RADIUS = 8

    BRAND_LOGO_PATH = "assets/logo_brand.png"
    BRAND_LOGO_MARGIN = 20
    BRAND_LOGO_SCALE = 0.30
    BRAND_LOGO_MIN_SIZE = 80
    BRAND_LOGO_MAX_SIZE = 400
    BRAND_LOGO_OPACITY = 0.8


# 图片格式选项
class OutputFormatEnum(str, Enum):
    png = "png"
    jpg = "jpg"

# 通用绘画风格：所有模型共用同一组标识与提示词。
class ArtStyleEnum(str, Enum):
    pixel_art = 'pixel_art'
    realistic = 'realistic'
    oil_painting = 'oil_painting'
    heavy_oil_painting = 'heavy_oil_painting'
    watercolor = 'watercolor'
    anime = 'anime'
    japanese_anime = 'japanese_anime'
    detailed_anime = 'detailed_anime'
    sketch = 'sketch'
    cartoon = 'cartoon'
    pixar = 'pixar'
    cyberpunk = 'cyberpunk'
    fantasy = 'fantasy'
    steampunk = 'steampunk'
    abstract = 'abstract'
    acrylic = 'acrylic'
    tattoo = 'tattoo'
    illustration = 'illustration'
    thick_paint_illustration = 'thick_paint_illustration'
    thick_paint_glossy = 'thick_paint_glossy'
    thick_paint_portrait = 'thick_paint_portrait'
    digital_paint_illustration = 'digital_paint_illustration'
    digital_paint_glossy = 'digital_paint_glossy'
    bright_cartoon_adventure = 'bright_cartoon_adventure'
    style_3d_cartoon_doll = '3d_cartoon_doll'

    @classmethod
    def _missing_(cls, value):
        # 旧请求和历史任务仍可读取；对外只返回不带模型前缀的名称。
        canonical = STYLE_ALIASES.get(value) if isinstance(value, str) else None
        return cls(canonical) if canonical else None


STYLE_ALIASES = {
    'flux_pixel_art': 'pixel_art',
    'flux_pixel_art_16': 'pixel_art',
    'flux_cartoon_pixel_art': 'pixel_art',
    'flux_realistic': 'realistic',
    'flux_oil_painting': 'oil_painting',
    'flux_heavy_oil_painting': 'heavy_oil_painting',
    'flux_watercolor': 'watercolor',
    'flux_anime': 'anime',
    'flux_japanese_anime': 'japanese_anime',
    'flux_detailed_anime': 'detailed_anime',
    'flux_sketch': 'sketch',
    'flux_cartoon': 'cartoon',
    'flux_pixar': 'pixar',
    'flux_cyberpunk': 'cyberpunk',
    'flux_fantasy': 'fantasy',
    'flux_steampunk': 'steampunk',
    'flux_abstract': 'abstract',
    'flux_watercolor_1': 'watercolor',
    'flux_acrylic': 'acrylic',
    'flux_tattoo': 'tattoo',
    'gemini_realistic': 'realistic',
    'gemini_oil_painting': 'oil_painting',
    'gemini_illustration': 'illustration',
    'gemini_thick_paint_illustration': 'thick_paint_illustration',
    'gemini_thick_paint_glossy': 'thick_paint_glossy',
    'gemini_thick_paint_portrait': 'thick_paint_portrait',
    'gemini_digital_paint_illustration': 'digital_paint_illustration',
    'gemini_digital_paint_glossy': 'digital_paint_glossy',
    'gemini_acrylic': 'acrylic',
    'gemini_watercolor': 'watercolor',
    'chatgpt_bright_cartoon_adventure': 'bright_cartoon_adventure',
    'seedream-4_realistic': 'realistic',
    'seedream-4_fal_realistic': 'realistic',
    'gemini_replicate_oil_painting': 'oil_painting',
}

# 每种风格只维护一份描述和提示词。
STYLE_PRESETS = {
    'pixel_art': ('像素画 - 清晰像素边缘、有限调色板和复古游戏质感', 'Render as retro pixel art with crisp pixel edges, a limited palette and deliberate pixel clusters. Keep the requested subjects recognizable at the chosen pixel scale'),
    'realistic': ('写实 - 保持原始提示词，不额外添加风格', ''),
    'oil_painting': ('油画 - 层叠油彩、适度笔触和细腻画布质感', 'Render in oils with layered color, controlled brushwork and subtle canvas texture. For reference subjects, retain recognizable features and original skin or fur colors'),
    'heavy_oil_painting': ('厚重油画 - 浮雕厚涂，面部笔触适度', 'Render in heavy impasto oils with raised paint and bold brushstrokes. Concentrate the strongest texture in clothing and backgrounds; keep facial features recognizable with controlled brushwork'),
    'watercolor': ('水彩 - 透明叠色、自然晕染和轻柔纸纹', 'Render in watercolor with translucent washes, soft pigment blooms and subtle paper grain. Balance diffused edges with clear defining features'),
    'anime': ('动漫 - 清晰线稿、赛璐璐上色和鲜明色彩', 'Render as Japanese anime with clean linework, cel shading and a clear color palette. Stylize reference subjects while retaining their distinguishing features'),
    'japanese_anime': ('手绘动漫 - 柔和色彩、手绘线条和细腻笔触', 'Render as hand-drawn Japanese animation with delicate outlines, softly blended colors and a gently painted finish. Retain the requested setting and character details'),
    'detailed_anime': ('精致动漫 - 精细刻画人物及已有服饰细节', 'Render as detailed anime illustration with precise linework and refined shading. Carefully depict the specified clothing and existing accessories while retaining distinguishing character features'),
    'sketch': ('素描 - 单色铅笔线条、炭笔明暗和纸纹', 'Render as a monochrome pencil and charcoal drawing with varied line weight, controlled hatching and soft tonal shading on paper'),
    'cartoon': ('卡通 - 明确轮廓、简洁形状和鲜明色彩', 'Render as a cartoon illustration with confident outlines, simplified shapes and clear color blocks. Keep the requested subjects recognizable within the stylization'),
    'pixar': ('3D动画 - 生动角色、细腻材质和电影光影', 'Render with a polished 3D animated feature-film look, expressive character design, nuanced materials and soft cinematic lighting. Retain distinguishing features of reference subjects'),
    'cyberpunk': ('赛博朋克 - 霓虹边缘光、科技材质和高反差色彩', 'Use a cyberpunk visual treatment with neon rim lighting, high-contrast colors and futuristic material finishes. Apply these qualities to the requested scene rather than introducing a new city setting'),
    'fantasy': ('奇幻 - 梦幻光影、绘画质感和魔法氛围', 'Render as fantasy illustration with luminous painterly shading, ethereal light and a sense of wonder. Keep the characters and setting specified in the main prompt'),
    'steampunk': ('蒸汽朋克 - 复古工业美学、黄铜和铜质感', 'Use a steampunk visual treatment with Victorian-inspired craftsmanship, aged brass and copper finishes. Apply these details to the requested objects and clothing without adding unrelated machinery'),
    'abstract': ('抽象表现 - 简化形态和色彩构成，保留主体辨识度', 'Render with expressive abstract shapes, layered color and rhythmic brushwork. When a recognizable subject is requested, retain its defining silhouette and key features'),
    'acrylic': ('丙烯 - 鲜明色彩、干净哑光画面和自然肤色', 'Render in acrylics with rich color, subtle brushwork and a clean matte finish. For faces, retain recognizable features and original skin tones, using soft tonal transitions and gentle painted highlights instead of coarse texture or plastic shine'),
    'tattoo': ('纹身艺术 - 黑色轮廓、纯色填充和墨线阴影', 'Render as traditional tattoo flash artwork with bold black contours, solid color fills and deliberate ink shading. Follow the background specified in the main prompt'),
    'illustration': ('插画 - 干净线条、简化形状和协调配色', 'Render as modern editorial illustration with clean linework, simplified shapes, restrained shading and a cohesive palette'),
    'thick_paint_illustration': ('厚涂插画 - 层叠笔触、色块塑形和立体明暗', 'Render as a digital painterly illustration with layered visible brushstrokes, broad color planes and volumetric shading. Keep defining features of reference subjects clear'),
    'thick_paint_glossy': ('厚涂高光 - 层叠笔触和受控的材质反光', 'Render with layered impasto brushwork and dramatic light-shadow contrast. Use distinct specular highlights on reflective materials while keeping facial texture controlled'),
    'thick_paint_portrait': ('厚涂肖像 - 清晰五官、自然肤色和层叠笔触', 'Render as a painterly impasto portrait with sculptural lighting. Preserve facial likeness and original skin tone, using controlled facial brushwork and bolder paint texture in clothing and backgrounds'),
    'digital_paint_illustration': ('数字绘画 - 光滑渐变、细腻明暗和干净画面', 'Render as a polished digital painting with smooth color transitions, precise edges and dimensional shading. Use highlights appropriate to each material'),
    'digital_paint_glossy': ('数字高光肖像 - 精准五官、自然肤色和细腻高光', 'Render as a polished digital portrait with smooth brushwork, dimensional lighting and restrained highlights. Preserve facial likeness and original skin tone with a painted finish'),
    'bright_cartoon_adventure': ('亮丽卡通冒险 - 明快配色、欢乐氛围和电影光影', 'Render as a colorful 3D animated adventure with a cheerful mood, a cohesive bright palette and cinematic lighting. Use balanced character proportions and recognizable features, with colors suited to the requested setting'),
    '3d_cartoon_doll': ('3D卡通娃娃 - 圆润造型、柔和乙烯基材质和玩具质感', 'Render as a collectible 3D vinyl cartoon doll with rounded stylized proportions, smooth toy surfaces and glossy eyes. Preserve distinguishing features of reference subjects within the doll design'),
}

STYLE_PROMPTS = {
    ArtStyleEnum(value): f"{prompt}. " if prompt else ""
    for value, (_, prompt) in STYLE_PRESETS.items()
}
STYLE_DESCRIPTIONS = {
    value: description for value, (description, _) in STYLE_PRESETS.items()
}

# 初始化配置
def initialize_config():
    """初始化配置，确保必要的目录存在"""
    DirectoryConfig.ensure_directories()
    if APIConfig.REPLICATE_API_TOKEN:
        os.environ["REPLICATE_API_TOKEN"] = APIConfig.REPLICATE_API_TOKEN

def initialize_test_config():
    """初始化测试配置，确保测试目录存在"""
    DirectoryConfig.ensure_test_directories()

