#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
项目配置文件
集中管理所有配置项、枚举类型和常量
"""

import os
from enum import Enum
from dotenv import dotenv_values

# 加载环境变量
_project_env = dotenv_values()

def project_getenv(name, default=None):
    """Project overrides never mutate the shared process environment."""
    return os.getenv('MY3DFIGURE_' + name, _project_env.get('MY3DFIGURE_' + name, os.getenv(name, _project_env.get(name, default))))

def env_float(name: str, default: float, minimum: float, maximum: float) -> float:
    """Read a bounded float without making the service fail on an invalid env value."""
    try:
        value = float(project_getenv(name, str(default)))
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(maximum, value))

# 应用配置
class AppConfig:
    DEBUG = project_getenv("DEBUG", "False").lower() == "true"
    PORT = int(project_getenv("PORT", "8001"))
    CORS_ORIGINS = project_getenv("CORS_ORIGINS", "*").split(",")
    CORS_ORIGIN_REGEX = (
        project_getenv("CORS_ORIGIN_REGEX", r"https://([a-zA-Z0-9-]+\.)*shopifypreview\.com").strip()
        or None
    )
    MAX_CONCURRENT_TASKS = int(project_getenv("MAX_CONCURRENT_TASKS", "5"))
    ADMIN_API_KEY = project_getenv("ADMIN_API_KEY", "").strip()
    ADMIN_USER = project_getenv("ADMIN_USER", "").strip()
    ADMIN_PASSWORD = project_getenv("ADMIN_PASSWORD", "")
    MAX_UPLOAD_FILES = int(project_getenv("MAX_UPLOAD_FILES", "4"))
    MAX_UPLOAD_FILE_MB = int(project_getenv("MAX_UPLOAD_FILE_MB", "20"))
    ALLOWED_UPLOAD_CONTENT_TYPES = {
        item.strip().lower()
        for item in project_getenv("ALLOWED_UPLOAD_CONTENT_TYPES", "image/jpeg,image/png,image/webp").split(",")
        if item.strip()
    }

# 算法配置
class AlgorithmConfig:
    # 文件名算法因数：当前年份 * ALGORITHM_FACTOR
    ALGORITHM_FACTOR = int(project_getenv("ALGORITHM_FACTOR", "6987"))

    @classmethod
    def get_algorithm_value(cls):
        """获取算法值：当前年份 * 算法因数"""
        from datetime import datetime
        current_year = datetime.now().year
        return current_year * cls.ALGORITHM_FACTOR
# API配置
class APIConfig:
    # Fal 开关：只切换下面 True 行的注释；取消注释开启，加回注释关闭。
    FAL_GPT_IMAGE2_ENABLED = False
    # FAL_GPT_IMAGE2_ENABLED = True
    IMAGE_GENERATION_PROVIDER = project_getenv("IMAGE_GENERATION_PROVIDER", "gpt-image-2_aiapiroute").strip() or "gpt-image-2_aiapiroute"
    if IMAGE_GENERATION_PROVIDER not in {"gpt-image-2_aiapiroute", "gpt-image-2_fal"} or (
        IMAGE_GENERATION_PROVIDER == "gpt-image-2_fal" and not FAL_GPT_IMAGE2_ENABLED
    ):
        IMAGE_GENERATION_PROVIDER = "gpt-image-2_aiapiroute"

    # Replicate remains configured for the independent upscale utility.
    REPLICATE_API_TOKEN = project_getenv("REPLICATE_API_TOKEN")

    # Keep Fal credentials and model settings for the retained, disabled client.
    FAL_API_KEY = project_getenv("FAL_API_KEY")
    FAL_GPT_IMAGE2_MODEL_ID = project_getenv("FAL_GPT_IMAGE2_MODEL_ID", "openai/gpt-image-2/edit")
    FAL_GPT_IMAGE2_QUALITY = project_getenv("FAL_GPT_IMAGE2_QUALITY", "medium")

    # aiapiroute/Sub2API GPT-image 配置（OpenAI 兼容 Images API）
    AIAPIROUTE_API_KEY = project_getenv("AIAPIROUTE_API_KEY") or project_getenv("SUB2API_API_KEY")
    AIAPIROUTE_BASE_URL = project_getenv("AIAPIROUTE_BASE_URL") or project_getenv("SUB2API_BASE_URL", "https://aiapiroute.com")
    AIAPIROUTE_GPT_IMAGE2_MODEL = project_getenv("AIAPIROUTE_GPT_IMAGE2_MODEL", "gpt-image-2")
    AIAPIROUTE_IMAGE_RESOLUTION = project_getenv("AIAPIROUTE_IMAGE_RESOLUTION", project_getenv("AIAPIROUTE_GPT_IMAGE2_RESOLUTION", "1K"))
    AIAPIROUTE_IMAGE_QUALITY = project_getenv("AIAPIROUTE_IMAGE_QUALITY", project_getenv("AIAPIROUTE_GPT_IMAGE2_QUALITY", ""))
    # Sub2API v0.2.5 image edits are most reliable with non-streaming JSON.
    AIAPIROUTE_IMAGE_STREAM = project_getenv("AIAPIROUTE_IMAGE_STREAM", project_getenv("AIAPIROUTE_GPT_IMAGE2_STREAM", "false")).lower() == "true"
    AIAPIROUTE_TIMEOUT_SECONDS = int(project_getenv("AIAPIROUTE_TIMEOUT_SECONDS", project_getenv("SUB2API_TIMEOUT_SECONDS", "300")))

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
        "gpt-image-2_fal": {
            "enabled": FAL_GPT_IMAGE2_ENABLED,
            "label": "fal.ai API (GPT Image 2)",
            "key": "FAL_API_KEY",
            "aspect_ratios": ["match_input_image", *STANDARD_ASPECT_RATIOS],
            "sizes": AIAPIROUTE_IMAGE_SIZES,
        },
        "gpt-image-2_aiapiroute": {
            "label": "aiapiroute/Sub2API gpt-image-2",
            "key": "AIAPIROUTE_API_KEY",
            "model": AIAPIROUTE_GPT_IMAGE2_MODEL,
            "aspect_ratios": STANDARD_ASPECT_RATIOS,
            "sizes": AIAPIROUTE_IMAGE_SIZES,
        },
    }

    ALL_PROVIDERS = [
        provider for provider, config in PROVIDERS.items() if config.get("enabled", True)
    ]
    DEFAULT_PROVIDER_FALLBACK_GROUPS = {
        "gpt-image-2": ["gpt-image-2_aiapiroute", "gpt-image-2_fal"],
    }
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
    def validate_provider(cls, provider: str):
        """运行时校验指定服务商是否已配置 API Key，未配置时抛出 ValueError"""
        if provider == "gpt-image-2_fal" and not cls.FAL_GPT_IMAGE2_ENABLED:
            raise ValueError("My3dFigure Fal GPT Image 2 generation is disabled.")
        if provider not in cls.ALL_PROVIDERS:
            raise ValueError(f"不支持的服务提供商: {provider}，可选: {cls.ALL_PROVIDERS}")
        key_name = cls.PROVIDERS[provider].get("key")
        if key_name and not getattr(cls, key_name, None):
            raise ValueError(f"使用 {provider} 需要设置环境变量 {key_name}")

    @classmethod
    def resolve_provider(cls, provider: str | None, validate_config: bool = True) -> str:
        """解析本次请求使用的 provider；空值回退到 IMAGE_GENERATION_PROVIDER。"""
        provider_value = getattr(provider, "value", provider)
        effective_provider = (provider_value or cls.IMAGE_GENERATION_PROVIDER).strip()
        if validate_config:
            cls.validate_provider(effective_provider)
        elif effective_provider not in cls.ALL_PROVIDERS:
            raise ValueError(f"不支持的默认服务提供商: {effective_provider}，可选: {cls.ALL_PROVIDERS}")
        return effective_provider

    @classmethod
    def resolve_provider_chain(cls, provider: str | None) -> list[str]:
        """Return the My3dFigure-only ordered provider fallback chain."""
        primary_provider = cls.resolve_provider(provider, validate_config=False)
        chain = next(
            (
                list(providers)
                for providers in cls.DEFAULT_PROVIDER_FALLBACK_GROUPS.values()
                if primary_provider in providers
            ),
            [primary_provider],
        )
        if primary_provider in chain:
            chain = [primary_provider, *[item for item in chain if item != primary_provider]]

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
                **({"model": config["model"]} if "model" in config else {}),
            }
            for provider, config in cls.PROVIDERS.items()
            if config.get("enabled", True)
        }

    @classmethod
    def get_available_providers(cls) -> list:
        """返回所有已配置 API Key 的服务提供商列表"""
        result = []
        for p, config in cls.PROVIDERS.items():
            if not config.get("enabled", True):
                continue
            key_name = config.get("key")
            configured = bool(getattr(cls, key_name, None)) if key_name else True
            result.append({
                "provider": p,
                "label": config.get("label", p),
                "configured": configured,
                "is_default": p == cls.IMAGE_GENERATION_PROVIDER,
            })
        return result


# 目录配置
class DirectoryConfig:
    # 生产环境目录
    TASKS_DIR = "tasks/my3dfigure"  # 生产API调用的输出目录
    ASSETS_DIR = "app/projects/my3dfigure/assets"

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


# 图片格式选项
class OutputFormatEnum(str, Enum):
    png = "png"
    jpg = "jpg"

# 绘画风格选项
class ArtStyleEnum(str, Enum):

    chatgpt_bright_cartoon_adventure = "chatgpt_bright_cartoon_adventure"



    # 通用风格
    style_3d_cartoon_doll = "3d_cartoon_doll"


# 风格提示词映射
STYLE_PROMPTS = {

    ArtStyleEnum.chatgpt_bright_cartoon_adventure: "bright colorful 3D animated movie style, vibrant family-friendly adventure look, polished character render, natural human head size, realistic child body proportions, preserve normal facial proportions, expressive eyes without oversized head, glossy cinematic lighting, tropical fantasy colors, turquoise ocean, sunny blue sky, magical golden sparkles, cheerful adventure atmosphere, clean high-detail animation concept art, charming storybook fantasy, smooth detailed 3D character design, lively dynamic pose, joyful warm expression, glossy water splashes, premium animated film aesthetic, avoid chibi proportions, avoid doll-like oversized head, avoid bobblehead, ",



    # 通用风格提示词
    ArtStyleEnum.style_3d_cartoon_doll: "3D cartoon doll style, cute collectible toy figure, soft vinyl doll look, rounded facial features, glossy eyes, smooth plastic texture, charming stylized proportions, playful product render, clean detailed 3D character design, ",


}

# 初始化配置
def initialize_config():
    """初始化配置，确保必要的目录存在"""
    DirectoryConfig.ensure_directories()

def initialize_test_config():
    """初始化测试配置，确保测试目录存在"""
    DirectoryConfig.ensure_test_directories()
