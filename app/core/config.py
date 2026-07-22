#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
项目配置文件
集中管理所有配置项、枚举类型和常量
"""

import os
from enum import Enum
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 应用配置
class AppConfig:
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"
    PORT = int(os.getenv("PORT", "8001"))
    CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")
    CORS_ORIGIN_REGEX = (
        os.getenv("CORS_ORIGIN_REGEX", r"https://([a-zA-Z0-9-]+\.)*shopifypreview\.com").strip()
        or None
    )
    MAX_CONCURRENT_TASKS = int(os.getenv("MAX_CONCURRENT_TASKS", "5"))
    ADMIN_API_KEY = os.getenv("ADMIN_API_KEY", "").strip()
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
        return current_year * cls.ALGORITHM_FACTOR
# API配置
class APIConfig:
    IMAGE_GENERATION_PROVIDER = os.getenv("IMAGE_GENERATION_PROVIDER", "seedream-4_replicate").strip() or "seedream-4_replicate"

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
    FAL_API_KEY = os.getenv("FAL_API_KEY")
    if FAL_API_KEY:
        os.environ["FAL_KEY"] = FAL_API_KEY

    # aiapiroute/Sub2API GPT-image 配置（OpenAI 兼容 Images API）
    AIAPIROUTE_API_KEY = os.getenv("AIAPIROUTE_API_KEY") or os.getenv("SUB2API_API_KEY")
    AIAPIROUTE_BASE_URL = os.getenv("AIAPIROUTE_BASE_URL") or os.getenv("SUB2API_BASE_URL", "https://aiapiroute.com")
    AIAPIROUTE_GPT_IMAGE1_MODEL = os.getenv("AIAPIROUTE_GPT_IMAGE1_MODEL", "gpt-image-1")
    AIAPIROUTE_GPT_IMAGE15_MODEL = os.getenv("AIAPIROUTE_GPT_IMAGE15_MODEL", "gpt-image-1.5")
    AIAPIROUTE_GPT_IMAGE2_MODEL = os.getenv("AIAPIROUTE_GPT_IMAGE2_MODEL", "gpt-image-2")
    AIAPIROUTE_IMAGE_RESOLUTION = os.getenv("AIAPIROUTE_IMAGE_RESOLUTION", os.getenv("AIAPIROUTE_GPT_IMAGE2_RESOLUTION", "1K"))
    AIAPIROUTE_IMAGE_QUALITY = os.getenv("AIAPIROUTE_IMAGE_QUALITY", os.getenv("AIAPIROUTE_GPT_IMAGE2_QUALITY", ""))
    AIAPIROUTE_IMAGE_STREAM = os.getenv("AIAPIROUTE_IMAGE_STREAM", os.getenv("AIAPIROUTE_GPT_IMAGE2_STREAM", "true")).lower() == "true"
    AIAPIROUTE_TIMEOUT_SECONDS = int(os.getenv("AIAPIROUTE_TIMEOUT_SECONDS", os.getenv("SUB2API_TIMEOUT_SECONDS", "300")))
    
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
        "aiapiroute_gpt-image-2": {
            "label": "aiapiroute/Sub2API gpt-image-2",
            "key": "AIAPIROUTE_API_KEY",
            "model": AIAPIROUTE_GPT_IMAGE2_MODEL,
            "aspect_ratios": STANDARD_ASPECT_RATIOS,
            "sizes": AIAPIROUTE_IMAGE_SIZES,
        },
    }

    ALL_PROVIDERS = list(PROVIDERS.keys())
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
    aiapiroute_gpt_image_1 = "aiapiroute_gpt-image-1"
    aiapiroute_gpt_image_1_5 = "aiapiroute_gpt-image-1.5"
    aiapiroute_gpt_image_2 = "aiapiroute_gpt-image-2"

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
    DEFAULT_TEXT = "Paintingify"
    DEFAULT_FONT_PATH = "assets/holidayvibesfreeregular-wppxv.ttf"
    DEFAULT_FONT_SIZE = 48
    # 铺满画面的 Paintingify 文字水印颜色；最后一位是 alpha 值（0=完全透明，255=完全不透明）。
    # 100 约等于 39% 不透明度。
    DEFAULT_COLOR = (255, 255, 255, 100)
    DEFAULT_ANGLE = 30
    DEFAULT_STEP = 120
    DEFAULT_PADDING = 10
    DEFAULT_BLEND_MODE = "screen"
    DEFAULT_RESIZE_SCALE = 0.8

    @classmethod
    def get_resize_scale(cls):
        return cls.DEFAULT_RESIZE_SCALE

    ENABLE_RANDOM_PATTERNS = True
    RANDOM_ANGLE_RANGE = 25
    RANDOM_SIZE_RANGE = 0.2
    RANDOM_OPACITY_RANGE = 30
    MULTIPLE_BLEND_MODES = ["screen", "screen", "screen"]

    LOGO_PATH = "assets/logo_watermark.png"

    CORNER_LABEL_TEXT = "Purchase to get high-resolution watermark-free image by Paintingify.com"
    CORNER_LABEL_FONT_PATH = "assets/arial.ttf"
    CORNER_LABEL_FONT_SIZE = 18
    CORNER_LABEL_PADDING = 8
    CORNER_LABEL_MARGIN = 12
    CORNER_LABEL_TEXT_COLOR = (255, 255, 255, 500)
    CORNER_LABEL_BG_COLOR = None
    CORNER_LABEL_BORDER_COLOR = (255, 255, 255, 500)
    CORNER_LABEL_BORDER_WIDTH = 1
    CORNER_LABEL_CORNER_RADIUS = 6

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

# 绘画风格选项
class ArtStyleEnum(str, Enum):
    # Flux 风格（原有风格加flux前缀）
    flux_realistic = "flux_realistic"
    flux_oil_painting = "flux_oil_painting"
    flux_heavy_oil_painting = "flux_heavy_oil_painting"
    flux_watercolor = "flux_watercolor"
    flux_anime = "flux_anime"
    flux_japanese_anime = "flux_japanese_anime"
    flux_detailed_anime = "flux_detailed_anime"
    flux_sketch = "flux_sketch"
    flux_cartoon = "flux_cartoon"
    flux_pixar = "flux_pixar"
    flux_cyberpunk = "flux_cyberpunk"
    flux_fantasy = "flux_fantasy"
    flux_steampunk = "flux_steampunk"
    flux_abstract = "flux_abstract"
    flux_watercolor_1 = "flux_watercolor_1"
    flux_acrylic = "flux_acrylic"
    flux_tattoo = "flux_tattoo"
    
    # Gemini Google 风格
    gemini_realistic = "gemini_realistic"
    gemini_oil_painting = "gemini_oil_painting"
    gemini_illustration = "gemini_illustration"
    gemini_thick_paint_illustration = "gemini_thick_paint_illustration"
    gemini_thick_paint_glossy = "gemini_thick_paint_glossy"
    gemini_thick_paint_portrait = "gemini_thick_paint_portrait"
    gemini_digital_paint_illustration = "gemini_digital_paint_illustration"
    gemini_digital_paint_glossy = "gemini_digital_paint_glossy"
    gemini_acrylic = "gemini_acrylic"
    gemini_watercolor = "gemini_watercolor"

    # Seedream-4 Replicate 风格
    seedream_4_realistic = "seedream-4_realistic"

    # Seedream-4 Fal.ai 风格
    seedream_4_fal_realistic = "seedream-4_fal_realistic"

    # 通用风格
    style_3d_cartoon_doll = "3d_cartoon_doll"


# 风格提示词映射
STYLE_PROMPTS = {
    # Flux 风格提示词（原有的）
    ArtStyleEnum.flux_realistic: "",
    ArtStyleEnum.flux_oil_painting: "oil painting, artistic style, painterly, canvas texture, heavy brush strokes, impasto technique, thick paint layers, visible brushwork, traditional art, masterpiece, textured surface, bold strokes, ",
    ArtStyleEnum.flux_heavy_oil_painting: "oil painting, heavy impasto technique, extremely thick paint layers, dramatic brush strokes, textured canvas, bold visible brushwork, traditional oil painting, masterpiece, rough texture, expressive strokes, paint texture, ",
    ArtStyleEnum.flux_watercolor: "watercolor painting, soft colors, flowing brushwork, artistic illustration, hand-drawn style, ",
    ArtStyleEnum.flux_anime: "anime style, Japanese animation, cel-shaded, manga art, stylized character, vibrant colors, detailed eyes, flowing hair, anime character design, ",
    ArtStyleEnum.flux_japanese_anime: "Japanese anime style, Studio Ghibli inspired, detailed character design, expressive eyes, flowing hair, soft lighting, pastel colors, anime aesthetic, manga influence, intricate details, elaborate costume design, ",
    ArtStyleEnum.flux_detailed_anime: "detailed anime style, intricate character design, elaborate costume with fine details, ornate accessories, complex patterns, high-quality rendering, detailed facial features, sophisticated clothing design, premium anime art, ultra-detailed, masterpiece quality, ",
    ArtStyleEnum.flux_sketch: "pencil sketch, charcoal drawing, artistic sketch, hand-drawn, monochrome art, ",
    ArtStyleEnum.flux_cartoon: "cartoon style, comic book art, vibrant colors, stylized illustration, ",
    ArtStyleEnum.flux_pixar: "Pixar style, 3D animation, computer-generated imagery, smooth textures, expressive characters, high-quality rendering, digital art, animated movie style, detailed character design, ",
    ArtStyleEnum.flux_cyberpunk: "cyberpunk style, neon lights, futuristic technology, digital art, sci-fi aesthetic, glowing effects, high-tech elements, dystopian atmosphere, ",
    ArtStyleEnum.flux_fantasy: "fantasy art style, magical atmosphere, ethereal lighting, mystical elements, enchanted world, fantasy character design, magical effects, ",
    ArtStyleEnum.flux_steampunk: "steampunk style, Victorian era, mechanical elements, brass and copper, vintage technology, industrial aesthetic, retro-futuristic, ",
    ArtStyleEnum.flux_abstract: "abstract art, modern painting, artistic interpretation, creative style, ",
    ArtStyleEnum.flux_watercolor_1: "Watercolor-A watercolor-based style, characterized by transparent, blended colors and blurred edges with watermarks. ",
    ArtStyleEnum.flux_acrylic: "Acrylic-Acrylic painting with saturated, vibrant colors and a texture between oil and watercolor. ",
    ArtStyleEnum.flux_tattoo: "Tattoo Art – Traditional tattoo design style with bold black outlines, solid color fills, White background, minimal background elements, classic tattoo flash art, body art aesthetic, ink-style shading, focus on tattoo design, Sketch Color. ",
    
    # Gemini 风格提示词
    ArtStyleEnum.gemini_realistic: "",
    ArtStyleEnum.gemini_oil_painting: "oil painting style, thick brushstrokes, textured canvas, rich oil colors, classical painting technique, impasto technique, vibrant paint layers, artistic brush marks, traditional oil medium, painterly style, fine art painting, masterpiece quality, museum-worthy artwork, renaissance painting style, impressionist brushwork, expressive painting, artistic interpretation, hand-painted masterpiece, textured paint surface, visible brush strokes, traditional canvas texture, ",
    ArtStyleEnum.gemini_illustration: "digital illustration style, clean vector art, modern illustration design, cartoon illustration, stylized artwork, digital art style, contemporary illustration, graphic design art, minimalist illustration, character design, concept art style, digital painting, artistic illustration, professional illustration, creative artwork, digital concept art, illustrated poster style, graphic novel art, animation style, commercial illustration, ",
    ArtStyleEnum.gemini_thick_paint_illustration: "thick paint illustration style, heavy impasto digital painting, maintaining character likeness, rich volumetric lighting, detailed shading and highlights, semi-realistic anime style, thick paint texture with visible brush strokes, layered paint application, dimensional character art, tactile surface quality, sculptural paint work, saturated color palette, dramatic lighting effects, painterly finish with depth, textured digital artwork, thick acrylic paint style, cel-shading with volume, rendered illustration with weight, heavy paint medium digital art, chunky paint texture illustration, dimensional anime artwork, thick paint character design, ",
    ArtStyleEnum.gemini_digital_paint_illustration: "digital illustration style, smooth digital painting, strong digital highlights, glossy digital skin texture, dramatic digital lighting, high saturation digital colors, smooth digital volumetric effects, clean digital surface reflections, metallic digital rendering, sharp digital highlight details, deep digital shadow contrast, lustrous digital hair highlights, skin shine and gloss, fabric texture with digital light reflection, intense digital color saturation, smooth digital brushwork, clean digital character art with depth, smooth digital surface quality, rich digital color gradients, dramatic digital chiaroscuro lighting, reflective digital surface materials, high contrast digital shading, vibrant digital highlight effects, clean digital finish, digital art medium, ",
    ArtStyleEnum.gemini_thick_paint_glossy: "thick paint illustration style, heavy impasto digital painting, strong specular highlights, glossy skin texture, dramatic light and shadow contrast, high saturation colors, volumetric lighting effects, detailed surface reflections, metallic and wet material rendering, sharp highlight details, deep shadow contrast, lustrous hair highlights, skin shine and gloss, fabric texture with light reflection, intense color saturation, painterly thick brush strokes, dimensional character art with depth, tactile surface quality, sculptural paint application, rich color gradients, dramatic chiaroscuro lighting, reflective surface materials, high contrast shading, vibrant highlight effects, thick paint texture with visible impasto, layered paint application, rendered illustration with material depth, ",
    ArtStyleEnum.gemini_digital_paint_glossy: "digital portrait style, smooth digital painting technique, precise facial feature preservation, accurate character likeness, detailed digital facial anatomy, smooth digital texture while maintaining facial accuracy, clean digital painting application with careful attention to facial details, portrait-focused digital painting work, dimensional digital character portraiture, facial feature emphasis, authentic character representation, smooth digital work on facial features, rich digital color depth in skin tones, dramatic digital lighting that enhances facial structure, smooth digital portrait technique, digital painting medium optimized for character accuracy, detailed digital facial rendering with smooth texture, portrait-quality digital painting illustration, character-focused dimensional digital artwork, clean digital finish, digital art medium, ",
    ArtStyleEnum.gemini_thick_paint_portrait: "thick paint portrait style, heavy impasto technique, precise facial feature preservation, accurate character likeness, detailed facial anatomy, thick paint texture while maintaining facial accuracy, layered paint application with careful attention to facial details, portrait-focused thick paint work, dimensional character portraiture, facial feature emphasis, authentic character representation, sculptural paint work on facial features, rich color depth in skin tones, dramatic lighting that enhances facial structure, painterly portrait technique, thick paint medium optimized for character accuracy, detailed facial rendering with impasto texture, portrait-quality thick paint illustration, character-focused dimensional artwork, ",
    ArtStyleEnum.gemini_acrylic: "acrylic painting style, vibrant acrylic colors, bold brushstrokes, matte finish, contemporary acrylic art, bright pigments, modern acrylic technique, plastic paint medium, quick-drying paint, colorful acrylic artwork, abstract acrylic style, expressive acrylic brushwork, contemporary art style, acrylic on canvas, vivid color palette, modern painting technique, artistic acrylic expression, bold color application, acrylic paint texture, contemporary acrylic masterpiece, ",
    ArtStyleEnum.gemini_watercolor: "watercolor painting style, soft watercolor washes, translucent layers, paper texture, watercolor bloom effects, delicate brush strokes, flowing watercolor technique, transparent pigments, aquarelle style, wet-on-wet technique, watercolor gradients, soft color transitions, artistic watercolor expression, traditional watercolor method, gentle color blending, watercolor paper texture, ethereal watercolor effects, fluid paint application, watercolor transparency, dreamy watercolor atmosphere, ",
    
    # Seedream-4 风格提示词
    ArtStyleEnum.seedream_4_realistic: "",

    # Seedream-4 Fal.ai 风格提示词
    ArtStyleEnum.seedream_4_fal_realistic: "",

    # 通用风格提示词
    ArtStyleEnum.style_3d_cartoon_doll: "3D cartoon doll style, cute collectible toy figure, soft vinyl doll look, rounded facial features, glossy eyes, smooth plastic texture, charming stylized proportions, playful product render, clean detailed 3D character design, ",
    

}

# 风格描述
STYLE_DESCRIPTIONS = {
    # Flux 风格描述
    "flux_realistic": "写实风格 - 保持真实感，不添加艺术效果",
    "flux_oil_painting": "油画风格 - 模拟传统油画笔触和质感",
    "flux_heavy_oil_painting": "强烈油画风格 - 厚重的笔触和强烈的质感",
    "flux_watercolor": "水彩风格 - 柔和的色彩和流动的笔触",
    "flux_anime": "动漫风格 - 日式动漫和漫画风格",
    "flux_japanese_anime": "日本动漫风格 - 吉卜力工作室风格",
    "flux_detailed_anime": "精致动漫风格 - 高细节和精致服装设计",
    "flux_sketch": "素描风格 - 铅笔或炭笔素描效果",
    "flux_cartoon": "卡通风格 - 美式卡通和漫画风格",
    "flux_pixar": "皮克斯风格 - 3D动画电影风格",
    "flux_cyberpunk": "赛博朋克风格 - 未来科技风格",
    "flux_fantasy": "奇幻风格 - 魔法世界风格",
    "flux_steampunk": "蒸汽朋克风格 - 维多利亚机械风格",
    "flux_abstract": "抽象风格 - 现代抽象艺术风格",
    "flux_watercolor_1": "水彩画 - 水彩画风格，透明、混合颜色和模糊边缘",
    "flux_acrylic": "丙烯画 - 丙烯画风格，饱和、鲜艳的颜色和介于油和之间的纹理",
    "flux_tattoo": "纹身艺术 - 传统纹身设计风格，采用大胆的黑色轮廓、纯色填充、白色背景、极简背景元素、经典纹身闪光艺术、人体艺术美学、水墨风格阴影、专注于纹身设计、素描色彩",
    
    # Gemini 风格描述
    "gemini_realistic": "写实风格 - 保持真实感，不添加艺术效果",
    "gemini_oil_painting": "Gemini油画风格 - 丰富的油画质感，厚重笔触，传统绘画技法",
    "gemini_illustration": "Gemini插画风格 - 现代数字插画，清洁的矢量艺术风格",
    "gemini_thick_paint_illustration": "Gemini厚涂插画风格 - 重厚涂数字绘画，保持角色特征",
    "gemini_thick_paint_glossy": "Gemini厚涂高光风格 - 厚涂技法配合强烈的光影对比",
    "gemini_thick_paint_portrait": "Gemini厚涂肖像风格 - 专注面部特征的厚涂人像技法",
    "gemini_digital_paint_illustration": "Gemini数字绘画插画风格 - 光滑数字绘画，强烈高光效果",
    "gemini_digital_paint_glossy": "Gemini数字绘画高光风格 - 光滑数字绘画技法，专注面部特征",
    "gemini_acrylic": "Gemini丙烯画风格 - 鲜艳的丙烯颜色，现代绘画技法",
    "gemini_watercolor": "Gemini水彩风格 - 柔和水彩洗染效果，透明层次",
    
    # Seedream-4 风格描述
    "seedream-4_realistic": "Seedream4写实风格 - 保持真实感，不添加艺术效果",

    # Seedream-4 Fal.ai 风格描述
    "seedream-4_fal_realistic": "Seedream4 (Fal.ai) 写实风格 - 保持真实感，不添加艺术效果",

    # 通用风格描述
    "3d_cartoon_doll": "3D卡通娃娃风格 - 可爱收藏玩具质感，圆润五官和3D卡通比例",
    
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

