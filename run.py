#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI Image Generation API 主启动文件
"""

import os
import uvicorn
from config import AppConfig, APIConfig
import traceback


PROVIDER_LABELS = {
    "flux_bfl": "BFL API (Flux)",
    "flux_replicate": "Replicate API (Flux)",
    "flux_fireworks": "Fireworks API (Flux)",
    "gemini-nanobanana_google": "Google Gemini API (Nano-Banana / gemini-2.5-flash-image-preview)",
    "gemini-nanobanana_replicate": "Replicate API (Nano-Banana / google/nano-banana)",
    "gemini-nanobanana_openrouter": "OpenRouter API (Nano-Banana / gemini-2.5-flash-image-preview)",
    "seedream-4_replicate": "Replicate API (Seedream-4)",
    "seedream-4_fal": "fal.ai API (Seedream-4)",
    "aiapiroute_gpt-image-1": "aiapiroute/Sub2API gpt-image-1",
    "aiapiroute_gpt-image-1.5": "aiapiroute/Sub2API gpt-image-1.5",
    "aiapiroute_gpt-image-2": "aiapiroute/Sub2API gpt-image-2",
}

if __name__ == "__main__":
    try:
        print(f"启动AI图像生成API服务")
        default_provider = APIConfig.IMAGE_GENERATION_PROVIDER
        configured_providers = [
            p["provider"] for p in APIConfig.get_available_providers() if p["configured"]
        ]

        print(f"默认API提供商: {default_provider}")
        print("提示信息:")
        print("   - 已支持多 provider 运行，可在每次请求中覆盖 provider 参数")
        print(f"   - 默认 provider: {PROVIDER_LABELS.get(default_provider, default_provider)}")

        if configured_providers:
            configured_labels = [PROVIDER_LABELS.get(p, p) for p in configured_providers]
            print(f"   - 已配置 provider({len(configured_providers)}): {', '.join(configured_labels)}")
        else:
            print("   - 当前未检测到可用 provider，请检查 .env 中 API Key 配置")

        print("   - 可通过 GET /providers 查看配置状态")
        print("   - 可通过 POST /generate-async/ 的 provider 参数按请求指定通道")
        
        print("=" * 50)
        
        uvicorn.run(
            "ai_image_api:app",
            host="0.0.0.0",
            port=AppConfig.PORT,
            reload=AppConfig.DEBUG,
            log_level="info" if not AppConfig.DEBUG else "debug"
        )
    except Exception as e:
        print("Error during startup:")
        print(traceback.format_exc())

