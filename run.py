#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI Image Generation API 主启动文件
"""

import os
import uvicorn
from config import AppConfig, APIConfig
import traceback

if __name__ == "__main__":
    try:
        print(f"启动AI图像生成API服务")
        # 检查配置
        # 检查环境变量
        print(f"当前API提供商: {APIConfig.IMAGE_GENERATION_PROVIDER}")
        
        # 根据API服务提供商打印不同的信息
        print("提示信息:")
        if APIConfig.IMAGE_GENERATION_PROVIDER == "flux_bfl":
            print("   - 使用BFL API (Flux模型) 进行图像生成")
        elif APIConfig.IMAGE_GENERATION_PROVIDER == "flux_replicate":
            print("   - 使用Replicate API (Flux模型) 进行图像生成")
        elif APIConfig.IMAGE_GENERATION_PROVIDER == "gemini_google":
            print("   - 使用Google Gemini API进行图像生成")
        elif APIConfig.IMAGE_GENERATION_PROVIDER == "gemini_replicate":
            print("   - 使用Replicate Gemini API进行图像生成")
        elif APIConfig.IMAGE_GENERATION_PROVIDER == "gemini_openrouter":
            print("   - 使用OpenRouter Gemini API进行图像生成")
            print("   - 模型: Google Gemini 2.5 Flash Image Preview")
            print("   - 平台: OpenRouter (https://openrouter.ai)")
        elif APIConfig.IMAGE_GENERATION_PROVIDER == "seedream-4_replicate":
            print("   - 使用Replicate API (Seedream-4模型) 进行图像生成")
        elif APIConfig.IMAGE_GENERATION_PROVIDER == "seedream-4_fal":
            print("   - 使用fal.ai API (Seedream-4模型) 进行图像生成")
        
        print("=" * 50)
        
        uvicorn.run(
            "ai_image_api:app",
            host="0.0.0.0",
            port=int(os.environ.get("PORT", 8000)),
            reload=AppConfig.DEBUG,
            log_level="info" if not AppConfig.DEBUG else "debug"
        )
    except Exception as e:
        print("Error during startup:")
        print(traceback.format_exc())
