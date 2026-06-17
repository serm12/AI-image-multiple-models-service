#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI Image Generation API 主启动文件
"""

import uvicorn
from config import AppConfig, APIConfig
import traceback

if __name__ == "__main__":
    try:
        print(f"启动AI图像生成API服务")
        configured_providers = [p for p in APIConfig.get_available_providers() if p["configured"]]

        print("提示信息:")
        print("   - 已支持多 provider 运行，每次请求必须显式传入 provider 参数")

        if configured_providers:
            configured_labels = [p.get("label", p["provider"]) for p in configured_providers]
            print(f"   - 已配置 provider({len(configured_providers)}): {', '.join(configured_labels)}")
        else:
            print("   - 当前未检测到可用 provider，请检查 .env 中 API Key 配置")

        print("   - 可通过 GET /providers 查看配置状态")
        print("   - 可通过 GET /provider-options 查看 provider 与参数对应关系")
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

