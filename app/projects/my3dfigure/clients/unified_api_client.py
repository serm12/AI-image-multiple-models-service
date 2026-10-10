#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""My3dFigure GPT Image 2 routing with its dedicated ordered fallback."""
from app.projects.my3dfigure.core.config import APIConfig
class UnifiedAPIClient:
    """Route requests only to the retained GPT Image 2 clients."""

    def __init__(self, aiapiroute_api_key=None):
        self.aiapiroute_api_key = aiapiroute_api_key or APIConfig.AIAPIROUTE_API_KEY

        self._fal_gpt_image2_client = None
        self._aiapiroute_clients = {}

    # ---- Lazy loading for the two retained clients ----

    def _get_fal_gpt_image2_client(self):
        if self._fal_gpt_image2_client is None:
            from .fal_gpt_image2_client import FalGPTImage2Client
            self._fal_gpt_image2_client = FalGPTImage2Client()
        return self._fal_gpt_image2_client

    def _get_aiapiroute_client(self, model: str):
        if model not in self._aiapiroute_clients:
            from .aiapiroute_gpt_image2_client import AIApiRouteGPTImageClient
            self._aiapiroute_clients[model] = AIApiRouteGPTImageClient(self.aiapiroute_api_key, model=model)
        return self._aiapiroute_clients[model]

    # ---- 核心路由方法 ----

    async def generate_image(self, prompt, input_image_paths=None, input_image_url=None,
                             aspect_ratio=None, size=None, seed=None, task_id=None,
                             provider=None):
        """
        统一的图像生成接口 - 路由分发器

        Args:
            prompt: 提示词
            input_image_paths: 输入图片路径列表（本地文件）
            input_image_url: 输入图片URL
            aspect_ratio: 宽高比
            seed: 随机种子
            size: 输出尺寸
            task_id: 任务ID（保留调用兼容）
            provider: 服务提供商，可选值见 APIConfig.ALL_PROVIDERS；空值回退到 IMAGE_GENERATION_PROVIDER。

        Returns:
            dict: 生成结果
        """
        provider_chain = APIConfig.resolve_provider_chain(provider)
        errors = []
        for index, effective_provider in enumerate(provider_chain):
            try:
                result = await self._generate_image_with_provider(
                    effective_provider, prompt, input_image_paths, input_image_url,
                    aspect_ratio, size, seed,
                )
                result["api_provider"] = effective_provider
                result["provider_fallback_chain"] = provider_chain
                result["provider_fallback_attempts"] = [
                    {"provider": item, "error": error} for item, error in errors
                ]
                return result
            except Exception as exc:
                error_message = str(exc)
                errors.append((effective_provider, error_message))
                print(f"My3dFigure provider {effective_provider} generation failed: {error_message}")
                if index + 1 < len(provider_chain):
                    print(f"My3dFigure switching fallback provider: {provider_chain[index + 1]}")

        summary = "; ".join(f"{item}: {error}" for item, error in errors)
        raise ValueError(f"所有 My3dFigure 兜底 provider 均失败: {summary}")

    async def _generate_image_with_provider(self, effective_provider, prompt, input_image_paths=None,
                                            input_image_url=None, aspect_ratio=None, size=None,
                                            seed=None):
        if effective_provider == "gpt-image-2_fal":
            return await self._generate_with_fal_gpt_image2(
                prompt, input_image_paths, input_image_url, seed, aspect_ratio, size
            )
        if effective_provider in APIConfig.AIAPIROUTE_PROVIDER_MODEL_MAP:
            return await self._generate_with_aiapiroute(
                prompt, effective_provider, input_image_paths, input_image_url, seed, aspect_ratio, size
            )
        raise ValueError(f"不支持的 My3dFigure 生图服务: {effective_provider}")

    async def _generate_with_fal_gpt_image2(self, prompt, input_image_paths=None, input_image_url=None,
                                             seed=None, aspect_ratio=None, size=None):
        """Generate through the My3dFigure-local GPT Image 2 Fal fallback client."""
        try:
            client = self._get_fal_gpt_image2_client()
            return await client.generate_image(
                prompt=prompt,
                input_image_paths=input_image_paths,
                input_image_url=input_image_url,
                seed=seed,
                aspect_ratio=aspect_ratio,
                size=size,
            )
        except Exception as e:
            raise ValueError(f"GPT Image 2 Fal.ai API调用失败: {e}")

    async def _generate_with_aiapiroute(self, prompt, provider, input_image_paths=None, input_image_url=None, seed=None, aspect_ratio=None, size=None):
        """使用 aiapiroute/Sub2API GPT-image 系列模型生成图像"""
        try:
            model = APIConfig.AIAPIROUTE_PROVIDER_MODEL_MAP[provider]
            client = self._get_aiapiroute_client(model)
            return await client.generate_image(
                prompt=prompt,
                input_image_paths=input_image_paths,
                input_image_url=input_image_url,
                seed=seed,
                aspect_ratio=aspect_ratio,
                size=size,
            )
        except Exception as e:
            raise ValueError(f"aiapiroute {provider} API调用失败: {e}")


# 全局单例
api_client = UnifiedAPIClient()
