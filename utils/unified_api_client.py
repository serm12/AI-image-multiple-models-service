#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一API客户端
支持Flux和Gemini API的路由分发，支持 per-request provider 动态切换
"""
import asyncio
from config import APIConfig


class UnifiedAPIClient:
    """统一API客户端，负责路由分发到具体的API客户端。

    支持 per-request provider 覆盖：在调用 generate_image() 时传入 provider 参数，
    可在不重启服务的情况下切换到任意已配置的服务商。
    """

    def __init__(self, bfl_api_key=None, replicate_token=None, gemini_api_key=None, openrouter_api_key=None, aiapiroute_api_key=None):
        # 默认 provider（来自环境变量），可被每次请求的 provider 参数覆盖
        self.default_provider = APIConfig.IMAGE_GENERATION_PROVIDER
        self.bfl_api_key = bfl_api_key or APIConfig.BFL_API_KEY
        self.replicate_token = replicate_token or APIConfig.REPLICATE_API_TOKEN
        self.gemini_api_key = gemini_api_key or APIConfig.GOOGLE_GEMINI_API_KEY
        self.openrouter_api_key = openrouter_api_key or APIConfig.OPENROUTER_API_KEY
        self.aiapiroute_api_key = aiapiroute_api_key or APIConfig.AIAPIROUTE_API_KEY

        # 延迟加载客户端缓存（不再锁定到单一 provider）
        self._flux_client = None
        self._gemini_client = None
        self._gemini_replicate_client = None
        self._openrouter_client = None
        self._seedream_4_replicate_client = None
        self._seedream_4_fal_client = None
        self._aiapiroute_clients = {}

    # ---- 向后兼容属性 ----
    @property
    def api_provider(self):
        return self.default_provider

    # ---- 延迟加载各客户端（按需初始化，不再依赖 provider 条件） ----

    def _get_flux_client(self):
        if self._flux_client is None:
            from .flux_client import FluxClient
            self._flux_client = FluxClient(self.bfl_api_key, self.replicate_token, APIConfig.FIREWORKS_API_KEY)
        return self._flux_client

    def _get_gemini_client(self):
        if self._gemini_client is None:
            from .gemini_client import GeminiClient
            self._gemini_client = GeminiClient(self.gemini_api_key)
        return self._gemini_client

    def _get_gemini_replicate_client(self):
        if self._gemini_replicate_client is None:
            from .gemini_replicate_client import GeminiReplicateClient
            self._gemini_replicate_client = GeminiReplicateClient(self.replicate_token)
        return self._gemini_replicate_client

    def _get_gemini_openrouter_client(self):
        if self._openrouter_client is None:
            from .gemini_openrouter_client import GeminiOpenRouterClient
            self._openrouter_client = GeminiOpenRouterClient(self.openrouter_api_key)
        return self._openrouter_client

    def _get_seedream_4_replicate_client(self):
        if self._seedream_4_replicate_client is None:
            from .seedream4_replicate_client import Seedream4ReplicateClient
            self._seedream_4_replicate_client = Seedream4ReplicateClient(self.replicate_token)
        return self._seedream_4_replicate_client

    def _get_seedream_4_fal_client(self):
        if self._seedream_4_fal_client is None:
            from .seedream4_fal_client import Seedream4FalClient
            self._seedream_4_fal_client = Seedream4FalClient()
        return self._seedream_4_fal_client

    def _get_aiapiroute_client(self, model: str):
        if model not in self._aiapiroute_clients:
            from .aiapiroute_gpt_image2_client import AIApiRouteGPTImage2Client
            self._aiapiroute_clients[model] = AIApiRouteGPTImage2Client(self.aiapiroute_api_key, model=model)
        return self._aiapiroute_clients[model]

    # ---- 核心路由方法 ----

    async def generate_image(self, prompt, input_image_paths=None, input_image_url=None,
                             flux_model_variant=None, aspect_ratio=None, output_format=None,
                             seed=None, art_style=None, width=None, height=None, size=None,
                             sequential_image_generation=None, task_id=None, provider=None):
        """
        统一的图像生成接口 - 路由分发器

        Args:
            prompt: 提示词
            input_image_paths: 输入图片路径列表（本地文件）
            input_image_url: 输入图片URL
            flux_model_variant: Flux模型变体（Gemini忽略此参数）
            aspect_ratio: 宽高比
            output_format: 输出格式
            seed: 随机种子
            art_style: 艺术风格
            width / height / size: Seedream 尺寸参数
            sequential_image_generation: Seedream 顺序生成参数
            task_id: 任务ID（OpenRouter 需要）
            provider: 覆盖默认服务提供商，可选值见 APIConfig.ALL_PROVIDERS。
                      不传则使用 .env 中的 IMAGE_GENERATION_PROVIDER。

        Returns:
            dict: 生成结果
        """
        # 确定本次请求实际使用的 provider，并做运行时校验
        effective_provider = provider or self.default_provider
        APIConfig.validate_provider(effective_provider)

        if effective_provider == "gemini-nanobanana_google":
            return await self._generate_with_gemini(
                prompt, input_image_paths, input_image_url, seed, art_style
            )
        elif effective_provider == "gemini-nanobanana_replicate":
            return await self._generate_with_gemini_replicate(
                prompt, input_image_paths, input_image_url, seed, art_style
            )
        elif effective_provider == "gemini-nanobanana_openrouter":
            return await self._generate_with_gemini_openrouter(
                prompt, input_image_paths, input_image_url, seed, art_style, task_id
            )
        elif effective_provider == "seedream-4_replicate":
            return await self._generate_with_seedream_4_replicate(
                prompt, input_image_paths, seed, art_style, width, height, size, sequential_image_generation
            )
        elif effective_provider == "seedream-4_fal":
            return await self._generate_with_seedream_4_fal(
                prompt, input_image_paths, seed, art_style, aspect_ratio
            )
        elif effective_provider in APIConfig.AIAPIROUTE_PROVIDER_MODEL_MAP:
            return await self._generate_with_aiapiroute(
                prompt, effective_provider, input_image_paths, input_image_url, seed, aspect_ratio, size
            )
        elif effective_provider in ["flux_bfl", "flux_replicate", "flux_fireworks"]:
            flux_client = self._get_flux_client()
            main_input_image_path = input_image_paths[0] if input_image_paths else None
            return await flux_client.generate_image(
                effective_provider, prompt, main_input_image_path, input_image_url,
                flux_model_variant, aspect_ratio, output_format, seed
            )
        else:
            raise ValueError(f"不支持的API提供商: {effective_provider}")

    # ---- 各服务商具体实现 ----

    async def _generate_with_gemini(self, prompt, input_image_paths=None, input_image_url=None, seed=None, art_style=None):
        """使用Gemini API生成图像"""
        try:
            gemini_client = self._get_gemini_client()
            if not art_style:
                art_style = "gemini_oil_painting"
            result = await gemini_client.generate_image(
                prompt=prompt,
                input_image_paths=input_image_paths,
                input_image_urls=[input_image_url] if input_image_url else None,
                art_style=art_style,
                seed=seed
            )
            return result
        except Exception as e:
            raise ValueError(f"Gemini API调用失败: {e}")

    async def _generate_with_gemini_replicate(self, prompt, input_image_paths=None, input_image_url=None, seed=None, art_style=None):
        """使用Gemini Replicate API生成图像"""
        try:
            gemini_replicate_client = self._get_gemini_replicate_client()
            if not art_style:
                art_style = "gemini_replicate_oil_painting"
            result = await gemini_replicate_client.generate_image(
                prompt=prompt,
                input_image_paths=input_image_paths,
                input_image_urls=[input_image_url] if input_image_url else None,
                art_style=art_style,
                seed=seed
            )
            return result
        except Exception as e:
            raise ValueError(f"Gemini Replicate API调用失败: {e}")

    async def _generate_with_gemini_openrouter(self, prompt, input_image_paths=None, input_image_url=None, seed=None, art_style=None, task_id=None):
        """使用OpenRouter API生成图像"""
        try:
            openrouter_client = self._get_gemini_openrouter_client()
            if not art_style:
                art_style = "gemini_realistic"
            reference_images = []
            if input_image_paths:
                reference_images.extend(input_image_paths)
            if input_image_url:
                reference_images.append(input_image_url)
            result = await openrouter_client.generate_image(
                prompt=prompt,
                reference_images=reference_images,
                style=art_style,
                seed=seed
            )
            if result and result.get("status") == "succeeded":
                return result
            else:
                error_details = result.get("raw", "No additional details") if result else "No response"
                raise ValueError(f"OpenRouter image generation failed. Details: {error_details}")
        except Exception as e:
            raise ValueError(f"OpenRouter API调用失败: {e}")

    async def _generate_with_seedream_4_replicate(self, prompt, input_image_paths=None, seed=None, art_style=None, width=None, height=None, size=None, sequential_image_generation=None):
        """使用Seedream4 Replicate API生成图像"""
        try:
            client = self._get_seedream_4_replicate_client()
            if not art_style:
                art_style = "seedream-4_realistic"
            result = await client.generate_image(
                prompt=prompt,
                input_image_paths=input_image_paths,
                art_style=art_style,
                seed=seed,
                width=width,
                height=height,
                size=size,
                sequential_image_generation=sequential_image_generation
            )
            return result
        except Exception as e:
            raise ValueError(f"Seedream4 Replicate API调用失败: {e}")

    async def _generate_with_seedream_4_fal(self, prompt, input_image_paths=None, seed=None, art_style=None, aspect_ratio=None):
        """使用Seedream4 Fal.ai API生成图像"""
        try:
            client = self._get_seedream_4_fal_client()
            if not art_style:
                art_style = "seedream-4_fal_realistic"
            result = await client.generate_image(
                prompt=prompt,
                input_image_paths=input_image_paths,
                art_style=art_style,
                seed=seed,
                aspect_ratio=aspect_ratio
            )
            return result
        except Exception as e:
            raise ValueError(f"Seedream4 Fal.ai API调用失败: {e}")

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

