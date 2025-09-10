#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一API客户端
支持Flux和Gemini API的路由分发
"""
import asyncio
from config import APIConfig

class UnifiedAPIClient:
    """统一API客户端，负责路由分发到具体的API客户端"""
    
    def __init__(self, bfl_api_key=None, replicate_token=None, gemini_api_key=None, openrouter_api_key=None):
        self.api_provider = APIConfig.IMAGE_GENERATION_PROVIDER
        self.bfl_api_key = bfl_api_key or APIConfig.BFL_API_KEY
        self.replicate_token = replicate_token or APIConfig.REPLICATE_API_TOKEN
        self.gemini_api_key = gemini_api_key or APIConfig.GOOGLE_GEMINI_API_KEY
        self.openrouter_api_key = openrouter_api_key or APIConfig.OPENROUTER_API_KEY
        
        # 延迟加载客户端
        self._flux_client = None
        self._gemini_client = None
        self._gemini_replicate_client = None
        self._openrouter_client = None
        self._seedream_4_replicate_client = None
        self._seedream_4_fal_client = None
    
    def _get_flux_client(self):
        """延迟加载Flux客户端"""
        if self._flux_client is None:
            from .flux_client import FluxClient
            self._flux_client = FluxClient(self.bfl_api_key, self.replicate_token, APIConfig.FIREWORKS_API_KEY)
        return self._flux_client
    
    def _get_gemini_client(self):
        """延迟加载Gemini客户端"""
        if self._gemini_client is None and self.api_provider == "gemini_google":
            from .gemini_client import GeminiClient
            self._gemini_client = GeminiClient(self.gemini_api_key)
        return self._gemini_client
    
    def _get_gemini_replicate_client(self):
        """延迟加载Gemini Replicate客户端"""
        if self._gemini_replicate_client is None and self.api_provider == "gemini_replicate":
            from .gemini_replicate_client import GeminiReplicateClient
            self._gemini_replicate_client = GeminiReplicateClient(self.replicate_token)
        return self._gemini_replicate_client
    
    def _get_gemini_openrouter_client(self):
        """延迟加载Gemini OpenRouter客户端"""
        if self._openrouter_client is None and self.api_provider == "gemini_openrouter":
            from .gemini_openrouter_client import GeminiOpenRouterClient
            self._openrouter_client = GeminiOpenRouterClient(self.openrouter_api_key)
        return self._openrouter_client

    def _get_seedream_4_replicate_client(self):
        """延迟加载Seedream4 Replicate客户端"""
        if self._seedream_4_replicate_client is None and self.api_provider == "seedream-4_replicate":
            from .seedream4_replicate_client import Seedream4ReplicateClient
            self._seedream_4_replicate_client = Seedream4ReplicateClient(self.replicate_token)
        return self._seedream_4_replicate_client

    def _get_seedream_4_fal_client(self):
        """延迟加载Seedream4 Fal.ai客户端"""
        if self._seedream_4_fal_client is None and self.api_provider == "seedream-4_fal":
            from .seedream4_fal_client import Seedream4FalClient
            self._seedream_4_fal_client = Seedream4FalClient()
        return self._seedream_4_fal_client
        
    async def generate_image(self, prompt, input_image_paths=None, input_image_url=None, 
                           flux_model_variant=None, aspect_ratio=None, output_format=None, seed=None, art_style=None, 
                           width=None, height=None, size=None, sequential_image_generation=None, task_id=None):
        """
        统一的图像生成接口 - 路由分发器
        
        Args:
            prompt: 提示词
            input_image_paths: 输入图片路径列表（本地文件）
            input_image_url: 输入图片URL
            flux_model_variant: Flux模型变体（Gemini忽略此参数）
            aspect_ratio: 宽高比（Gemini忽略此参数）
            output_format: 输出格式（Gemini忽略此参数）
            seed: 随机种子
            art_style: 艺术风格
            
        Returns:
            dict: 生成结果
        """
        if self.api_provider == "gemini_google":
            # 路由到Gemini客户端
            return await self._generate_with_gemini(
                prompt, input_image_paths, input_image_url, seed, art_style
            )
        elif self.api_provider == "gemini_replicate":
            # 路由到Gemini Replicate客户端
            return await self._generate_with_gemini_replicate(
                prompt, input_image_paths, input_image_url, seed, art_style
            )
        elif self.api_provider == "gemini_openrouter":
            # 路由到OpenRouter客户端
            return await self._generate_with_gemini_openrouter(
                prompt, input_image_paths, input_image_url, seed, art_style, task_id
            )
        elif self.api_provider == "seedream-4_replicate":
            # 路由到Seedream4 Replicate客户端
            return await self._generate_with_seedream_4_replicate(
                prompt, input_image_paths, seed, art_style, width, height, size, sequential_image_generation
            )
        elif self.api_provider == "seedream-4_fal":
            # 路由到Seedream4 Fal.ai客户端
            return await self._generate_with_seedream_4_fal(
                prompt, input_image_paths, seed, art_style, aspect_ratio
            )
        elif self.api_provider in ["flux_bfl", "flux_replicate", "flux_fireworks"]:
            # 路由到Flux客户端
            flux_client = self._get_flux_client()
            # Flux 模型通常只接受一张参考图，我们取第一张
            main_input_image_path = input_image_paths[0] if input_image_paths else None
            return await flux_client.generate_image(
                self.api_provider, prompt, main_input_image_path, input_image_url,
                flux_model_variant, aspect_ratio, output_format, seed
            )
        else:
            raise ValueError(f"不支持的API提供商: {self.api_provider}")
    
    async def _generate_with_gemini(self, prompt, input_image_paths=None, input_image_url=None, seed=None, art_style=None):
        """使用Gemini API生成图像"""
        try:
            gemini_client = self._get_gemini_client()
            if not gemini_client:
                raise ValueError("Gemini客户端未正确初始化")
            
            # 如果没有提供艺术风格，使用默认的Gemini风格
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
            if not gemini_replicate_client:
                raise ValueError("Gemini Replicate客户端未正确初始化")
            
            # 如果没有提供艺术风格，使用默认的Gemini Replicate风格
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
            if not openrouter_client:
                raise ValueError("OpenRouter客户端未正确初始化")
            
            # 如果没有提供艺术风格，使用默认的Gemini风格
            if not art_style:
                art_style = "gemini_realistic"
            
            # 准备参考图像列表
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
                # 记录更详细的错误信息
                error_details = result.get("raw", "No additional details") if result else "No response"
                raise ValueError(f"OpenRouter image generation failed. Details: {error_details}")
                
        except Exception as e:
            raise ValueError(f"OpenRouter API调用失败: {e}")

    async def _generate_with_seedream_4_replicate(self, prompt, input_image_paths=None, seed=None, art_style=None, width=None, height=None, size=None, sequential_image_generation=None):
        """使用Seedream4 Replicate API生成图像"""
        try:
            client = self._get_seedream_4_replicate_client()
            if not client:
                raise ValueError("Seedream4 Replicate客户端未正确初始化")
            
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
            if not client:
                raise ValueError("Seedream4 Fal.ai客户端未正确初始化")
            
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

# 导出实例
api_client = UnifiedAPIClient()