"""My3dFigure-local Fal GPT Image 2 fallback client."""

import asyncio
import base64
import mimetypes
import os
from datetime import datetime
from typing import Any

import fal_client

from app.projects.my3dfigure.core.config import APIConfig


class FalGPTImage2Client:
    """Submit the validated My3dFigure upload to Fal GPT Image 2 without reprocessing it."""

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or APIConfig.FAL_API_KEY
        if not self.api_key:
            raise ValueError("FAL_API_KEY not found in environment variables")
        self.model_id = APIConfig.FAL_GPT_IMAGE2_MODEL_ID
        self.quality = APIConfig.FAL_GPT_IMAGE2_QUALITY

    async def _upload_local_image(self, image_path: str) -> str:
        if not os.path.exists(image_path):
            raise ValueError(f"Image file not found: {image_path}")
        try:
            return await asyncio.to_thread(fal_client.upload_file, image_path)
        except Exception:
            mime_type, _ = mimetypes.guess_type(image_path)
            mime_type = mime_type or "image/jpeg"
            with open(image_path, "rb") as image_file:
                encoded = base64.b64encode(image_file.read()).decode("utf-8")
            return f"data:{mime_type};base64,{encoded}"

    async def _prepare_image_urls(self, input_image_paths=None, input_image_url=None) -> list[str]:
        urls = []
        for image_path in input_image_paths or []:
            urls.append(await self._upload_local_image(image_path) if os.path.exists(image_path) else image_path)
        if input_image_url:
            urls.append(input_image_url)
        return [url for url in urls if url]

    @staticmethod
    def _resolve_image_size(aspect_ratio: Any = None, size: str | None = None):
        ratio_value = getattr(aspect_ratio, "value", aspect_ratio) or "2:3"
        size_value = (size or "1K").strip().upper()
        if ratio_value == "match_input_image" or size_value == "AUTO":
            return "auto"
        try:
            width_ratio, height_ratio = [int(part.strip()) for part in str(ratio_value).split(":", 1)]
        except (TypeError, ValueError):
            width_ratio, height_ratio = 2, 3
        # GPT Image's native resolution tiers use the short edge.  Keeping the
        # same convention here makes a 2:3 1K Fal fallback match the primary
        # provider's actual 1024x1536 output instead of shrinking it to 683x1024.
        short_edge = {"1K": 1024, "2K": 2048, "4K": 4096}.get(size_value, 1024)
        if width_ratio >= height_ratio:
            return {"width": max(1, round(short_edge * width_ratio / height_ratio)), "height": short_edge}
        return {"width": short_edge, "height": max(1, round(short_edge * height_ratio / width_ratio))}

    async def generate_image(self, prompt: str, input_image_paths=None, input_image_url=None,
                             seed=None, aspect_ratio=None, size=None, **kwargs) -> dict:
        image_urls = await self._prepare_image_urls(input_image_paths, input_image_url)
        if not image_urls:
            raise ValueError("GPT Image 2 Fal.ai 模型需要至少一个参考图像。")
        image_size = self._resolve_image_size(aspect_ratio, size)
        arguments = {
            "prompt": prompt,
            "image_urls": image_urls,
            "image_size": image_size,
            "quality": self.quality,
            "num_images": 1,
            "output_format": "png",
        }
        if seed is not None:
            arguments["seed"] = seed
        handler = await asyncio.to_thread(fal_client.submit, self.model_id, arguments)
        result = await asyncio.to_thread(handler.get)
        if not isinstance(result, dict) or not result.get("images"):
            raise ValueError("Fal.ai GPT Image 2 API 未返回有效图像")
        image = result["images"][0]
        image_url = image.get("url") if isinstance(image, dict) else image
        if not image_url:
            raise ValueError("Fal.ai GPT Image 2 API 响应缺少图片 URL")
        return {
            "id": f"my3d_gpt_image2_fal_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{seed or 'na'}",
            "status": "succeeded",
            "output": image_url,
            "logs": f"My3dFigure GPT Image 2 Fal.ai fallback; image_size={image_size}; quality={self.quality}",
            "input": {"prompt": prompt, "image_size": image_size, "quality": self.quality, "seed": seed, "model_id": self.model_id},
            "api_type": "my3d_gpt_image2_fal",
            "raw": {"images_count": len(result["images"])},
            "extracted_seed": result.get("seed") or seed,
        }
