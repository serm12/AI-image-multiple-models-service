"""
fal.ai GPT Image 2 client.

Paintingify uses this provider mainly for image-to-image portrait previews.
The output size can be explicit, based on the requested product aspect ratio,
or left as "auto" / "match_input_image" when the caller wants the model to
follow the reference image.
"""

import asyncio
import base64
import mimetypes
import os
from datetime import datetime
from typing import Any

import fal_client

from app.core.config import APIConfig


class FalGPTImage2Client:
    """GPT Image 2 via fal.ai."""

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or APIConfig.FAL_API_KEY
        if not self.api_key:
            raise ValueError("FAL_API_KEY not found in environment variables")

        os.environ["FAL_KEY"] = self.api_key
        self.model_id = APIConfig.FAL_GPT_IMAGE2_MODEL_ID or "openai/gpt-image-2/edit"
        self.quality = APIConfig.FAL_GPT_IMAGE2_QUALITY or "medium"

    async def _upload_local_image(self, image_path: str) -> str:
        if not os.path.exists(image_path):
            raise ValueError(f"Image file not found: {image_path}")

        try:
            if hasattr(fal_client, "upload_file"):
                return fal_client.upload_file(image_path)
            if hasattr(fal_client, "upload"):
                return fal_client.upload(image_path)
            raise ImportError("fal_client module has no upload function")
        except Exception:
            mime_type, _ = mimetypes.guess_type(image_path)
            if not mime_type:
                mime_type = "image/jpeg"
            with open(image_path, "rb") as file:
                b64_img = base64.b64encode(file.read()).decode("utf-8")
            return f"data:{mime_type};base64,{b64_img}"

    async def _prepare_image_urls(
        self,
        input_image_paths: list[str] | None = None,
        input_image_url: str | None = None,
    ) -> list[str]:
        urls: list[str] = []

        for image_path in input_image_paths or []:
            if isinstance(image_path, str) and os.path.exists(image_path):
                urls.append(await self._upload_local_image(image_path))
            else:
                urls.append(image_path)

        if input_image_url:
            urls.append(input_image_url)

        return [url for url in urls if url]

    def _resolve_image_size(self, aspect_ratio: Any = None, size: str | None = None) -> str | dict[str, int]:
        ratio_value = getattr(aspect_ratio, "value", aspect_ratio) or "3:4"
        size_value = (size or "2K").strip().upper()

        if ratio_value == "match_input_image" or size_value == "AUTO":
            return "auto"

        if ":" not in str(ratio_value):
            ratio_value = "3:4"

        try:
            width_ratio, height_ratio = [int(part.strip()) for part in str(ratio_value).split(":", 1)]
        except (TypeError, ValueError):
            width_ratio, height_ratio = 3, 4

        long_edge_map = {
            "1K": 1024,
            "2K": 2048,
            "4K": 4096,
        }
        long_edge = long_edge_map.get(size_value, 2048)

        if width_ratio >= height_ratio:
            width = long_edge
            height = max(1, round(long_edge * height_ratio / width_ratio))
        else:
            height = long_edge
            width = max(1, round(long_edge * width_ratio / height_ratio))

        return {"width": width, "height": height}

    async def generate_image(
        self,
        prompt: str,
        input_image_paths: list[str] | None = None,
        input_image_url: str | None = None,
        seed: int | None = None,
        aspect_ratio: Any = None,
        size: str | None = None,
        quality: str | None = None,
        **kwargs,
    ) -> dict:
        image_urls = await self._prepare_image_urls(input_image_paths, input_image_url)
        if not image_urls:
            raise ValueError("GPT Image 2 Fal.ai 模型需要至少一个参考图像。")

        image_size = self._resolve_image_size(aspect_ratio=aspect_ratio, size=size)
        request_quality = (quality or self.quality or "medium").strip()

        arguments = {
            "prompt": prompt,
            "image_urls": image_urls,
            "image_size": image_size,
            "quality": request_quality,
            "num_images": 1,
        }
        if seed is not None:
            arguments["seed"] = seed

        print(
            "GPT Image 2 Fal.ai 生成图像: "
            f"model_id={self.model_id}, aspect_ratio={getattr(aspect_ratio, 'value', aspect_ratio)}, "
            f"size={size}, image_size={image_size}, quality={request_quality}, refs={len(image_urls)}"
        )

        try:
            handler = fal_client.submit(self.model_id, arguments)
        except TypeError:
            handler = fal_client.submit(self.model_id, **arguments)

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, handler.get)

        if not isinstance(result, dict) or not result.get("images"):
            raise ValueError(f"Fal.ai GPT Image 2 API 未返回有效图像: {type(result)}")

        image = result["images"][0]
        image_url = image.get("url") if isinstance(image, dict) else image
        if not image_url:
            raise ValueError("Fal.ai GPT Image 2 API 响应缺少图片 URL")

        prediction_id = f"gpt_image2_fal_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{seed or 'na'}"

        return {
            "id": prediction_id,
            "status": "succeeded",
            "output": image_url,
            "logs": f"GPT Image 2 Fal.ai 生成 - 种子: {seed}, 尺寸: {image_size}, 质量: {request_quality}",
            "input": {
                "prompt": prompt,
                "image_size": image_size,
                "quality": request_quality,
                "seed": seed,
                "model_id": self.model_id,
            },
            "api_type": "gpt_image2_fal",
            "raw": {
                "seed": result.get("seed"),
                "images_count": len(result.get("images", [])),
            },
            "extracted_seed": result.get("seed") or seed,
        }
