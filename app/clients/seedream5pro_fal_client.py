"""Fal client for ByteDance Seedream 5.0 Pro image editing."""

import asyncio
import os
from datetime import datetime

import fal_client

from app.core.config import APIConfig


class Seedream5ProFalClient:
    """Generate reference-guided images with ``bytedance/seedream/v5/pro/edit``."""

    _IMAGE_SIZES = {
        "1:1": {"width": 1536, "height": 1536},
        "4:3": {"width": 2048, "height": 1536},
        "3:4": {"width": 1536, "height": 2048},
        "16:9": {"width": 2048, "height": 1152},
        "9:16": {"width": 1152, "height": 2048},
        "match_input_image": {"width": 1536, "height": 2048},
    }

    def __init__(self, api_key=None, model_id=None):
        self.api_key = api_key or APIConfig.FAL_API_KEY
        if not self.api_key:
            raise ValueError("FAL_API_KEY not found in environment variables")
        os.environ["FAL_KEY"] = self.api_key
        self.model_id = model_id or APIConfig.FAL_SEEDREAM5_PRO_MODEL_ID

    @classmethod
    def image_size_for_ratio(cls, aspect_ratio):
        ratio = getattr(aspect_ratio, "value", aspect_ratio) or "3:4"
        return dict(cls._IMAGE_SIZES.get(str(ratio), cls._IMAGE_SIZES["3:4"]))

    async def _upload_local_image(self, image_path: str) -> str:
        if not os.path.exists(image_path):
            raise ValueError(f"Image file not found: {image_path}")
        try:
            if hasattr(fal_client, "upload_file"):
                return await asyncio.to_thread(fal_client.upload_file, image_path)
            if hasattr(fal_client, "upload"):
                return await asyncio.to_thread(fal_client.upload, image_path)
            raise ImportError("fal_client module has no upload function")
        except Exception as exc:
            raise ValueError(f"Failed to upload reference image to Fal: {exc}") from exc

    async def generate_image(self, prompt, input_image_paths=None, seed=None, art_style=None, aspect_ratio="3:4", **_kwargs):
        if not input_image_paths:
            raise ValueError("Seedream 5 Pro Fal model requires at least one reference image.")

        image_urls = []
        for image_path in input_image_paths:
            if isinstance(image_path, str) and image_path.startswith(("http://", "https://", "data:")):
                image_urls.append(image_path)
            else:
                image_urls.append(await self._upload_local_image(str(image_path)))

        arguments = {
            "prompt": prompt,
            "image_urls": image_urls,
            "image_size": self.image_size_for_ratio(aspect_ratio),
            "num_images": 1,
            "output_format": "png",
            "enable_safety_checker": True,
        }
        # Seedream 5 Pro has no seed input; retain it in task metadata only.
        try:
            handler = fal_client.submit(self.model_id, arguments)
        except TypeError:
            handler = fal_client.submit(self.model_id, **arguments)

        result = await asyncio.get_running_loop().run_in_executor(None, handler.get)
        images = result.get("images", []) if isinstance(result, dict) else []
        if not images or not images[0].get("url"):
            raise ValueError("Seedream 5 Pro Fal API did not return an image URL")

        return {
            "id": f"seedream5pro_fal_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{seed or 'na'}",
            "status": "succeeded",
            "output": images[0]["url"],
            "logs": "Seedream 5 Pro Fal.ai generation completed",
            "input": {
                "prompt": prompt,
                "image_size": arguments["image_size"],
                "reference_image_count": len(image_urls),
            },
            "api_type": "seedream5pro_fal",
            "extracted_seed": seed,
        }
