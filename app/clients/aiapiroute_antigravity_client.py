"""Generate images through AIAPIROUTE's native Antigravity Gemini endpoint."""

import asyncio
import base64
import mimetypes
from pathlib import Path
from uuid import uuid4

import httpx

from app.core.config import APIConfig


async def generate_image(provider, prompt, input_image_paths=None, input_image_url=None, aspect_ratio=None, size=None):
    key = APIConfig.AIAPIROUTE_Antigravity_API_KEY
    if not key:
        raise ValueError("AIAPIROUTE_Antigravity_API_KEY is required")

    config = APIConfig.PROVIDERS[provider]
    model = config["model"]
    parts = [{"text": prompt}]
    for image_path in input_image_paths or []:
        data = await asyncio.to_thread(Path(image_path).read_bytes)
        mime_type = mimetypes.guess_type(image_path)[0] or "image/jpeg"
        parts.append({"inlineData": {"mimeType": mime_type, "data": base64.b64encode(data).decode("ascii")}})
    if input_image_url:
        if input_image_url.startswith("data:image/"):
            header, separator, encoded = input_image_url.partition(",")
            if not separator or not header.endswith(";base64") or not encoded:
                raise ValueError("Invalid reference image data URL")
            parts.append({"inlineData": {"mimeType": header[5:-7], "data": encoded}})
        else:
            raise ValueError("Antigravity requires an uploaded reference image or an inline image data URL")

    generation_config = {"responseModalities": ["IMAGE"]}
    image_config = {}
    if aspect_ratio:
        image_config["aspectRatio"] = getattr(aspect_ratio, "value", aspect_ratio)
    if size and size.lower() != "auto":
        image_config["imageSize"] = size.upper()
        if image_config["imageSize"] not in config["sizes"]:
            raise ValueError(f"Unsupported Antigravity image size: {size}")
    if image_config:
        generation_config["imageConfig"] = image_config

    endpoint = f"{APIConfig.AIAPIROUTE_BASE_URL.rstrip('/')}/antigravity/v1beta/models/{model}:generateContent"
    async with httpx.AsyncClient(timeout=APIConfig.AIAPIROUTE_TIMEOUT_SECONDS) as client:
        response = await client.post(
            endpoint,
            headers={"x-goog-api-key": key},
            json={"contents": [{"role": "user", "parts": parts}], "generationConfig": generation_config},
        )
    response.raise_for_status()
    payload = response.json()
    for candidate in payload.get("candidates") or []:
        for part in (candidate.get("content") or {}).get("parts") or []:
            inline = part.get("inlineData") or {}
            mime_type = inline.get("mimeType", "")
            if not part.get("thought") and mime_type.startswith("image/") and inline.get("data"):
                return {
                    "status": "succeeded",
                    "output": f"data:{mime_type};base64,{inline['data']}",
                    "id": payload.get("responseId") or str(uuid4()),
                    "api_type": provider,
                    "raw": {"model": model},
                    "extracted_seed": None,
                }
    raise ValueError("AIAPIROUTE Antigravity returned no generated image")
