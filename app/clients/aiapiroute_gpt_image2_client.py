#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
aiapiroute/Sub2API GPT-image 客户端

通过 OpenAI 兼容 Images API 调用 aiapiroute/Sub2API GPT-image 系列模型。
"""

import base64
import json
import mimetypes
import os
import re
from datetime import datetime
from typing import Any, Optional

import httpx

from app.core.config import APIConfig

AIAPIROUTE_IMAGE_MAX_EDGE = 3840
AIAPIROUTE_IMAGE_MIN_PIXELS = 655_360
AIAPIROUTE_IMAGE_MAX_PIXELS = 8_294_400
AIAPIROUTE_IMAGE_MAX_RATIO = 3
RESOLUTION_LONG_EDGE = {
    "1k": 1536,
    "2k": 2048,
    "4k": AIAPIROUTE_IMAGE_MAX_EDGE,
}
RESOLUTION_SQUARE_EDGE = {
    "1k": 1024,
    "2k": 2048,
    "4k": 2880,
}


class AIApiRouteGPTImageClient:
    """aiapiroute GPT-image 客户端，返回项目统一的生成结果结构。"""

    def __init__(self, api_key: str = None, base_url: str = None, model: str = None, timeout: int = None):
        self.api_key = api_key or APIConfig.AIAPIROUTE_API_KEY
        self.base_url = (base_url or APIConfig.AIAPIROUTE_BASE_URL).rstrip("/")
        self.model = model or APIConfig.AIAPIROUTE_GPT_IMAGE2_MODEL
        self.timeout = timeout or APIConfig.AIAPIROUTE_TIMEOUT_SECONDS

        if not self.api_key:
            raise ValueError("AIAPIROUTE_API_KEY is required")
        if not self.base_url:
            raise ValueError("AIAPIROUTE_BASE_URL is required")

    async def generate_image(self, prompt, input_image_paths=None, input_image_url=None,
                             seed=None, aspect_ratio=None, size=None, quality=None, stream=None, **kwargs):
        """调用 GPT-image 系列模型生成/编辑图像。"""
        reference_images = []
        if input_image_paths:
            reference_images.extend(input_image_paths)
        if input_image_url:
            reference_images.append(input_image_url)

        request_size = self._resolve_size(size, aspect_ratio)
        should_stream = APIConfig.AIAPIROUTE_IMAGE_STREAM if stream is None else bool(stream)
        full_prompt = f"{prompt}\n\nSeed: {seed}" if seed is not None else prompt

        payload = {
            "model": self.model,
            "prompt": full_prompt,
            "size": request_size,
            "n": 1,
            "response_format": "b64_json",
        }
        if quality or APIConfig.AIAPIROUTE_IMAGE_QUALITY:
            payload["quality"] = quality or APIConfig.AIAPIROUTE_IMAGE_QUALITY

        if reference_images:
            payload["images"] = [{"image_url": self._to_data_url(image)} for image in reference_images]

        endpoint = "/v1/images/edits" if reference_images else "/v1/images/generations"
        response_data, raw_text = await self._post_json(endpoint, payload, stream=should_stream)
        b64_image = self._find_base64(response_data) or self._find_base64(raw_text)

        if not b64_image and reference_images and endpoint == "/v1/images/edits":
            response_payload = self._build_responses_payload(full_prompt, reference_images, request_size, quality)
            response_data, raw_text = await self._post_json("/v1/responses", response_payload, stream=False)
            b64_image = self._find_base64(response_data) or self._find_base64(raw_text)
            endpoint = "/v1/responses"

        if not b64_image:
            preview = raw_text[:1000] if raw_text else str(response_data)[:1000]
            raise ValueError(f"aiapiroute GPT-image 未返回可识别的 base64 图片。响应预览: {preview}")

        mime_type = self._detect_mime_from_base64(b64_image)
        data_url = f"data:{mime_type};base64,{self._normalize_base64(b64_image)}"
        prediction_id = f"aiapiroute_gpt_image_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{seed or 'na'}"

        return {
            "id": prediction_id,
            "status": "succeeded",
            "output": data_url,
            "output_for_json": "base64_data_removed_for_brevity",
            "logs": f"aiapiroute GPT-image endpoint={endpoint}, size={request_size}",
            "input": {
                "prompt": prompt,
                "model": self.model,
                "size": request_size,
                "seed": seed,
                "reference_image_count": len(reference_images),
            },
            "raw": self._scrub_large_base64(response_data),
            "api_type": "aiapiroute_gpt_image",
            "extracted_seed": seed,
        }

    async def _post_json(self, endpoint: str, payload: dict[str, Any], stream: bool = False):
        request_payload = {**payload}
        if stream:
            request_payload["stream"] = True

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream, application/json" if stream else "application/json",
            "Connection": "keep-alive",
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(f"{self.base_url}{endpoint}", headers=headers, json=request_payload)
            text = response.text
            if response.status_code >= 400:
                raise ValueError(f"aiapiroute HTTP {response.status_code}: {text[:1000]}")

        if stream or "text/event-stream" in response.headers.get("content-type", ""):
            events = self._parse_sse_events(text)
            return {"events": events}, text

        try:
            return response.json(), text
        except json.JSONDecodeError:
            return {}, text

    def _build_responses_payload(self, prompt: str, reference_images: list[str], size: str, quality: Optional[str]):
        content = [{"type": "input_text", "text": prompt}]
        for image in reference_images:
            content.append({"type": "input_image", "image_url": self._to_data_url(image)})

        tool = {"type": "image_generation", "size": size}
        if quality or APIConfig.AIAPIROUTE_IMAGE_QUALITY:
            tool["quality"] = quality or APIConfig.AIAPIROUTE_IMAGE_QUALITY

        return {
            "model": self.model,
            "input": [{"role": "user", "content": content}],
            "tools": [tool],
        }

    def _to_data_url(self, image: str) -> str:
        if not image:
            raise ValueError("空图片输入")
        if image.startswith("data:image/"):
            return image
        if image.startswith(("http://", "https://")):
            return image
        if not os.path.exists(image):
            raise ValueError(f"Image file not found: {image}")

        mime_type, _ = mimetypes.guess_type(image)
        if not mime_type or not mime_type.startswith("image/"):
            mime_type = "image/jpeg"
        with open(image, "rb") as image_file:
            b64 = base64.b64encode(image_file.read()).decode("utf-8")
        return f"data:{mime_type};base64,{b64}"

    def _resolve_size(self, size: Optional[str], aspect_ratio) -> str:
        size_value = str(size or "").strip()
        if size_value.lower() == "auto":
            return "auto"
        if size_value and "x" in size_value.lower():
            return size_value

        resolution_key = str(size or APIConfig.AIAPIROUTE_IMAGE_RESOLUTION or "1K").strip().lower()
        long_side = RESOLUTION_LONG_EDGE.get(resolution_key, RESOLUTION_LONG_EDGE["1k"])
        ratio_value = getattr(aspect_ratio, "value", aspect_ratio) or "1:1"
        if ratio_value == "match_input_image":
            ratio_value = "1:1"

        try:
            width_ratio, height_ratio = [int(part) for part in str(ratio_value).split(":", 1)]
        except Exception:
            width_ratio, height_ratio = 1, 1

        if width_ratio == height_ratio:
            square_edge = RESOLUTION_SQUARE_EDGE.get(resolution_key, RESOLUTION_SQUARE_EDGE["1k"])
            width, height = square_edge, square_edge
        else:
            width, height = self._size_from_ratio(width_ratio, height_ratio, long_side)
        return f"{width}x{height}"

    def _size_from_ratio(self, width_ratio: int, height_ratio: int, long_side: int) -> tuple[int, int]:
        width_ratio = max(1, int(width_ratio))
        height_ratio = max(1, int(height_ratio))
        long_ratio = max(width_ratio, height_ratio)
        short_ratio = min(width_ratio, height_ratio)
        if long_ratio / short_ratio > AIAPIROUTE_IMAGE_MAX_RATIO:
            raise ValueError("aiapiroute image size ratio must not exceed 3:1")

        scale = min(long_side / long_ratio, AIAPIROUTE_IMAGE_MAX_EDGE / long_ratio)
        width = self._round_to_multiple(width_ratio * scale, 16)
        height = self._round_to_multiple(height_ratio * scale, 16)

        pixels = width * height
        if pixels > AIAPIROUTE_IMAGE_MAX_PIXELS:
            scale *= (AIAPIROUTE_IMAGE_MAX_PIXELS / pixels) ** 0.5
            width = self._round_to_multiple(width_ratio * scale, 16)
            height = self._round_to_multiple(height_ratio * scale, 16)
        elif pixels < AIAPIROUTE_IMAGE_MIN_PIXELS:
            scale *= (AIAPIROUTE_IMAGE_MIN_PIXELS / pixels) ** 0.5
            width = self._round_to_multiple(width_ratio * scale, 16)
            height = self._round_to_multiple(height_ratio * scale, 16)

        if width * height > AIAPIROUTE_IMAGE_MAX_PIXELS:
            width = self._round_down_to_multiple(width, 16)
            height = self._round_down_to_multiple(height, 16)
        return width, height

    @staticmethod
    def _round_to_multiple(value: float, multiple: int) -> int:
        return max(multiple, round(value / multiple) * multiple)

    @staticmethod
    def _round_down_to_multiple(value: int, multiple: int) -> int:
        return max(multiple, (value // multiple) * multiple)

    def _parse_sse_events(self, text: str) -> list[Any]:
        events = []
        for block in str(text or "").split("\n\n"):
            data_lines = []
            for line in block.splitlines():
                if line.startswith("data:"):
                    data_lines.append(line[5:].strip())
            data = "\n".join(data_lines).strip()
            if not data or data == "[DONE]":
                continue
            try:
                events.append(json.loads(data))
            except json.JSONDecodeError:
                events.append({"raw": data})
        return events

    def _find_base64(self, value: Any) -> str:
        if not value:
            return ""
        if isinstance(value, str):
            if value.startswith("data:image/"):
                return value.split(",", 1)[1].strip()
            match = re.search(r'"(?:b64_json|base64|image_base64|result|data|url)"\s*:\s*"([A-Za-z0-9+/=\r\n]{1000,})"', value)
            if match:
                return self._normalize_base64(match.group(1))
            if len(value) > 1000 and re.fullmatch(r"[A-Za-z0-9+/=\r\n]+", value.strip()):
                return self._normalize_base64(value)
            return ""
        if isinstance(value, list):
            for item in value:
                found = self._find_base64(item)
                if found:
                    return found
            return ""
        if isinstance(value, dict):
            for key in ["b64_json", "base64", "image_base64", "result", "url", "data", "events", "output"]:
                found = self._find_base64(value.get(key))
                if found:
                    return found
            for item in value.values():
                found = self._find_base64(item)
                if found:
                    return found
        return ""

    @staticmethod
    def _normalize_base64(value: str) -> str:
        text = str(value or "").strip()
        if text.startswith("data:"):
            text = text.split(",", 1)[1]
        return re.sub(r"\s+", "", text)

    def _detect_mime_from_base64(self, value: str) -> str:
        normalized = self._normalize_base64(value)
        try:
            header = base64.b64decode(normalized[:64] + "===")[:12]
        except Exception:
            return "image/png"
        if header.startswith(b"\xff\xd8\xff"):
            return "image/jpeg"
        if header.startswith(b"RIFF") and b"WEBP" in header:
            return "image/webp"
        return "image/png"

    def _scrub_large_base64(self, value: Any):
        if isinstance(value, str):
            if value.startswith("data:image/") or (len(value) > 1000 and re.fullmatch(r"[A-Za-z0-9+/=\r\n]+", value.strip())):
                return "base64_data_removed_for_brevity"
            return value
        if isinstance(value, list):
            return [self._scrub_large_base64(item) for item in value]
        if isinstance(value, dict):
            scrubbed = {}
            for key, item in value.items():
                if key in {"b64_json", "base64", "image_base64", "result", "url"} and isinstance(item, str) and len(item) > 1000:
                    scrubbed[key] = "base64_data_removed_for_brevity"
                else:
                    scrubbed[key] = self._scrub_large_base64(item)
            return scrubbed
        return value


AIApiRouteGPTImage2Client = AIApiRouteGPTImageClient
