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

from app.projects.my3dfigure.core.config import APIConfig
AIAPIROUTE_IMAGE_MAX_RATIO = 3


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

        request_options = self._resolve_request_options(size, aspect_ratio)
        should_stream = APIConfig.AIAPIROUTE_IMAGE_STREAM if stream is None else bool(stream)
        full_prompt = prompt

        payload = {
            "model": self.model,
            "prompt": full_prompt,
            "n": 1,
            "response_format": "b64_json",
            **request_options,
        }
        if quality or APIConfig.AIAPIROUTE_IMAGE_QUALITY:
            payload["quality"] = quality or APIConfig.AIAPIROUTE_IMAGE_QUALITY

        # My3dFigure submits the validated upload master directly.  The upstream
        # edit request must not apply a second crop, resize, or aspect-ratio fit.
        request_reference_images = list(reference_images)

        endpoint = "/v1/images/edits" if reference_images else "/v1/images/generations"
        if reference_images:
            response_data, raw_text = await self._post_multipart(
                endpoint, payload, request_reference_images, stream=should_stream
            )
        else:
            response_data, raw_text = await self._post_json(endpoint, payload, stream=should_stream)
        b64_image = self._find_base64(response_data) or self._find_base64(raw_text)

        if not b64_image and reference_images and endpoint == "/v1/images/edits":
            response_payload = self._build_responses_payload(
                full_prompt, request_reference_images, request_options, quality
            )
            response_data, raw_text = await self._post_json("/v1/responses", response_payload, stream=False)
            b64_image = self._find_base64(response_data) or self._find_base64(raw_text)
            endpoint = "/v1/responses"

        if not b64_image:
            preview = raw_text[:1000] if raw_text else str(response_data)[:1000]
            raise ValueError(f"aiapiroute GPT-image 未返回可识别的 base64 图片。响应预览: {preview}")

        mime_type = self._detect_mime_from_base64(b64_image)
        data_url = f"data:{mime_type};base64,{self._normalize_base64(b64_image)}"
        prediction_id = f"aiapiroute_gpt_image_{datetime.now().strftime('%Y%m%d_%H%M%S')}_na"

        return {
            "id": prediction_id,
            "status": "succeeded",
            "output": data_url,
            "output_for_json": "base64_data_removed_for_brevity",
            "logs": (
                f"aiapiroute GPT-image endpoint={endpoint}, request_options={request_options}, "
                "reference_preprocessing=disabled, "
                "reference_transport=uploaded_master_direct"
            ),
            "input": {
                "prompt": prompt,
                "model": self.model,
                **request_options,
                "seed": None,
                "aspect_ratio": getattr(aspect_ratio, "value", aspect_ratio),
                "reference_ratio": None,
                "reference_image_count": len(reference_images),
            },
            "raw": self._scrub_large_base64(response_data),
            "api_type": "aiapiroute_gpt_image",
            "extracted_seed": seed,
        }

    async def _post_json(self, endpoint: str, payload: dict[str, Any], stream: bool = False):
        request_payload = {**payload}
        request_payload["stream"] = bool(stream)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "x-api-key": self.api_key,
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

    async def _post_multipart(
        self,
        endpoint: str,
        payload: dict[str, Any],
        reference_images: list[str],
        stream: bool = False,
    ):
        """Submit a reference edit using Sub2API v0.2.5 multipart form data."""
        form_data = {
            key: str(value).lower() if isinstance(value, bool) else str(value)
            for key, value in payload.items()
            if value is not None
        }
        form_data["stream"] = str(stream).lower()
        files = []

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            for index, image in enumerate(reference_images, start=1):
                filename, content, mime_type = await self._load_image_part(client, image, index)
                files.append(("image", (filename, content, mime_type)))
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "x-api-key": self.api_key,
                "Accept": "text/event-stream, application/json" if stream else "application/json",
                "Connection": "keep-alive",
            }
            response = await client.post(
                f"{self.base_url}{endpoint}", headers=headers, data=form_data, files=files
            )
            text = response.text
            if response.status_code >= 400:
                raise ValueError(f"aiapiroute HTTP {response.status_code}: {text[:1000]}")

        if stream or "text/event-stream" in response.headers.get("content-type", ""):
            return {"events": self._parse_sse_events(text)}, text
        try:
            return response.json(), text
        except json.JSONDecodeError:
            return {}, text

    async def _load_image_part(
        self, client: httpx.AsyncClient, image: str, index: int
    ) -> tuple[str, bytes, str]:
        if not image:
            raise ValueError("空图片输入")
        if image.startswith("data:image/"):
            header, encoded = image.split(",", 1)
            mime_type = header[5:].split(";", 1)[0] or "image/png"
            extension = mimetypes.guess_extension(mime_type) or ".png"
            return f"image-{index}{extension}", base64.b64decode(encoded), mime_type
        if image.startswith(("http://", "https://")):
            response = await client.get(image)
            response.raise_for_status()
            mime_type = response.headers.get("content-type", "image/jpeg").split(";", 1)[0]
            filename = os.path.basename(response.url.path) or f"image-{index}"
            if "." not in filename:
                filename += mimetypes.guess_extension(mime_type) or ".jpg"
            return filename, response.content, mime_type
        if not os.path.exists(image):
            raise ValueError(f"Image file not found: {image}")
        mime_type, _ = mimetypes.guess_type(image)
        with open(image, "rb") as image_file:
            return os.path.basename(image), image_file.read(), mime_type or "image/jpeg"

    def _build_responses_payload(
        self,
        prompt: str,
        reference_images: list[str],
        request_options: dict[str, str],
        quality: Optional[str],
    ):
        content = [{"type": "input_text", "text": prompt}]
        for image in reference_images:
            content.append({
                "type": "input_image",
                "image_url": self._to_data_url(image),
            })

        tool = {"type": "image_generation", **request_options}
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

    def _resolve_request_options(self, size: Optional[str], aspect_ratio) -> dict[str, str]:
        """Use native Sub2API ratio/resolution parameters for GPT-image edits."""
        ratio_value = str(getattr(aspect_ratio, "value", aspect_ratio) or "").strip()
        size_value = str(size or APIConfig.AIAPIROUTE_IMAGE_RESOLUTION or "1K").strip()
        options = {}
        if ratio_value and ratio_value != "match_input_image":
            self._validate_aspect_ratio(ratio_value)
            options["aspect_ratio"] = ratio_value
        if re.fullmatch(r"(?i)[124]k", size_value):
            options["resolution"] = size_value.upper()
        elif size_value.lower() == "auto":
            options["size"] = "auto"
        elif re.fullmatch(r"\d+x\d+", size_value.lower()):
            options["size"] = size_value.lower()
        return options

    @staticmethod
    def _validate_aspect_ratio(aspect_ratio: str) -> None:
        try:
            width_ratio, height_ratio = [
                int(part.strip()) for part in str(aspect_ratio).split(":", 1)
            ]
            if width_ratio <= 0 or height_ratio <= 0:
                raise ValueError
            if max(width_ratio, height_ratio) / min(width_ratio, height_ratio) > AIAPIROUTE_IMAGE_MAX_RATIO:
                raise ValueError
        except (TypeError, ValueError, ZeroDivisionError) as exc:
            raise ValueError(
                "aiapiroute aspect_ratio 必须是有效且不超过 3:1 的比例，例如 1:1、3:4 或 16:9"
            ) from exc

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
