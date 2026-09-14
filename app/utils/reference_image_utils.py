#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""参考图预处理工具。"""

import base64
from io import BytesIO

from PIL import Image, ImageFilter, ImageOps


def image_file_to_fitted_data_url(image_path: str, aspect_ratio: str) -> str:
    """将参考图无损适配为指定比例并编码为 PNG Data URL。

    原图完整居中保留；比例不一致时扩展画布并用模糊背景填充，
    不会再次裁掉前端已经选定的有效区域。只处理发送给模型的数据，
    不覆盖任务目录中的原始上传文件。
    """
    width_ratio, height_ratio = [int(part.strip()) for part in aspect_ratio.split(":", 1)]
    target_ratio = width_ratio / height_ratio

    with Image.open(image_path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        source_width, source_height = image.size
        source_ratio = source_width / source_height

        if abs(source_ratio - target_ratio) < 0.0001:
            fitted = image.copy()
        elif source_ratio < target_ratio:
            canvas_width = max(source_width, round(source_height * target_ratio))
            canvas_height = source_height
            background = ImageOps.fit(
                image,
                (canvas_width, canvas_height),
                method=Image.Resampling.LANCZOS,
            )
            blur_radius = max(8, round(min(canvas_width, canvas_height) * 0.025))
            fitted = background.filter(ImageFilter.GaussianBlur(blur_radius))
            fitted.paste(image, ((canvas_width - source_width) // 2, 0))
        else:
            canvas_width = source_width
            canvas_height = max(source_height, round(source_width / target_ratio))
            background = ImageOps.fit(
                image,
                (canvas_width, canvas_height),
                method=Image.Resampling.LANCZOS,
            )
            blur_radius = max(8, round(min(canvas_width, canvas_height) * 0.025))
            fitted = background.filter(ImageFilter.GaussianBlur(blur_radius))
            fitted.paste(image, (0, (canvas_height - source_height) // 2))

        output = BytesIO()
        fitted.save(output, format="PNG", optimize=True)

    encoded = base64.b64encode(output.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"
