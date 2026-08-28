#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""参考图预处理工具。"""

import base64
from io import BytesIO

from PIL import Image, ImageOps


def image_file_to_cropped_data_url(image_path: str, aspect_ratio: str) -> str:
    """将本地参考图居中裁剪为指定比例并编码为 PNG Data URL。

    只处理发送给模型的数据，不覆盖任务目录中的原始上传文件。
    """
    width_ratio, height_ratio = [int(part.strip()) for part in aspect_ratio.split(":", 1)]
    target_ratio = width_ratio / height_ratio

    with Image.open(image_path) as source:
        image = ImageOps.exif_transpose(source)
        source_width, source_height = image.size
        source_ratio = source_width / source_height

        if source_ratio > target_ratio:
            crop_width = max(1, round(source_height * target_ratio))
            left = (source_width - crop_width) // 2
            box = (left, 0, left + crop_width, source_height)
        else:
            crop_height = max(1, round(source_width / target_ratio))
            top = (source_height - crop_height) // 2
            box = (0, top, source_width, top + crop_height)

        cropped = image.crop(box)
        output = BytesIO()
        cropped.save(output, format="PNG", optimize=True)

    encoded = base64.b64encode(output.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"