#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
应用底层工具包

这个包只保留应用底层通用工具和脚本依赖：
- upscale_utils: 图像放大工具
- task_utils: 任务文件工具
- watermark_utils: 水印工具
- cleanup_tasks: 任务目录清理工具

应用服务位于 app/services，外部 provider 客户端位于 app/clients。
"""

__version__ = "1.0.0"
__author__ = "AI-replicate Team"

# 导出主要功能
from .upscale_utils import (
    upscale_image_with_replicate,
    download_upscaled_image,
    generate_upscale_filename,
    save_upscale_task_info,
    create_upscale_lookup_folder,
    get_upscale_models_info,
    validate_upscale_params,
    build_upscale_input_params
)

from .task_utils import (
    generate_task_dir,
    save_params,
    generate_output_filenames
)

from .watermark_utils import (
    create_logo_watermark,
    add_logo_watermark,
    add_corner_label
)

__all__ = [
    # 放大功能
    'upscale_image_with_replicate',
    'download_upscaled_image',
    'generate_upscale_filename',
    'save_upscale_task_info',
    'create_upscale_lookup_folder',
    'get_upscale_models_info',
    'validate_upscale_params',
    'build_upscale_input_params',
    
    # 任务管理
    'generate_task_dir',
    'save_params',
    'generate_output_filenames',
    
    # 水印功能
    'create_logo_watermark',
    'add_logo_watermark',
    'add_corner_label',
] 
