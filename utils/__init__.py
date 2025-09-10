#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
工具模块包

这个包包含了所有用于Flux Kontext AI图像生成API的工具模块：
- utils_upscale: 图像放大功能
- utils_task: 任务管理功能  
- utils_watermark: 水印功能
"""

__version__ = "1.0.0"
__author__ = "AI-replicate Team"

# 导出主要功能
from .utils_upscale import (
    upscale_image_with_replicate,
    download_upscaled_image,
    generate_upscale_filename,
    save_upscale_task_info,
    create_upscale_lookup_folder,
    get_upscale_models_info,
    validate_upscale_params,
    build_upscale_input_params
)

from .utils_task import (
    generate_task_dir,
    save_params,
    generate_output_filenames
)

from .utils_watermark import (
    create_logo_watermark,
    add_logo_watermark,
    add_corner_label
)

# 异步组件
from .async_task_manager import task_manager, TaskStatus, get_task_status, get_system_stats

# 统一API客户端
from .unified_api_client import api_client





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
    
    # 异步组件
    'task_manager',
    'TaskStatus',
    'get_task_status',
    'get_system_stats',
    
    # 统一API客户端
    'api_client',
    
    
] 