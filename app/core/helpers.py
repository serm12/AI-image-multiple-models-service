#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
核心模块
包含项目的核心类和共享功能
"""

from app.core.config import STYLE_DESCRIPTIONS
from app.utils.time_utils import now_china

def get_style_description(style_value: str) -> str:
    """获取风格描述"""
    return STYLE_DESCRIPTIONS.get(style_value, "未知风格")

def create_response_record(prediction_id=None, output_url="", status="succeeded", **kwargs):
    """创建标准的响应记录"""
    return {
        "completed_at": now_china().isoformat(),
        "created_at": now_china().isoformat(),
        "data_removed": False,
        "error": None,
        "id": prediction_id,
        "logs": kwargs.get("logs", ""),
        "metrics": {
            "image_count": 1,
            "predict_time": 0,
            "total_time": 0
        },
        "output": output_url,
        "started_at": now_china().isoformat(),
        "status": status,
        "urls": {
            "stream": "",
            "cancel": "",
            "get": "",
            "web": ""
        },
        "version": "hidden",
        **kwargs.get("extra_fields", {})
    } 
