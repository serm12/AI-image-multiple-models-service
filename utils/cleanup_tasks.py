#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
定期清理tasks目录中的文件夹
根据规则：
1. 如果文件夹里面没有watermark.md和original.md文件，则删除7天前的所有文件夹。
2. 如果文件夹里面有watermark.md而没有original.md则删除28天前的所有文件夹。
3. 如果文件夹里面有original.md则删除365天前的所有文件夹。
"""

import os
import shutil
import logging
from datetime import datetime, timedelta

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/cleanup.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def get_file_age_days(file_path):
    """
    获取文件的年龄（天数）
    
    Args:
        file_path (str): 文件路径
        
    Returns:
        int: 文件年龄（天数）
    """
    try:
        file_mod_time = datetime.fromtimestamp(os.path.getmtime(file_path))
        age = datetime.now() - file_mod_time
        return age.days
    except Exception as e:
        logger.error(f"获取文件 {file_path} 年龄时出错: {e}")
        return 0

def cleanup_old_folders_by_rule(folder_path, base_path):
    """
    根据规则清理文件夹
    
    Args:
        folder_path (str): 要检查的文件夹路径
        base_path (str): tasks基础路径，用于计算相对路径
        
    Returns:
        tuple: (操作类型, 描述)
    """
    try:
        folder_name = os.path.basename(folder_path)
        
        # 检查文件夹是否存在
        if not os.path.exists(folder_path):
            return ("skip", f"文件夹不存在: {folder_name}")
            
        # 检查watermark.md和original.md是否存在
        watermark_md_path = os.path.join(folder_path, "watermark.md")
        original_md_path = os.path.join(folder_path, "original.md")
        
        has_watermark = os.path.exists(watermark_md_path)
        has_original = os.path.exists(original_md_path)
        
        folder_age = get_file_age_days(folder_path)
        
        # 规则1: 如果文件夹里面没有watermark.md和original.md文件，则删除7天前的所有文件夹
        if not has_watermark and not has_original:
            if folder_age > 7:
                shutil.rmtree(folder_path)
                return ("delete_folder", f"删除文件夹 {folder_name} (无watermark.md和original.md，{folder_age}天前)")
            else:
                return ("keep", f"保留文件夹 {folder_name} (无watermark.md和original.md，{folder_age}天前，未到7天)")
                
        # 规则2: 如果文件夹里面有watermark.md而没有original.md则删除28天前的所有文件夹
        elif has_watermark and not has_original:
            if folder_age > 28:
                shutil.rmtree(folder_path)
                return ("delete_folder", f"删除文件夹 {folder_name} (有watermark.md无original.md，{folder_age}天前)")
            else:
                return ("keep", f"保留文件夹 {folder_name} (有watermark.md无original.md，{folder_age}天前，未到28天)")
                
        # 规则3: 如果文件夹里面有original.md则删除365天前的所有文件夹
        elif has_original:
            if folder_age > 365:
                shutil.rmtree(folder_path)
                return ("delete_folder", f"删除文件夹 {folder_name} (有original.md，{folder_age}天前)")
            else:
                return ("keep", f"保留文件夹 {folder_name} (有original.md，{folder_age}天前，未到365天)")
                
    except Exception as e:
        logger.error(f"处理文件夹 {folder_path} 时出错: {e}")
        return ("error", f"处理文件夹 {folder_path} 时出错: {e}")

def cleanup_tasks_folder(tasks_dir="tasks"):
    """
    清理tasks文件夹
    
    Args:
        tasks_dir (str): tasks文件夹路径
    """
    logger.info(f"开始清理tasks文件夹: {tasks_dir}")
    
    if not os.path.exists(tasks_dir):
        logger.warning(f"tasks文件夹不存在: {tasks_dir}")
        return
    
    # 确保logs目录存在
    os.makedirs("logs", exist_ok=True)
    
    results = {
        "delete_folder": 0,
        "keep": 0,
        "skip": 0,
        "error": 0
    }
    
    # 遍历tasks文件夹中的所有项目
    for item in os.listdir(tasks_dir):
        item_path = os.path.join(tasks_dir, item)
        
        # 只处理文件夹
        if os.path.isdir(item_path):
            try:
                operation, description = cleanup_old_folders_by_rule(item_path, tasks_dir)
                logger.info(description)
                results[operation] += 1
            except Exception as e:
                logger.error(f"处理文件夹 {item} 时出错: {e}")
                results["error"] += 1
    
    logger.info("清理完成统计:")
    logger.info(f"  删除的文件夹: {results['delete_folder']}")
    logger.info(f"  保留的文件夹: {results['keep']}")
    logger.info(f"  跳过的项目: {results['skip']}")
    logger.info(f"  错误数量: {results['error']}")

if __name__ == "__main__":
    # 执行清理任务
    cleanup_tasks_folder("tasks")