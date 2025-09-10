#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
定期清理tasks目录的调度脚本
可以设置为定时任务定期执行
"""

import os
import sys
import argparse

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='清理tasks目录中的旧文件和文件夹')
    parser.add_argument(
        '--tasks-dir', 
        default='tasks',
        help='tasks目录路径 (默认: tasks)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='试运行模式，只显示将要执行的操作而不实际执行'
    )
    
    args = parser.parse_args()
    
    print(f"开始执行tasks目录清理任务...")
    print(f"Tasks目录: {args.tasks_dir}")
    print(f"试运行模式: {args.dry_run}")
    
    try:
        # 延迟导入，确保路径已正确设置
        from utils.cleanup_tasks import cleanup_tasks_folder, set_dry_run_mode
        # 设置试运行模式
        set_dry_run_mode(args.dry_run)
        cleanup_tasks_folder(args.tasks_dir)
        print("清理任务执行完成。")
    except Exception as e:
        print(f"清理任务执行出错: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()