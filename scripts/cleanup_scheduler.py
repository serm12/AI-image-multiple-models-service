#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
定期清理tasks目录的调度脚本
可以设置为定时任务定期执行
"""

import os
import sys
import time
import argparse

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

def run_cleanup(tasks_dir: str, dry_run: bool):
    """执行一次清理"""
    try:
        from utils.cleanup_tasks import cleanup_tasks_folder, set_dry_run_mode
        set_dry_run_mode(dry_run)
        cleanup_tasks_folder(tasks_dir)
        print("清理任务执行完成。")
    except Exception as e:
        print(f"清理任务执行出错: {e}")

def main():
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
    parser.add_argument(
        '--loop',
        action='store_true',
        help='循环模式（Docker 容器用），持续运行不退出'
    )
    parser.add_argument(
        '--interval',
        type=int,
        default=86400,
        help='循环模式下的执行间隔秒数 (默认: 86400 = 24小时)'
    )

    args = parser.parse_args()

    if args.loop:
        print(f"循环清理模式启动，间隔 {args.interval} 秒")
        print(f"Tasks目录: {args.tasks_dir}")
        while True:
            print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] 开始执行清理...")
            run_cleanup(args.tasks_dir, args.dry_run)
            time.sleep(args.interval)
    else:
        print(f"开始执行tasks目录清理任务...")
        print(f"Tasks目录: {args.tasks_dir}")
        print(f"试运行模式: {args.dry_run}")
        run_cleanup(args.tasks_dir, args.dry_run)
        # 非 loop 模式出错时返回非零退出码
        # （run_cleanup 内部已打印错误，这里不 sys.exit）

if __name__ == "__main__":
    main()