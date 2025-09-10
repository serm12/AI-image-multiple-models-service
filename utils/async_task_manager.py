#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
异步任务状态管理器
管理异步生成任务的状态和并发控制
"""

import asyncio
import time
from typing import Dict, Any, Optional
from datetime import datetime
from enum import Enum

class TaskStatus(str, Enum):
    """任务状态枚举"""
    SUBMITTED = "submitted"        # 已提交
    PROCESSING = "processing"      # 处理中
    GENERATING = "generating"      # 生成中
    DOWNLOADING = "downloading"    # 下载中
    COMPLETED = "completed"        # 已完成
    FAILED = "failed"             # 失败
    CANCELLED = "cancelled"       # 已取消

class AsyncTaskManager:
    """异步任务管理器"""
    
    def __init__(self, max_concurrent: int = 1):
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.tasks: Dict[str, Dict[str, Any]] = {}
        self.active_tasks: Dict[str, asyncio.Task] = {}
    
    def create_task(self, task_id: str, **kwargs) -> Dict[str, Any]:
        """创建新任务"""
        task_info = {
            "task_id": task_id,
            "status": TaskStatus.SUBMITTED,
            "progress": 0,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "error": None,
            "prediction_id": None,
            "result": None,
            **kwargs
        }
        self.tasks[task_id] = task_info
        return task_info
    
    def update_task(self, task_id: str, **updates) -> bool:
        """更新任务状态"""
        if task_id not in self.tasks:
            return False
        
        self.tasks[task_id].update(updates)
        self.tasks[task_id]["updated_at"] = datetime.now().isoformat()
        return True
    
    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """获取任务信息"""
        return self.tasks.get(task_id)
    
    def set_task_failed(self, task_id: str, error: str):
        """设置任务失败"""
        self.update_task(task_id, status=TaskStatus.FAILED, error=error)
    
    def set_task_completed(self, task_id: str, result: Any = None):
        """设置任务完成"""
        self.update_task(task_id, status=TaskStatus.COMPLETED, progress=100, result=result)
    
    async def execute_with_concurrency_limit(self, task_id: str, coro):
        """带并发限制执行任务"""
        async with self.semaphore:
            try:
                # 记录活跃任务
                current_task = asyncio.current_task()
                self.active_tasks[task_id] = current_task
                
                # 执行任务
                result = await coro
                return result
                
            except asyncio.CancelledError:
                self.update_task(task_id, status=TaskStatus.CANCELLED)
                raise
            except Exception as e:
                self.set_task_failed(task_id, str(e))
                raise
            finally:
                # 清理活跃任务记录
                self.active_tasks.pop(task_id, None)
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        total_tasks = len(self.tasks)
        active_count = len(self.active_tasks)
        
        status_counts = {}
        for task in self.tasks.values():
            status = task["status"]
            status_counts[status] = status_counts.get(status, 0) + 1
        
        return {
            "total_tasks": total_tasks,
            "active_tasks": active_count,
            "available_slots": self.max_concurrent - active_count,
            "max_concurrent": self.max_concurrent,
            "status_breakdown": status_counts,
            "concurrency_usage": f"{active_count}/{self.max_concurrent}"
        }
    
    async def cancel_task(self, task_id: str) -> bool:
        """取消任务"""
        if task_id in self.active_tasks:
            task = self.active_tasks[task_id]
            task.cancel()
            self.update_task(task_id, status=TaskStatus.CANCELLED)
            return True
        return False
    
    def cleanup_old_tasks(self, max_age_hours: int = 24):
        """清理旧任务"""
        current_time = time.time()
        cutoff_time = current_time - (max_age_hours * 3600)
        
        to_remove = []
        for task_id, task_info in self.tasks.items():
            created_timestamp = datetime.fromisoformat(task_info["created_at"]).timestamp()
            if created_timestamp < cutoff_time and task_info["status"] in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
                to_remove.append(task_id)
        
        for task_id in to_remove:
            del self.tasks[task_id]
        
        return len(to_remove)

# 全局任务管理器实例
task_manager = AsyncTaskManager(max_concurrent=1)

# 辅助函数
async def create_and_run_task(task_id: str, coro, **task_kwargs):
    """创建并运行任务的便捷函数"""
    task_manager.create_task(task_id, **task_kwargs)
    return await task_manager.execute_with_concurrency_limit(task_id, coro)

def get_task_status(task_id: str) -> Optional[Dict[str, Any]]:
    """获取任务状态的便捷函数"""
    return task_manager.get_task(task_id)

def get_system_stats() -> Dict[str, Any]:
    """获取系统统计信息"""
    return task_manager.get_statistics()

# 测试功能
async def test_task_manager():
    """测试任务管理器"""
    print("🧪 测试异步任务管理器")
    
    # 创建测试任务
    task_id = "test_task_001"
    task_manager.create_task(task_id, test_param="test_value")
    
    print(f"✅ 任务创建成功: {task_id}")
    
    # 更新任务状态
    task_manager.update_task(task_id, status=TaskStatus.PROCESSING, progress=50)
    print(f"✅ 任务状态更新: {task_manager.get_task(task_id)['status']}")
    
    # 完成任务
    task_manager.set_task_completed(task_id, result="test_result")
    print(f"✅ 任务完成: {task_manager.get_task(task_id)['status']}")
    
    # 统计信息
    stats = task_manager.get_statistics()
    print(f"✅ 系统统计: {stats}")

if __name__ == "__main__":
    asyncio.run(test_task_manager())