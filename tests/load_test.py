#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
压力测试脚本 - 模拟多个并发请求，测试系统稳定性
用法: python load_test.py --concurrency 5 --duration 300 --mode mixed
"""

import asyncio
import aiohttp
import time
import json
import statistics
import tempfile
import os
from datetime import datetime
from typing import Dict, List
import argparse

TEST_IMAGE_URL = "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=400"
_cached_image_path: str = ""


async def get_test_image_path(session: aiohttp.ClientSession) -> str:
    """下载测试图片到临时文件（只下一次）"""
    global _cached_image_path
    if _cached_image_path and os.path.exists(_cached_image_path):
        return _cached_image_path
    async with session.get(TEST_IMAGE_URL, timeout=aiohttp.ClientTimeout(total=30)) as resp:
        data = await resp.read()
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    tmp.write(data)
    tmp.close()
    _cached_image_path = tmp.name
    return _cached_image_path


class LoadTester:
    def __init__(self, base_url: str = "http://localhost:8001", concurrency: int = 5):
        self.base_url = base_url
        self.concurrency = concurrency
        self.results: Dict[str, List[float]] = {
            "query_time": [],
            "generate_time": [],
            "status_check_time": [],
            "errors": []
        }
        self.active_tasks = []
        self.created_task_ids = []

    async def test_query_styles(self, session: aiohttp.ClientSession) -> bool:
        """测试查询风格端点"""
        try:
            start = time.time()
            async with session.get(f"{self.base_url}/styles/", timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status == 200:
                    await resp.json()
                    elapsed = time.time() - start
                    self.results["query_time"].append(elapsed)
                    return True
                else:
                    self.results["errors"].append(f"styles: HTTP {resp.status}")
                    return False
        except Exception as e:
            self.results["errors"].append(f"styles: {str(e)}")
            return False

    async def test_query_seeds(self, session: aiohttp.ClientSession) -> bool:
        """测试查询seeds端点"""
        try:
            start = time.time()
            async with session.get(f"{self.base_url}/seeds/", timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status == 200:
                    await resp.json()
                    elapsed = time.time() - start
                    self.results["query_time"].append(elapsed)
                    return True
                else:
                    self.results["errors"].append(f"seeds: HTTP {resp.status}")
                    return False
        except Exception as e:
            self.results["errors"].append(f"seeds: {str(e)}")
            return False

    async def test_generate_image(self, session: aiohttp.ClientSession, worker_id: int) -> tuple:
        """提交一个生成任务"""
        try:
            image_path = await get_test_image_path(session)
            data = aiohttp.FormData()
            data.add_field("prompt", f"测试图片 #{worker_id} - {datetime.now().isoformat()}")
            data.add_field("aspect_ratio", "3:4")
            data.add_field("art_style", "flux_realistic")
            data.add_field("auto_upscale", "false")
            with open(image_path, "rb") as img_f:
                data.add_field("files", img_f, filename="test_input.jpg", content_type="image/jpeg")
                start = time.time()
                async with session.post(
                    f"{self.base_url}/generate-async/",
                    data=data,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        task_id = result.get("task_id")
                        elapsed = time.time() - start
                        self.results["generate_time"].append(elapsed)
                        if task_id:
                            self.created_task_ids.append(task_id)
                        return (True, task_id)
                    else:
                        self.results["errors"].append(f"generate: HTTP {resp.status}")
                        return (False, None)
        except Exception as e:
            self.results["errors"].append(f"generate: {str(e)}")
            return (False, None)

    async def test_check_status(self, session: aiohttp.ClientSession, task_id: str) -> bool:
        """检查任务状态"""
        try:
            start = time.time()
            async with session.get(
                f"{self.base_url}/task-status/{task_id}",
                timeout=aiohttp.ClientTimeout(total=10)
            ) as resp:
                if resp.status == 200:
                    await resp.json()
                    elapsed = time.time() - start
                    self.results["status_check_time"].append(elapsed)
                    return True
                else:
                    self.results["errors"].append(f"status: HTTP {resp.status}")
                    return False
        except Exception as e:
            self.results["errors"].append(f"status: {str(e)}")
            return False

    async def worker_mixed(self, session: aiohttp.ClientSession, worker_id: int, duration: int):
        """混合工作线程 - 混合执行查询和生成"""
        start_time = time.time()
        count = 0
        
        while time.time() - start_time < duration:
            try:
                # 30% 查询，30% 生成，40% 状态检查
                rand = count % 10
                
                if rand < 3:
                    await self.test_query_styles(session)
                elif rand < 6:
                    success, task_id = await self.test_generate_image(session, worker_id)
                else:
                    # 检查最近创建的任务状态
                    if self.created_task_ids:
                        task_id = self.created_task_ids[-1]
                        await self.test_check_status(session, task_id)
                    else:
                        await self.test_query_seeds(session)
                
                count += 1
                await asyncio.sleep(0.1)  # 稍微延迟，避免过度快速请求
            except Exception as e:
                self.results["errors"].append(f"worker {worker_id}: {str(e)}")

    async def worker_generate_only(self, session: aiohttp.ClientSession, worker_id: int):
        """仅生成工作线程 - 一次提交一个生成任务并等待"""
        try:
            success, task_id = await self.test_generate_image(session, worker_id)
            if success and task_id:
                # 轮询检查状态（最多 200s）
                for i in range(200):
                    await asyncio.sleep(1)
                    success, status = await self.check_task_complete(session, task_id)
                    if success:
                        print(f"  ✅ Worker {worker_id}: 任务 {task_id[:8]} 完成")
                        return True
                    if i % 10 == 0:
                        print(f"  ⏳ Worker {worker_id}: 任务 {task_id[:8]} 进行中...")
                print(f"  ⏱️  Worker {worker_id}: 任务 {task_id[:8]} 超时")
        except Exception as e:
            self.results["errors"].append(f"worker {worker_id}: {str(e)}")
        return False

    async def check_task_complete(self, session: aiohttp.ClientSession, task_id: str) -> tuple:
        """检查任务是否完成"""
        try:
            async with session.get(
                f"{self.base_url}/task-status/{task_id}",
                timeout=aiohttp.ClientTimeout(total=5)
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    status = data.get("status", "unknown")
                    return (True, status)
                return (False, "error")
        except Exception as e:
            return (False, str(e))

    async def run_mixed_load_test(self, duration: int):
        """运行混合负载测试"""
        print(f"\n🚀 开始混合负载测试 (并发数: {self.concurrency}, 持续时间: {duration}s)")
        print("=" * 60)
        
        connector = aiohttp.TCPConnector(limit=100)
        timeout = aiohttp.ClientTimeout(total=300)
        
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            tasks = [
                self.worker_mixed(session, i, duration)
                for i in range(self.concurrency)
            ]
            
            # 监控任务
            start_time = time.time()
            while time.time() - start_time < duration:
                elapsed = int(time.time() - start_time)
                print(f"⏱️  运行时间: {elapsed}s | 错误: {len(self.results['errors'])} | 查询: {len(self.results['query_time'])} | 生成: {len(self.results['generate_time'])}")
                await asyncio.sleep(5)
            
            # 等待所有任务完成
            await asyncio.gather(*tasks, return_exceptions=True)

    async def run_generate_only_test(self):
        """运行仅生成负载测试 - 每个worker提交一个生成任务"""
        print(f"\n🚀 开始生成负载测试 (并发数: {self.concurrency})")
        print("=" * 60)
        
        connector = aiohttp.TCPConnector(limit=100)
        timeout = aiohttp.ClientTimeout(total=600)
        
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            tasks = [
                self.worker_generate_only(session, i)
                for i in range(self.concurrency)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            completed = sum(1 for r in results if r is True)
            print(f"\n✅ 完成: {completed}/{self.concurrency} 个生成任务")

    def print_report(self):
        """打印测试报告"""
        print("\n" + "=" * 60)
        print("📊 测试报告")
        print("=" * 60)
        
        if self.results["query_time"]:
            times = self.results["query_time"]
            print(f"\n📌 查询请求 (共 {len(times)} 个):")
            print(f"   - 平均: {statistics.mean(times):.3f}s")
            print(f"   - 中位数: {statistics.median(times):.3f}s")
            print(f"   - 最小: {min(times):.3f}s")
            print(f"   - 最大: {max(times):.3f}s")
            if len(times) > 1:
                print(f"   - 标准差: {statistics.stdev(times):.3f}s")

        if self.results["generate_time"]:
            times = self.results["generate_time"]
            print(f"\n📌 生成请求 (共 {len(times)} 个):")
            print(f"   - 平均: {statistics.mean(times):.3f}s")
            print(f"   - 中位数: {statistics.median(times):.3f}s")
            print(f"   - 最小: {min(times):.3f}s")
            print(f"   - 最大: {max(times):.3f}s")
            if len(times) > 1:
                print(f"   - 标准差: {statistics.stdev(times):.3f}s")

        if self.results["status_check_time"]:
            times = self.results["status_check_time"]
            print(f"\n📌 状态检查 (共 {len(times)} 个):")
            print(f"   - 平均: {statistics.mean(times):.3f}s")
            print(f"   - 中位数: {statistics.median(times):.3f}s")
            print(f"   - 最小: {min(times):.3f}s")
            print(f"   - 最大: {max(times):.3f}s")
            if len(times) > 1:
                print(f"   - 标准差: {statistics.stdev(times):.3f}s")

        if self.results["errors"]:
            print(f"\n⚠️  错误 (共 {len(self.results['errors'])} 个):")
            # 统计错误类型
            error_types = {}
            for err in self.results["errors"]:
                error_type = err.split(":")[0]
                error_types[error_type] = error_types.get(error_type, 0) + 1
            for err_type, count in error_types.items():
                print(f"   - {err_type}: {count} 次")
            # 显示前5个错误详情
            print(f"   - 详情 (前5个):")
            for err in self.results["errors"][:5]:
                print(f"     • {err}")

        print("\n" + "=" * 60)
        total_requests = len(self.results["query_time"]) + len(self.results["generate_time"]) + len(self.results["status_check_time"])
        success_rate = (total_requests / (total_requests + len(self.results["errors"]))) * 100 if (total_requests + len(self.results["errors"])) > 0 else 0
        print(f"📈 总体统计:")
        print(f"   - 总请求数: {total_requests + len(self.results['errors'])}")
        print(f"   - 成功数: {total_requests}")
        print(f"   - 失败数: {len(self.results['errors'])}")
        print(f"   - 成功率: {success_rate:.2f}%")
        print("=" * 60)


async def main():
    parser = argparse.ArgumentParser(description="API 压力测试")
    parser.add_argument("--url", default="http://localhost:8001", help="API 基础 URL")
    parser.add_argument("--concurrency", type=int, default=5, help="并发数")
    parser.add_argument("--duration", type=int, default=60, help="测试持续时间（秒）")
    parser.add_argument("--mode", choices=["mixed", "generate"], default="mixed", help="测试模式")
    
    args = parser.parse_args()
    
    print(f"""
╔════════════════════════════════════════════════════════════╗
║           🚀 API 压力测试工具                             ║
╚════════════════════════════════════════════════════════════╝
""")
    
    tester = LoadTester(base_url=args.url, concurrency=args.concurrency)
    
    try:
        if args.mode == "mixed":
            await tester.run_mixed_load_test(args.duration)
        else:  # generate
            await tester.run_generate_only_test()
    except KeyboardInterrupt:
        print("\n⏸️  测试被中断")
    except Exception as e:
        print(f"\n❌ 测试出错: {e}")
    
    tester.print_report()


if __name__ == "__main__":
    asyncio.run(main())
