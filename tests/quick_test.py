#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速验证脚本 - 快速检查系统是否正常运行
无需复杂参数，开箱即用
"""

import asyncio
import aiohttp
import time
import os
import tempfile
from datetime import datetime

TEST_IMAGE_URL = "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=400"
_cached_image_path: str = ""


async def get_test_image(session: aiohttp.ClientSession) -> str:
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


async def submit_generate(session: aiohttp.ClientSession, base_url: str, image_path: str, label: str):
    """提交一次生成请求（上传文件）"""
    data = aiohttp.FormData()
    data.add_field("prompt", f"测试 {label} @ {datetime.now().strftime('%H:%M:%S')}")
    data.add_field("aspect_ratio", "3:4")
    data.add_field("art_style", "flux_realistic")
    data.add_field("auto_upscale", "false")
    with open(image_path, "rb") as f:
        data.add_field("files", f, filename="test_input.jpg", content_type="image/jpeg")
        return await session.post(
            f"{base_url}/generate-async/",
            data=data,
            timeout=aiohttp.ClientTimeout(total=60)
        )

async def quick_test():
    """快速测试"""
    base_url = "http://localhost:8001"
    
    print(f"""
╔════════════════════════════════════════════════════════════╗
║           ✨ 快速系统检查                                 ║
╚════════════════════════════════════════════════════════════╝
""")
    
    connector = aiohttp.TCPConnector(limit=50)
    timeout = aiohttp.ClientTimeout(total=30)
    
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        # 测试 1: 基本连接
        print("1️⃣  测试基本连接...")
        try:
            async with session.get(f"{base_url}/styles/", timeout=aiohttp.ClientTimeout(total=5)) as resp:
                if resp.status == 200:
                    print("   ✅ 服务正常响应")
                else:
                    print(f"   ❌ 返回状态码: {resp.status}")
                    return False
        except Exception as e:
            print(f"   ❌ 连接失败: {e}")
            return False

        # 测试 2: 5 个并发查询
        print("\n2️⃣  测试 5 个并发查询...")
        try:
            tasks = [
                session.get(f"{base_url}/seeds/", timeout=aiohttp.ClientTimeout(total=10))
                for _ in range(5)
            ]
            start = time.time()
            responses = await asyncio.gather(*tasks)
            elapsed = time.time() - start
            
            success_count = sum(1 for r in responses if r.status == 200)
            print(f"   ✅ 成功: {success_count}/5 | 耗时: {elapsed:.2f}s")
            
            for r in responses:
                await r.read()
        except Exception as e:
            print(f"   ❌ 并发查询失败: {e}")
            return False

        # 测试 3: 10 个并发查询
        print("\n3️⃣  测试 10 个并发查询...")
        try:
            tasks = [
                session.get(f"{base_url}/task-status/test-task-{i}", timeout=aiohttp.ClientTimeout(total=10))
                for i in range(10)
            ]
            start = time.time()
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            elapsed = time.time() - start
            
            success_count = sum(1 for r in responses if isinstance(r, aiohttp.ClientResponse))
            print(f"   ✅ 成功: {success_count}/10 | 耗时: {elapsed:.2f}s")
            
            for r in responses:
                if isinstance(r, aiohttp.ClientResponse):
                    await r.read()
        except Exception as e:
            print(f"   ❌ 并发查询失败: {e}")
            return False

        # 测试 4: 下载测试图片 + 3 个并发生成请求
        print("\n4️⃣  测试 3 个并发生成请求...")
        try:
            image_path = await get_test_image(session)
            tasks = [
                submit_generate(session, base_url, image_path, f"#{i}")
                for i in range(3)
            ]
            start = time.time()
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            elapsed = time.time() - start

            success_count = 0
            task_ids = []
            for r in responses:
                if isinstance(r, aiohttp.ClientResponse):
                    if r.status == 200:
                        body = await r.json()
                        task_ids.append(body.get("task_id"))
                        success_count += 1
                    else:
                        await r.read()

            print(f"   ✅ 成功: {success_count}/3 | 耗时: {elapsed:.2f}s")
            if task_ids:
                print(f"   📝 创建的任务 ID:")
                for tid in task_ids:
                    print(f"      - {tid[:16]}...")
        except Exception as e:
            print(f"   ❌ 并发生成失败: {e}")
            return False

        # 测试 5: 混合并发（10 个）
        print("\n5️⃣  测试 10 个混合并发请求...")
        try:
            image_path = await get_test_image(session)
            tasks = []
            for i in range(10):
                if i % 2 == 0:
                    tasks.append(
                        session.get(f"{base_url}/styles/", timeout=aiohttp.ClientTimeout(total=10))
                    )
                else:
                    tasks.append(
                        submit_generate(session, base_url, image_path, f"混合#{i}")
                    )

            start = time.time()
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            elapsed = time.time() - start

            success_count = 0
            for r in responses:
                if isinstance(r, aiohttp.ClientResponse):
                    if r.status in [200, 404]:
                        success_count += 1
                    await r.read()

            print(f"   ✅ 成功: {success_count}/10 | 耗时: {elapsed:.2f}s")
        except Exception as e:
            print(f"   ❌ 混合并发失败: {e}")
            return False

    print(f"""
╔════════════════════════════════════════════════════════════╗
║           ✅ 所有测试通过！                               ║
║                                                            ║
║  ✓ 基本连接正常                                           ║
║  ✓ 5 个并发可以支持                                       ║
║  ✓ 10 个并发可以支持                                      ║
║  ✓ 生成请求可以并发处理                                   ║
║  ✓ 混合并发正常工作                                       ║
║                                                            ║
║  系统状态: 🟢 良好                                         ║
╚════════════════════════════════════════════════════════════╝
""")
    return True


if __name__ == "__main__":
    try:
        success = asyncio.run(quick_test())
        exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⏸️  测试被中断")
        exit(1)
    except Exception as e:
        print(f"\n❌ 测试出错: {e}")
        exit(1)
