#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
端到端测试脚本 - 验证任务从提交到完成的全流程
包括：提交 → 轮询状态 → 确认 completed → 验证输出文件
"""

import asyncio
import aiohttp
import time
import os
import sys
import tempfile
import argparse
from datetime import datetime

BASE_URL = "http://localhost:8001"
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TASKS_DIR = os.path.join(PROJECT_ROOT, "tasks")

# 测试用公共图片（小图，加速测试）
TEST_IMAGE_URL = "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=400"

POLL_INTERVAL = 3      # 秒：每次轮询间隔
TASK_TIMEOUT = 300     # 秒：单任务最长等待时间

# 缓存下载好的测试图片路径
_test_image_path: str = ""


async def ensure_test_image(session: aiohttp.ClientSession) -> str:
    """下载测试图片到临时文件，返回本地路径（只下载一次）"""
    global _test_image_path
    if _test_image_path and os.path.exists(_test_image_path):
        return _test_image_path

    print("  📥 下载测试图片...")
    try:
        async with session.get(TEST_IMAGE_URL, timeout=aiohttp.ClientTimeout(total=30)) as resp:
            if resp.status != 200:
                raise RuntimeError(f"下载测试图片失败: HTTP {resp.status}")
            data = await resp.read()

        # 写入临时文件（进程退出后自动清理）
        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        tmp.write(data)
        tmp.close()
        _test_image_path = tmp.name
        print(f"  ✅ 测试图片已缓存: {os.path.basename(_test_image_path)} ({len(data)//1024}KB)")
        return _test_image_path
    except Exception as e:
        raise RuntimeError(f"无法获取测试图片: {e}")


class E2ETestResult:
    def __init__(self, task_id: str, provider: str):
        self.task_id = task_id
        self.provider = provider
        self.submit_time = time.time()
        self.complete_time: float = 0
        self.final_status: str = "unknown"
        self.output_files: list = []
        self.error: str = ""
        self.passed: bool = False

    @property
    def elapsed(self) -> float:
        return self.complete_time - self.submit_time if self.complete_time else time.time() - self.submit_time


async def submit_task(session: aiohttp.ClientSession, provider: str, worker_id: int) -> tuple[bool, str]:
    """提交生成任务，返回 (success, task_id)"""
    # 下载测试图片（只下一次，复用）
    try:
        image_path = await ensure_test_image(session)
    except RuntimeError as e:
        return False, str(e)

    data = aiohttp.FormData()
    data.add_field("prompt", f"E2E测试 #{worker_id} provider={provider or 'default'} @ {datetime.now().strftime('%H:%M:%S')}")
    data.add_field("aspect_ratio", "3:4")
    data.add_field("art_style", "flux_realistic")
    data.add_field("auto_upscale", "false")
    if provider:
        data.add_field("provider", provider)
    # 上传真实图片文件（seedream 系列要求本地文件，不接受纯 URL）
    with open(image_path, "rb") as img_f:
        data.add_field("files", img_f, filename="test_input.jpg", content_type="image/jpeg")

        try:
            async with session.post(
                f"{BASE_URL}/generate-async/",
                data=data,
                timeout=aiohttp.ClientTimeout(total=60)
            ) as resp:
                body = await resp.json()
                if resp.status == 200:
                    return True, body.get("task_id", "")
                else:
                    return False, f"HTTP {resp.status}: {body.get('error', '')}"
        except Exception as e:
            return False, str(e)


async def poll_until_done(session: aiohttp.ClientSession, task_id: str) -> tuple[str, list, str]:
    """轮询任务状态直到 completed/failed/cancelled 或超时
    返回 (final_status, output_files, error_msg)
    """
    deadline = time.time() + TASK_TIMEOUT
    last_status = "unknown"
    last_progress = 0

    while time.time() < deadline:
        try:
            async with session.get(
                f"{BASE_URL}/task-status/{task_id}",
                timeout=aiohttp.ClientTimeout(total=10)
            ) as resp:
                if resp.status == 404:
                    await asyncio.sleep(POLL_INTERVAL)
                    continue
                data = await resp.json()
                status = data.get("status", "unknown")
                progress = data.get("progress", 0)

                if status != last_status or progress != last_progress:
                    print(f"      [{task_id[:12]}] 状态: {status} | 进度: {progress}%")
                    last_status = status
                    last_progress = progress

                if status == "completed":
                    files = data.get("files", [])
                    return "completed", files, ""

                if status in ("failed", "cancelled"):
                    error = data.get("error", "未知错误")
                    return status, [], error

        except Exception as e:
            print(f"      [{task_id[:12]}] 轮询出错: {e}")

        await asyncio.sleep(POLL_INTERVAL)

    return "timeout", [], f"超过 {TASK_TIMEOUT}s 未完成"


async def verify_output_files(session: aiohttp.ClientSession, task_id: str, output_files: list) -> tuple[bool, str]:
    """通过 HTTP 验证输出文件存在且大小正常（文件由服务通过 /taskfile/ 端点提供）"""
    # 过滤出图片文件（排除输入图片和 watermark 占位文件）
    img_files = [
        f for f in output_files
        if any(f.lower().endswith(ext) for ext in (".png", ".jpg", ".jpeg", ".webp"))
        and "input" not in os.path.basename(f).lower()
        and "watermark" not in os.path.basename(f).lower()
    ]
    if not img_files:
        # 没有 output_files 时尝试构建路径查询服务端
        img_files = output_files  # 全部验证

    if not img_files:
        return False, "API 未返回任何输出文件"

    bad = []
    ok_count = 0
    for file_path in img_files:
        # file_path 形如 /taskfile/{task_id}/filename.png，直接拼接为 HTTP URL
        if file_path.startswith("/"):
            url = f"{BASE_URL}{file_path}"
        else:
            url = file_path
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                if resp.status != 200:
                    bad.append(f"{os.path.basename(file_path)}: HTTP {resp.status}")
                    continue
                # content-length 不一定有，读少量字节判断文件是否有内容
                size = int(resp.headers.get("content-length", 0))
                if size == 0:
                    chunk = await resp.content.read(2048)
                    size = len(chunk)
                if size < 1024:
                    bad.append(f"{os.path.basename(file_path)}: 大小仅 {size}B")
                    continue
                ok_count += 1
        except Exception as e:
            bad.append(f"{os.path.basename(file_path)}: {e}")

    if bad:
        return False, f"文件验证失败: {bad}"
    return True, f"{ok_count} 个输出文件验证通过"


async def run_single_e2e(session: aiohttp.ClientSession, provider: str, worker_id: int) -> E2ETestResult:
    """运行单个端到端任务测试"""
    result = E2ETestResult(task_id="", provider=provider or "default")

    # 1. 提交任务
    print(f"\n  [{worker_id}] 提交任务 (provider={provider or 'default'})...")
    ok, task_id_or_err = await submit_task(session, provider, worker_id)
    if not ok:
        result.error = f"提交失败: {task_id_or_err}"
        result.complete_time = time.time()
        result.final_status = "submit_failed"
        print(f"  [{worker_id}] ❌ {result.error}")
        return result

    result.task_id = task_id_or_err
    print(f"  [{worker_id}] ✅ 任务已提交: {task_id_or_err}")

    # 2. 轮询直到完成
    print(f"  [{worker_id}] ⏳ 等待任务完成 (超时 {TASK_TIMEOUT}s)...")
    final_status, output_files, error = await poll_until_done(session, result.task_id)
    result.complete_time = time.time()
    result.final_status = final_status
    result.output_files = output_files
    result.error = error

    # 3. 验证输出文件
    if final_status == "completed":
        file_ok, file_msg = await verify_output_files(session, result.task_id, output_files)
        if file_ok:
            result.passed = True
            print(f"  [{worker_id}] ✅ 完成! 耗时 {result.elapsed:.1f}s | {file_msg}")
        else:
            result.error = f"文件验证失败: {file_msg}"
            print(f"  [{worker_id}] ❌ 任务完成但文件有问题: {file_msg}")
    else:
        print(f"  [{worker_id}] ❌ 任务 {final_status} | {error}")

    return result


async def run_e2e_tests(providers: list, concurrency: int):
    """运行端到端测试"""
    print(f"""
╔════════════════════════════════════════════════════════════╗
║           🔬 端到端完整流程测试                           ║
╚════════════════════════════════════════════════════════════╝
测试内容: 提交 → 后台处理 → 状态轮询 → 文件验证
并发数  : {concurrency}
Provider: {', '.join(providers) if providers else 'default'}
任务超时: {TASK_TIMEOUT}s
""")

    connector = aiohttp.TCPConnector(limit=50)
    timeout = aiohttp.ClientTimeout(total=TASK_TIMEOUT + 60)

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        # 先验证服务可用
        try:
            async with session.get(f"{BASE_URL}/styles/", timeout=aiohttp.ClientTimeout(total=5)) as r:
                if r.status != 200:
                    print("❌ 服务不可用，请先启动服务")
                    return
        except Exception as e:
            print(f"❌ 无法连接服务: {e}")
            return

        print("✅ 服务连接正常，开始测试...\n")

        # 构造任务列表
        jobs = []
        for i in range(concurrency):
            provider = providers[i % len(providers)] if providers else ""
            jobs.append(run_single_e2e(session, provider, i + 1))

        results: list[E2ETestResult] = await asyncio.gather(*jobs)

    # 打印报告
    print(f"\n{'=' * 60}")
    print("📊 端到端测试报告")
    print("=" * 60)

    passed = [r for r in results if r.passed]
    failed = [r for r in results if not r.passed]

    for r in results:
        icon = "✅" if r.passed else "❌"
        task_short = r.task_id[:16] if r.task_id else "N/A"
        elapsed = f"{r.elapsed:.1f}s"
        files_count = len(r.output_files)
        detail = f"{files_count} 文件" if r.passed else r.error
        print(f"  {icon} [{r.provider:20s}] {task_short}... | {elapsed} | {detail}")

    print(f"\n{'=' * 60}")
    print(f"📈 结果汇总:")
    print(f"   总任务数 : {len(results)}")
    print(f"   成功数   : {len(passed)}")
    print(f"   失败数   : {len(failed)}")
    print(f"   成功率   : {len(passed)/len(results)*100:.1f}%")

    if passed:
        times = [r.elapsed for r in passed]
        avg = sum(times) / len(times)
        print(f"   平均耗时 : {avg:.1f}s")
        print(f"   最快     : {min(times):.1f}s")
        print(f"   最慢     : {max(times):.1f}s")

    if failed:
        print(f"\n⚠️  失败详情:")
        for r in failed:
            print(f"   - [{r.provider}] {r.task_id[:16] or 'N/A'}: {r.error}")

    overall_pass = len(failed) == 0
    status_icon = "🟢" if overall_pass else "🔴"
    status_text = "全部通过" if overall_pass else f"{len(failed)} 个任务失败"
    print(f"\n{status_icon} 整体状态: {status_text}")
    print("=" * 60)

    return overall_pass


def main():
    parser = argparse.ArgumentParser(description="端到端完整流程测试")
    parser.add_argument("--providers", nargs="*", default=[],
                        help="要测试的 provider 列表，留空则使用默认 provider")
    parser.add_argument("--concurrency", type=int, default=1,
                        help="并发任务数 (默认 1，建议先单任务测试)")
    parser.add_argument("--timeout", type=int, default=300,
                        help="单任务超时秒数 (默认 300)")
    args = parser.parse_args()

    global TASK_TIMEOUT
    TASK_TIMEOUT = args.timeout

    success = asyncio.run(run_e2e_tests(args.providers, args.concurrency))
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
