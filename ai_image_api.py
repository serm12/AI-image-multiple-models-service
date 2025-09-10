import os
# import replicate  # 移除：生图流程改为统一API客户端
import requests
import re
import json
import asyncio
import aiofiles
import httpx
from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import FileResponse, JSONResponse
from datetime import datetime

# 导入配置和核心模块
from config import (
    initialize_config, DirectoryConfig, APIConfig, AppConfig,
    FluxModelEnum, AspectRatioEnum, OutputFormatEnum, ArtStyleEnum,
    STYLE_PROMPTS
)
from core import get_style_description, create_response_record
from utils.utils_watermark import create_logo_watermark, add_logo_watermark
from utils.utils_task import generate_task_dir, save_params, generate_output_filenames
from utils.utils_upscale import (
    upscale_image_with_replicate, 
    download_upscaled_image, 
    generate_upscale_filename,
    save_upscale_task_info,
    create_upscale_lookup_folder,
    get_upscale_models_info
)

# 异步组件导入
from utils.async_task_manager import task_manager, TaskStatus

# 导入统一API客户端
from utils.unified_api_client import api_client

from fastapi.middleware.cors import CORSMiddleware

# 初始化配置
initialize_config()

app = FastAPI(
    title="AI Image Generation API",
    description="AI图像生成API服务 - 支持多种模型，异步高并发处理，支持20个并发能力",
    version="2.0.0"
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=AppConfig.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 设置环境变量
os.environ["REPLICATE_API_TOKEN"] = APIConfig.REPLICATE_API_TOKEN

# 轻量提示：根据API服务提供商打印相关信息
if APIConfig.IMAGE_GENERATION_PROVIDER == "flux_bfl" and not APIConfig.BFL_API_KEY:
    print("⚠️ 未检测到 BFL_API_KEY，生图调用将失败，请在 .env 中设置 BFL_API_KEY")
elif APIConfig.IMAGE_GENERATION_PROVIDER == "flux_replicate" and not APIConfig.REPLICATE_API_TOKEN:
    print("⚠️ 未检测到 REPLICATE_API_TOKEN，生图调用将失败，请在 .env 中设置 REPLICATE_API_TOKEN")
elif APIConfig.IMAGE_GENERATION_PROVIDER == "gemini_google" and not APIConfig.GOOGLE_GEMINI_API_KEY:
    print("⚠️ 未检测到 GOOGLE_GEMINI_API_KEY，生图调用将失败，请在 .env 中设置 GOOGLE_GEMINI_API_KEY")
elif APIConfig.IMAGE_GENERATION_PROVIDER == "gemini_replicate" and not APIConfig.REPLICATE_API_TOKEN:
    print("⚠️ 未检测到 REPLICATE_API_TOKEN，生图调用将失败，请在 .env 中设置 REPLICATE_API_TOKEN")

# 种子管理函数 - 从tasks目录读取
def get_last_seed_from_tasks():
    """从最近的生图任务中获取seed"""
    try:
        if not os.path.exists(DirectoryConfig.TASKS_DIR):
            return None
        
        # 获取所有任务目录，按时间排序
        task_dirs = []
        for item in os.listdir(DirectoryConfig.TASKS_DIR):
            task_path = os.path.join(DirectoryConfig.TASKS_DIR, item)
            if os.path.isdir(task_path):
                params_file = os.path.join(task_path, "params.json")
                if os.path.exists(params_file):
                    task_dirs.append((item, task_path))
        
        # 按任务ID（包含时间戳）倒序排列
        task_dirs.sort(reverse=True)
        
        # 查找最近的生图任务（不是放大任务）
        for task_id, task_path in task_dirs:
            params_file = os.path.join(task_path, "params.json")
            try:
                with open(params_file, 'r', encoding='utf-8') as f:
                    params = json.load(f)
                
                # 检查是否是生图任务（有art_style字段）而不是放大任务（有model字段）
                if 'art_style' in params and 'model' not in params:
                    return params.get('extracted_seed') or params.get('input_seed')
            except:
                continue
            
            return None
    except Exception as e:
        print(f"获取最后seed时出错: {e}")
        return None
    
def get_all_seeds_from_tasks():
    """从所有生图任务中获取seed列表"""
    try:
        seeds = []
        if not os.path.exists(DirectoryConfig.TASKS_DIR):
            return {"seeds": [], "count": 0}
        
        # 获取所有任务目录
        for item in os.listdir(DirectoryConfig.TASKS_DIR):
            task_path = os.path.join(DirectoryConfig.TASKS_DIR, item)
            if os.path.isdir(task_path):
                params_file = os.path.join(task_path, "params.json")
                if os.path.exists(params_file):
                    try:
                        with open(params_file, 'r', encoding='utf-8') as f:
                            params = json.load(f)
                        
                        # 只处理生图任务，不处理放大任务
                        if 'art_style' in params and 'model' not in params:
                            seed = params.get('extracted_seed') or params.get('input_seed')
                            if seed:
                                seeds.append({
                                    "seed": seed,
                                    "task_id": params.get('task_id', item),
                                    "description": params.get('description', ''),
                                    "art_style": params.get('art_style', ''),
                                    "created_at": params.get('time', ''),
                                    "prompt": params.get('original_prompt', '')[:100] + "..." if len(params.get('original_prompt', '')) > 100 else params.get('original_prompt', ''),
                                    "input_seed": params.get('input_seed'),  # 用户输入的seed
                                    "extracted_seed": params.get('extracted_seed')  # 从响应中提取的真实seed
                                })
                    except:
                        continue
        
        # 按时间倒序排列
        seeds.sort(key=lambda x: x['created_at'], reverse=True)
        
        return {"seeds": seeds, "count": len(seeds)}
    except Exception as e:
        print(f"获取seeds列表时出错: {e}")
        return {"seeds": [], "count": 0}

# ==================== 异步高并发端点 ====================

@app.post("/generate-async/")
async def generate_image_async(
    prompt: str = Form(...),
    aspect_ratio: AspectRatioEnum = Form(AspectRatioEnum.ratio_3_4),
    output_format: OutputFormatEnum = Form(OutputFormatEnum.png),
    model_version: str = Form(...),  # 兼容前端参数
    art_style: ArtStyleEnum = Form(ArtStyleEnum.flux_realistic),
    seed: str = Form(None),  # 接受字符串类型的seed
    use_last_seed: bool = Form(False),
    description: str = Form(""),
    auto_upscale: bool = Form(False),
    upscale_face_enhance: bool = Form(False),
    input_image_url: str = Form(None),
    files: list[UploadFile] = File([])
):
    """异步图像生成API - 立即返回任务ID，支持20个并发处理"""
    
    # -- 参数兼容性处理 --
    # 1. model_version: 前端发送此参数，但后端使用 flux_model_variant。我们固定为 pro。
    flux_model_variant = FluxModelEnum.pro
    
    # 2. seed: 前端可能发送空字符串，需转换为 int 或 None
    try:
        seed_int = int(seed) if seed and seed.isdigit() else None
    except (ValueError, TypeError):
        seed_int = None
    
    # 1. 快速创建任务目录
    task_id, task_dir, timestamp = generate_task_dir(DirectoryConfig.TASKS_DIR)
    
    try:
        input_image_paths = []
        input_filenames = []
        
        # 2. 异步保存上传文件（可选）
        if files:
            for file in files:
                input_filename = file.filename
                input_image_path = os.path.join(task_dir, input_filename)
                async with aiofiles.open(input_image_path, "wb") as f:
                    content = await file.read()
                    await f.write(content)
                input_image_paths.append(input_image_path)
                input_filenames.append(input_filename)
        
        # 允许仅传 URL 情况（无文件）
        if not input_image_paths and not input_image_url:
            raise ValueError("必须提供 files 或 input_image_url 其中之一")
        
        # 3. 处理seed参数
        final_seed = seed_int
        if use_last_seed:
            last_seed = get_last_seed_from_tasks()
            if last_seed:
                final_seed = last_seed
        
        # 4. 准备参数
        style_prompt = STYLE_PROMPTS.get(art_style, "")
        final_prompt = style_prompt + prompt
        
        # 生成一个用于文件命名的安全名称（URL 情况）
        if not input_filenames and input_image_url:
            try:
                from urllib.parse import urlparse
                parsed = urlparse(input_image_url)
                base = os.path.basename(parsed.path) or "url_input.jpg"
                input_filenames.append(base)
            except Exception:
                input_filenames.append("url_input.jpg")
        
        params = {
            "prompt": final_prompt,
            "original_prompt": prompt,
            "art_style": art_style.value,
            "aspect_ratio": aspect_ratio.value,
            "output_format": output_format.value,
            "flux_model_variant": flux_model_variant.value,
            "input_images": input_filenames,
            "input_image_url": input_image_url,
            "task_id": task_id,
            "time": timestamp,
            "input_seed": final_seed,
            "use_last_seed": use_last_seed,
            "description": description,
            "auto_upscale": auto_upscale,
            "upscale_face_enhance": upscale_face_enhance,
            "api_provider": APIConfig.IMAGE_GENERATION_PROVIDER  # 记录使用的API服务提供商
        }
        
        # 5. 异步保存参数
        async with aiofiles.open(os.path.join(task_dir, "params.json"), "w") as f:
            await f.write(json.dumps(params, ensure_ascii=False, indent=2))
        
        # 6. 创建任务记录
        task_manager.create_task(
            task_id,
            prompt=final_prompt,
            flux_model_variant=flux_model_variant.value,
            estimated_time="60-180秒"
        )
        
        # 7. 启动后台处理任务（不等待）
        # 注意：process_generation_background 需要适配多图路径
        asyncio.create_task(
            process_generation_background(
                task_id, task_dir, final_prompt, input_image_paths,
                flux_model_variant, aspect_ratio, output_format, final_seed, params
            )
        )
        
        # 8. 立即返回任务ID
        return JSONResponse({
            "task_id": task_id,
            "status": "submitted",
            "message": "任务已提交，正在后台处理",
            "estimated_time": "60-180秒",
            "status_url": f"/task-status/{task_id}",
            "created_at": timestamp,
            "api_provider": APIConfig.IMAGE_GENERATION_PROVIDER,  # 返回使用的API服务提供商
            "concurrent_improvement": "✅ 支持20个并发处理，不会阻塞其他请求"
        })
        
    except Exception as e:
        task_manager.set_task_failed(task_id, str(e))
        return JSONResponse({
            "error": f"任务创建失败: {str(e)}",
            "task_id": task_id
        }, status_code=500)

async def process_generation_background(
    task_id: str, task_dir: str, prompt: str, input_image_paths: list[str],
    flux_model_variant, aspect_ratio, output_format, seed: int, params: dict
):
    """后台异步处理生成任务"""
    
    try:
        # 使用并发限制执行
        await task_manager.execute_with_concurrency_limit(
            task_id,
            _do_generation_work(
                task_id, task_dir, prompt, input_image_paths,
                flux_model_variant, aspect_ratio, output_format, seed, params
            )
        )
    except Exception as e:
        print(f"❌ 后台任务 {task_id} 失败: {e}")
        task_manager.set_task_failed(task_id, str(e))

async def _do_generation_work(
    task_id: str, task_dir: str, prompt: str, input_image_paths: list[str],
    flux_model_variant, aspect_ratio, output_format, seed: int, params: dict
):
    """实际的生成工作"""
    import json  # 确保json模块在函数作用域中可用
    
    # 更新状态：开始处理
    task_manager.update_task(task_id, status=TaskStatus.PROCESSING, progress=10)
    
    # 允许 URL 作为输入
    input_image_url = params.get("input_image_url")
    
    # 创建预测
    task_manager.update_task(task_id, status=TaskStatus.GENERATING, progress=30)
    
    try:
        # 使用统一API客户端生成图像
        # 注意：api_client.generate_image 需要适配多图路径
        result = await api_client.generate_image(
            prompt=prompt,
            input_image_paths=input_image_paths if input_image_paths else None,
            input_image_url=input_image_url,
            flux_model_variant=flux_model_variant,
            aspect_ratio=aspect_ratio,
            output_format=output_format,
            seed=seed,
            art_style=params.get("art_style"),  # 传递艺术风格参数
            task_id=task_id  # 传递任务ID
        )
        
        # 处理seed逻辑差异 - Gemini不返回实际使用的seed
        extracted_seed = result.get("extracted_seed")
        if APIConfig.IMAGE_GENERATION_PROVIDER in ["gemini_google", "gemini_replicate"]:
            # Gemini自己生成seed，我们记录用户输入的seed
            extracted_seed = seed
        
        # 如果存在 output_for_json 字段，则用它替换 output 字段以进行保存
        result_to_save = result.copy()
        if "output_for_json" in result_to_save:
            result_to_save["output"] = result_to_save.pop("output_for_json")

        # 保存API响应到JSON文件
        api_type = result.get("api_type", "unknown")
        response_filename = f"{api_type}_response.json"
        response_file = os.path.join(task_dir, response_filename)
        async with aiofiles.open(response_file, 'w', encoding='utf-8') as f:
            await f.write(json.dumps(result_to_save, ensure_ascii=False, indent=2))
        
        # 更新params.json，添加API相关信息和seed信息
        try:
            params_path = os.path.join(task_dir, "params.json")
            try:
                with open(params_path, 'r', encoding='utf-8') as pf:
                    existing_params = json.load(pf)
            except Exception:
                existing_params = {}
            existing_params[f"{api_type}_id"] = result.get("id")
            existing_params["stage"] = "submitted"
            # 保存seed信息
            if extracted_seed is not None:
                existing_params["extracted_seed"] = extracted_seed
            with open(params_path, 'w', encoding='utf-8') as pf:
                json.dump(existing_params, pf, ensure_ascii=False, indent=2)
        except Exception as werr:
            print(f"⚠️ 写入API信息失败: {werr}")
        
    except Exception as e:
        task_manager.set_task_failed(task_id, f"API调用失败: {e}")
        return
    
    if result["status"] == "succeeded":
        # 异步下载和处理图片
        task_manager.update_task(task_id, status=TaskStatus.DOWNLOADING, progress=70)
        
        image_url = str(result["output"])
        # 使用第一个输入图像的文件名作为基础
        main_input_filename = params.get("input_images")[0] if params.get("input_images") else "output.png"
        await download_and_process_image_async_optimized(task_id, task_dir, image_url, main_input_filename)
        
        # 水印处理完成后，最终更新75%状态包含完整文件列表
        task_manager.update_task(task_id, 
            status=TaskStatus.PROCESSING, 
            progress=75,
            result={
                "image_url": image_url,
                "output_files": await get_output_files_async(task_dir, task_id)
            }
        )
        print(f"✅ 75% - 水印版本完成，完整文件列表已更新")
        
        # 创建主图像生成的定位文件夹
        prediction_id = result.get("id")
        if prediction_id:
            main_task_lookup_dir = f"{task_dir}_{prediction_id}"
            try:
                if not os.path.exists(main_task_lookup_dir):
                    os.makedirs(main_task_lookup_dir)
                    print(f"📁 创建主任务定位文件夹: {os.path.basename(main_task_lookup_dir)}")
                else:
                    print(f"📁 主任务定位文件夹已存在: {os.path.basename(main_task_lookup_dir)}")
            except Exception as e:
                print(f"⚠️ 创建主任务定位文件夹失败: {e}")
        
        # 🚀 关键优化：立即完成普通图片任务，放大处理移到后台
        task_manager.set_task_completed(task_id, {
            "image_url": image_url,
            "output_files": await get_output_files_async(task_dir, task_id),
            "stage": "regular_completed"
        })
        print(f"✅ 普通图片任务已完成，前端可立即使用")
        
        # 自动进行2倍放大处理（如果用户选择启用）- 后台异步处理
        auto_upscale_value = params.get("auto_upscale", False)
        print(f"🔍 检查auto_upscale参数: {auto_upscale_value} (类型: {type(auto_upscale_value)})")
        print(f"🔍 完整params: {params}")
        
        if auto_upscale_value:
            print("🔄 开始后台异步2倍放大处理...")
            asyncio.create_task(
                process_upscale_background(task_id, task_dir, params, image_url)
            )
            print("🚀 放大任务已启动后台处理，不影响用户体验")
        else:
            print("❌ auto_upscale为False，跳过放大处理")
        
        return
    else:
        error_msg = result.get("error", "未知错误")
        task_manager.set_task_failed(task_id, error_msg)

async def download_and_process_image_async(task_id: str, task_dir: str, image_url: str, filename: str):
    """异步下载和处理图片"""
    
    # 生成文件名
    OUTPUT_FILE, ORIGINAL_FILE, WATERMARK_FILE = generate_output_filenames(
        task_dir, filename, "png"
    )
    
    # 异步下载图片
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.get(image_url)
        response.raise_for_status()
        
        # 异步保存原图（现在OUTPUT_FILE和ORIGINAL_FILE指向同一个文件）
        async with aiofiles.open(ORIGINAL_FILE, "wb") as f:
            await f.write(response.content)
    
    # 在线程池中处理水印（避免阻塞事件循环）
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, create_logo_watermark)
    await loop.run_in_executor(None, add_logo_watermark, ORIGINAL_FILE, WATERMARK_FILE)

async def download_and_process_image_async_optimized(task_id: str, task_dir: str, image_url: str, filename: str):
    """优化版本：分阶段更新状态的异步下载和处理图片"""
    
    # 生成文件名
    OUTPUT_FILE, ORIGINAL_FILE, WATERMARK_FILE = generate_output_filenames(
        task_dir, filename, "png"
    )
    
    content_bytes = None
    if isinstance(image_url, str) and image_url.startswith("data:image/"):
        # 处理 data URL
        import base64
        try:
            header, b64data = image_url.split(",", 1)
            content_bytes = base64.b64decode(b64data)
        except Exception as e:
            raise ValueError(f"无法解析 data URL: {e}")
    
    if content_bytes is None:
        # 异步下载图片
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.get(image_url)
            response.raise_for_status()
            content_bytes = response.content
    
    # 异步保存原图（现在OUTPUT_FILE和ORIGINAL_FILE指向同一个文件）
    async with aiofiles.open(ORIGINAL_FILE, "wb") as f:
        await f.write(content_bytes)
    
    # 🚀 关键优化：原图保存完成后立即更新状态 (72%)，让前端能立即获取文件
    task_manager.update_task(task_id, 
        status=TaskStatus.PROCESSING, 
        progress=72,
        result={
            "image_url": image_url,
            "output_files": await get_output_files_async(task_dir, task_id)
        }
    )
    
    # 在线程池中处理水印（避免阻塞事件循环）
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, create_logo_watermark)
    await loop.run_in_executor(None, add_logo_watermark, ORIGINAL_FILE, WATERMARK_FILE)

async def process_upscale_background(task_id: str, task_dir: str, params: dict, image_url: str):
    """后台处理放大任务，不阻塞主流程"""
    try:
        print(f"🚀 开始后台放大处理 - 任务ID: {task_id}")
        
        # 修正：从 'input_images' 列表中获取主输入文件名
        main_input_filename = params.get("input_images")[0] if params.get("input_images") else "reference.png"
        
        # 获取原图路径
        OUTPUT_FILE, ORIGINAL_FILE, WATERMARK_FILE = generate_output_filenames(
            task_dir, main_input_filename, "png"
        )
        
        # 调用放大功能
        upscale_success, upscale_message, upscale_result = await upscale_image_with_replicate(
            ORIGINAL_FILE,  # 使用原图进行放大
            "upscaler",  # 使用upscaler模型
            2,  # 2倍放大
            params.get("upscale_face_enhance", False)  # 使用用户选择的面部增强设置
        )
        
        if upscale_success:
            print("✅ 后台放大成功!")
            
            # 下载放大后的图片
            upscale_output_url = upscale_result["output_url"]
            
            # 使用算法值生成文件名
            from utils.utils_upscale import generate_upscale_filename
            upscale_filename = generate_upscale_filename("output_reference.png", 2)
            upscale_output_path = os.path.join(task_dir, upscale_filename)
            
            if download_upscaled_image(upscale_output_url, upscale_output_path):
                print(f"📄 放大图片已保存: {upscale_output_path}")
                
                # 为放大图片添加水印（水印文件不包含算法值）
                upscaled_watermark_path = os.path.join(task_dir, "output_reference_upscaled_2x_watermark.png")
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, add_logo_watermark, upscale_output_path, upscaled_watermark_path)
                print(f"📄 放大图片水印版本已保存: {upscaled_watermark_path}")
                
                # 保存放大任务信息 - 修正：使用 main_input_filename
                save_upscale_task_info(
                    task_dir, task_id, params.get("time", ""), "upscaler", 2, 
                    params.get("upscale_face_enhance", False), main_input_filename, upscale_result["full_response"]
                )
                
                # 创建放大任务定位文件夹
                prediction_id = upscale_result.get("prediction_id")
                if prediction_id:
                    create_upscale_lookup_folder(task_dir, prediction_id)
                
                print(f"✅ 后台放大任务完成: {task_id}")
            else:
                print("❌ 下载放大图片失败")
        else:
            print(f"❌ 后台放大失败: {upscale_message}")
            
    except Exception as e:
        print(f"⚠️ 后台放大处理出错: {e}")

async def get_output_files_async(task_dir: str, task_id: str) -> list:
    """获取输出文件列表"""
    files = []
    try:
        for filename in os.listdir(task_dir):
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                files.append(f"/taskfile/{task_id}/{filename}")
    except Exception:
        pass
    return files

@app.get("/task-status/{task_id}")
async def get_task_status_async(task_id: str):
    """异步获取任务状态"""
    
    task_info = task_manager.get_task(task_id)
    if not task_info:
        return JSONResponse({"error": "任务不存在"}, status_code=404)
    
    # 构建基础响应
    response_data = {
        "task_id": task_id,
        "status": task_info["status"],
        "progress": task_info["progress"],
        "created_at": task_info["created_at"],
        "updated_at": task_info["updated_at"],
        "error": task_info.get("error"),
        "prediction_id": task_info.get("prediction_id"),  # 当前任务的ID
        "api_provider": APIConfig.IMAGE_GENERATION_PROVIDER  # 返回使用的API服务提供商
    }
    
    # 检查是否有文件（即使任务还在处理中）
    task_dir = os.path.join(DirectoryConfig.TASKS_DIR, task_id)
    if os.path.exists(task_dir):
        output_files = await get_output_files_async(task_dir, task_id)
        if output_files:
            response_data["files"] = output_files
    
    # 如果任务完成，添加额外信息
    if task_info["status"] == TaskStatus.COMPLETED and task_info.get("result"):
        response_data.update({
            "completed_at": task_info["updated_at"],
            "main_prediction_id": task_info.get("prediction_id"),  # 主图像生成ID
            "upscale_prediction_id": task_info.get("upscale_prediction_id")  # 放大任务ID
        })
    
    return JSONResponse(response_data)

@app.get("/system-stats")
async def get_system_stats():
    """获取系统统计信息"""
    stats = task_manager.get_statistics()
    return JSONResponse({
        "message": "AI Image Generation API - 异步并发统计",
        "concurrent_capacity": "20个同时处理",
        "performance_improvement": "2000%+ 相比阻塞版本",
        "api_version": "2.0.0",
        "api_provider": APIConfig.IMAGE_GENERATION_PROVIDER,  # 返回当前使用的API服务提供商
        **stats
    })

# ==================== 原有端点保持不变 ====================

@app.get("/taskfile/{task_id}/{filename}")
def get_task_file(task_id: str, filename: str, request: Request):
    task_dir = os.path.join(DirectoryConfig.TASKS_DIR, task_id)
    file_path = os.path.join(task_dir, filename)
    
    if os.path.exists(file_path):
        # 记录图片访问
        ip_address = request.client.host
        referer = request.headers.get("referer", "N/A")
        
        log_content = (
            f"Image: {filename}\n"
            f"Accessed At: {datetime.now()}\n"
            f"Client IP: {ip_address}\n"
            f"Referer: {referer}\n"
        )

        # 检查水印图片
        if 'watermark' in filename and filename.endswith('.png'):
            md_path = os.path.join(task_dir, 'watermark.md')
            with open(md_path, 'w', encoding='utf-8') as f:
                f.write(log_content)
        # 检查原图或高清放大图
        elif (filename.startswith('output_cropped_original_') or '_upscaled_' in filename) and filename.endswith('.png'):
            md_path = os.path.join(task_dir, 'original.md')
            with open(md_path, 'w', encoding='utf-8') as f:
                f.write(log_content)

        # 自动判断图片类型
        if filename.lower().endswith((".jpg", ".jpeg")):
            media_type = "image/jpeg"
        elif filename.lower().endswith(".png"):
            media_type = "image/png"
        elif filename.lower().endswith(".json"):
            media_type = "application/json"
        else:
            media_type = "application/octet-stream"
        
        # 使用 StreamingResponse 避免 Content-Length 计算问题
        def iterfile():
            with open(file_path, "rb") as f:
                yield from f
        
        from fastapi.responses import StreamingResponse
        return StreamingResponse(
            iterfile(),
            media_type=media_type,
            headers={
                "Cache-Control": "public, max-age=3600",
                "Content-Disposition": f"inline; filename={filename}"
            }
        )
    return JSONResponse({"error": "文件不存在"}, status_code=404)


@app.get("/styles/")
def get_art_styles():
    """获取所有可用的绘画风格（去重）"""
    styles = []
    seen_values = set()  # 用于去重
    
    for style in ArtStyleEnum:
        style_value = style.value
        
        # 跳过已经见过的值（去重）
        if style_value in seen_values:
            continue
        
        seen_values.add(style_value)
        
        style_info = {
            "value": style_value,
            "name": style_value.replace("_", " ").title(),
            "description": get_style_description(style_value)
        }
        styles.append(style_info)
    
    # 按风格类型排序：先Flux，再Gemini，最后其他
    def sort_key(style):
        value = style["value"]
        if value.startswith("flux_"):
            return (0, value)
        elif value.startswith("gemini_"):
            return (1, value)
        else:
            return (2, value)
    
    styles.sort(key=sort_key)
    return JSONResponse({"styles": styles})

@app.get("/seeds/")
def get_all_seeds():
    """获取所有记录的 seed（从生图任务中读取）"""
    return JSONResponse(get_all_seeds_from_tasks())

@app.get("/seeds/last")
def get_last_seed():
    """获取最后使用的 seed（从生图任务中读取）"""
    last_seed = get_last_seed_from_tasks()
    if last_seed:
        return JSONResponse({"seed": last_seed})
    else:
        return JSONResponse({"seed": None, "message": "没有找到最后使用的 seed"})

@app.get("/seeds/search/{description}")
def search_seed_by_description(description: str):
    """根据描述搜索 seed（从生图任务中搜索）"""
    seeds_data = get_all_seeds_from_tasks()
    
    # 在描述中搜索
    for seed_info in seeds_data["seeds"]:
        if description.lower() in seed_info["description"].lower():
            return JSONResponse({
                "seed": seed_info["seed"],
                "task_id": seed_info["task_id"],
                "description": seed_info["description"],
                "art_style": seed_info["art_style"]
            })
    
        return JSONResponse({"seed": None, "description": description, "message": "未找到匹配的 seed"})

@app.get("/upscale/models")
def get_upscale_models():
    """获取支持的放大模型信息"""
    return JSONResponse(get_upscale_models_info())

@app.get("/tasks/")
def list_all_tasks():
    """列出所有任务"""
    try:
        tasks = []
        if os.path.exists(DirectoryConfig.TASKS_DIR):
            for task_id in os.listdir(DirectoryConfig.TASKS_DIR):
                task_dir = os.path.join(DirectoryConfig.TASKS_DIR, task_id)
                if os.path.isdir(task_dir):
                    # 读取参数文件
                    params_file = os.path.join(task_dir, "params.json")
                    params = {}
                    if os.path.exists(params_file):
                        with open(params_file, 'r', encoding='utf-8') as f:
                            params = json.load(f)
                    
                    # 读取响应文件（支持多种API模式）
                    response = {}
                    api_response_files = ["bfl_response.json", "replicate_response.json"]
                    for response_file in api_response_files:
                        response_file_path = os.path.join(task_dir, response_file)
                        if os.path.exists(response_file_path):
                            with open(response_file_path, 'r', encoding='utf-8') as f:
                                response = json.load(f)
                                break
                    
                    # 检查输出文件
                    output_files = []
                    for filename in os.listdir(task_dir):
                        if filename.endswith(('.png', '.jpg', '.jpeg')):
                            output_files.append(filename)
                    
                    tasks.append({
                        "task_id": task_id,
                        "description": params.get('description', ''),
                        "created_at": params.get('time', ''),
                        "extracted_seed": response.get('extracted_seed'),
                        "output_files_count": len(output_files),
                        "status": response.get('status', 'unknown'),
                        "api_provider": params.get('api_provider', APIConfig.IMAGE_GENERATION_PROVIDER)
                    })
        
        # 按时间倒序排列
        tasks.sort(key=lambda x: x['created_at'], reverse=True)
        
        return JSONResponse({
            "tasks": tasks,
            "total": len(tasks)
        })
        
    except Exception as e:
        return JSONResponse({"error": f"获取任务列表失败: {str(e)}"}, status_code=500)

@app.post("/upscale/")
async def upscale_image(
    file: UploadFile = File(...),
    model: str = Form("upscaler"),  # 默认使用real-esrgan
    scale: int = Form(2),  # 默认2倍放大
    face_enhance: bool = Form(False)  # 是否启用面部增强（仅real-esrgan模型支持）
):
    """图片放大API
    
    支持的模型:
    - real-esrgan: 支持face_enhance参数，集成GFPGAN面部增强
    - gfpgan: 专门的面部增强模型
    - upscaler: 放大模型
    """
    
    try:
        # 创建任务目录
        task_id, task_dir, timestamp = generate_task_dir(DirectoryConfig.TASKS_DIR)
        
        # 保存上传图片
        input_image_path = os.path.join(task_dir, file.filename)
        with open(input_image_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # 调用放大工具模块
        success, message, result_data = await upscale_image_with_replicate(
            input_image_path, model, scale, face_enhance
        )
        
        if not success:
            return JSONResponse({"error": message}, status_code=500)
        
        # 下载放大后的图片
        output_url = result_data["output_url"]
        full_response = result_data["full_response"]
        prediction_id = result_data.get("prediction_id")
        
        # 生成输出文件名
        output_filename = generate_upscale_filename(file.filename, scale)
        output_path = os.path.join(task_dir, output_filename)
        
        # 下载图片
        if download_upscaled_image(output_url, output_path):
            print(f"✅ 图片放大完成: {output_filename}")
            
            # 保存任务信息
            save_upscale_task_info(
                task_dir, task_id, timestamp, model, scale, 
                face_enhance, file.filename, full_response
            )
            
            # 创建定位文件夹
            create_upscale_lookup_folder(task_dir, prediction_id)
            
            return JSONResponse({
                "task_id": task_id,
                "original_image": f"/taskfile/{task_id}/{file.filename}",
                "upscaled_image": f"/taskfile/{task_id}/{output_filename}",
                "model": model,
                "scale": scale,
                "face_enhance": face_enhance,
                "original_url": output_url
            })
        else:
            return JSONResponse({"error": "下载放大图片失败"}, status_code=500)
            
    except Exception as e:
        return JSONResponse({"error": f"放大图片时出错: {str(e)}"}, status_code=500)

@app.get("/task/{task_id}")
def get_task_info(task_id: str):
    """获取特定任务的详细信息"""
    task_dir = os.path.join(DirectoryConfig.TASKS_DIR, task_id)
    if not os.path.exists(task_dir):
        return JSONResponse({"error": "任务不存在"}, status_code=404)
    
    try:
        # 读取主任务参数文件
        params_file = os.path.join(task_dir, "params.json")
        main_params = {}
        if os.path.exists(params_file):
            with open(params_file, 'r', encoding='utf-8') as f:
                main_params = json.load(f)
        
        # 读取完整的远程API响应文件（支持多种API模式）
        api_responses = {}
        api_response_files = ["bfl_response.json", "replicate_response.json"]
        for response_file in api_response_files:
            response_file_path = os.path.join(task_dir, response_file)
            if os.path.exists(response_file_path):
                with open(response_file_path, 'r', encoding='utf-8') as f:
                    api_responses[response_file] = json.load(f)
        
        # 检查输出文件
        output_files = []
        for filename in os.listdir(task_dir):
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                output_files.append({
                    "filename": filename,
                    "url": f"/taskfile/{task_id}/{filename}"
                })
        
        # 构建响应
        response_data = {
            "task_id": task_id,
            "params": main_params,  # 用户提交的参数和主要返回数据
            "api_responses": api_responses,  # 完整的远程API响应
            "output_files": output_files
        }
        
        return JSONResponse(response_data)
        
    except Exception as e:
        return JSONResponse({"error": f"读取任务信息失败: {str(e)}"}, status_code=500)

@app.get("/token-usage/")
def get_token_usage():
    """获取Gemini Token使用记录"""
    import json
    
    log_file = "gemini_token_usage.json"
    if os.path.exists(log_file):
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                logs = json.load(f)
            
            # 计算统计信息
            if logs:
                total_tokens = sum(log.get('total_tokens', 0) for log in logs)
                total_cost = sum(log.get('estimated_cost', 0) for log in logs)
                total_requests = len(logs)
                avg_tokens = total_tokens / total_requests if total_requests > 0 else 0
                
                stats = {
                    "total_requests": total_requests,
                    "total_tokens": total_tokens,
                    "total_cost": round(total_cost, 6),
                    "avg_tokens_per_request": round(avg_tokens, 1)
                }
                
                return JSONResponse({
                    "success": True,
                    "stats": stats,
                    "recent_logs": logs[-10:],  # 最近10条记录
                    "all_logs": logs
                })
            else:
                return JSONResponse({
                    "success": True,
                    "stats": {"total_requests": 0, "total_tokens": 0, "total_cost": 0, "avg_tokens_per_request": 0},
                    "recent_logs": [],
                    "all_logs": []
                })
                
        except Exception as e:
            return JSONResponse({"error": f"读取Token记录失败: {e}"}, status_code=500)
    else:
        return JSONResponse({
            "success": True,
            "message": "暂无Token使用记录",
            "stats": {"total_requests": 0, "total_tokens": 0, "total_cost": 0, "avg_tokens_per_request": 0},
            "recent_logs": [],
            "all_logs": []
        })

# 根据API服务提供商提供特定的端点
if APIConfig.IMAGE_GENERATION_PROVIDER == "flux_bfl":
    @app.post("/resume-bfl/{task_id}")
    async def resume_bfl(task_id: str):
        """根据保存的 bfl_polling_url 继续轮询并把图片落地（断点续跑）。"""
        task_dir = os.path.join(DirectoryConfig.TASKS_DIR, task_id)
        if not os.path.exists(task_dir):
            return JSONResponse({"error": "任务目录不存在"}, status_code=404)
        params_path = os.path.join(task_dir, "params.json")
        try:
            with open(params_path, 'r', encoding='utf-8') as pf:
                params = json.load(pf)
        except Exception as e:
            return JSONResponse({"error": f"读取参数失败: {e}"}, status_code=500)
        polling_url = params.get("bfl_polling_url")
        if not polling_url:
            return JSONResponse({"error": "无 bfl_polling_url 可恢复"}, status_code=400)
        # 若文件已存在，直接返回
        output_file, original_file, watermark_file = generate_output_filenames(task_dir, params.get("input_image", "output.png"), "png")
        if os.path.exists(original_file):
            return JSONResponse({"message": "文件已存在", "files": [f"/taskfile/{task_id}/{os.path.basename(original_file)}"]})
        # 轮询
        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                output_url = None
                raw_final = None
                waited = 0
                interval = 2
                max_wait = 300
                headers = {"x-key": APIConfig.BFL_API_KEY or ""}
                while waited <= max_wait:
                    resp = await client.get(polling_url, headers=headers)
                    resp.raise_for_status()
                    data = resp.json()
                    raw_final = data
                    # 提取 URL 或 data URL
                    candidate = None
                    for key in ["url", "image_url", "output", "result", "data", "images"]:
                        val = data.get(key)
                        if isinstance(val, str) and (val.startswith("http") or val.startswith("data:image/")):
                            candidate = val
                            break
                        if isinstance(val, dict):
                            for k2 in ["url", "image", "image_url", "sample"]:
                                v2 = val.get(k2)
                                if isinstance(v2, str) and (v2.startswith("http") or v2.startswith("data:image/")):
                                    candidate = v2
                                    break
                            if candidate:
                                break
                        if isinstance(val, list) and val and isinstance(val[0], str) and (val[0].startswith("http") or val[0].startswith("data:image/")):
                            candidate = val[0]
                            break
                    if candidate:
                        output_url = candidate
                        break
                    status_val = str(data.get("status") or data.get("state") or "").lower()
                    if status_val in ["failed", "error", "canceled", "cancelled"]:
                        return JSONResponse({"error": f"BFL 任务失败: {data}"}, status_code=500)
                    await asyncio.sleep(interval)
                    waited += interval
                if not output_url:
                    return JSONResponse({"error": "轮询超时或未返回图片 URL"}, status_code=504)
                # 下载并保存
                await download_and_process_image_async_optimized(task_id, task_dir, output_url, params.get("input_image", "output.png"))
                # 保存响应
                try:
                    with open(os.path.join(task_dir, "bfl_response.json"), 'w', encoding='utf-8') as rf:
                        json.dump(raw_final, rf, ensure_ascii=False, indent=2)
                    params["bfl_output_url"] = output_url
                    params["stage"] = "succeeded"
                    with open(params_path, 'w', encoding='utf-8') as pf:
                        json.dump(params, pf, ensure_ascii=False, indent=2)
                except Exception as werr:
                    print(f"⚠️ 恢复时保存BFL响应失败: {werr}")
                return JSONResponse({
                    "message": "恢复完成",
                    "image_url": output_url,
                    "files": await get_output_files_async(task_dir, task_id)
                })
        except Exception as e:
            return JSONResponse({"error": f"恢复失败: {e}"}, status_code=500)