import os
# import replicate  # 移除：生图流程改为统一API客户端
import re
import json
import asyncio
import aiofiles
import httpx
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, Form, Request, Depends, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from datetime import datetime

# 导入配置和核心模块
from config import (
    initialize_config, DirectoryConfig, APIConfig, AppConfig,
    FluxModelEnum, AspectRatioEnum, OutputFormatEnum, ArtStyleEnum, ProviderEnum,
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
from utils.face_detection import contains_human

# 异步组件导入
from utils.async_task_manager import task_manager, TaskStatus

# 导入统一API客户端
from utils.unified_api_client import api_client

# 进程级标记：水印logo是否已生成，避免每次任务都进线程池检查
_watermark_logo_created: bool = False
# 保存后台任务引用，防止 GC 过早回收未完成的 Task
_background_tasks: set = set()
_UPLOAD_CHUNK_SIZE = 1024 * 1024

from fastapi.middleware.cors import CORSMiddleware

# 初始化配置
initialize_config()


async def _periodic_task_cleanup():
    """每 5 分钟清理一次超过 24 小时的已完成/失败任务，防止内存持续增长"""
    while True:
        try:
            await asyncio.sleep(300)  # 缩短为 5 分钟，更及时清理
            removed = task_manager.cleanup_old_tasks(max_age_hours=24)
            if removed:
                print(f"🧹 定期清理：已移除 {removed} 个过期任务记录")
        except Exception as e:
            # 不让清理任务程序崩溃，睡会继续运行
            print(f"⚠️ 任务清理出错: {e}")
            continue


@asynccontextmanager
async def app_lifespan(app: FastAPI):
    # 配置 HTTP 连接池：keepalive 连接数=10，总连接数=20（支持并发）
    limits = httpx.Limits(max_keepalive_connections=10, max_connections=20)
    app.state.http_client = httpx.AsyncClient(timeout=60.0, limits=limits)
    app.state.long_http_client = httpx.AsyncClient(timeout=300.0, limits=limits)
    cleanup_task = asyncio.create_task(_periodic_task_cleanup())
    try:
        yield
    finally:
        cleanup_task.cancel()
        await app.state.http_client.aclose()
        await app.state.long_http_client.aclose()


app = FastAPI(
    title="AI Image Generation API",
    description="AI图像生成API服务 - 支持多种模型，异步高并发处理，支持20个并发能力",
    version="2.0.0",
    lifespan=app_lifespan
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=AppConfig.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _get_bearer_token(authorization: str | None) -> str | None:
    if not authorization:
        return None
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        return None
    return token.strip()


async def require_admin_api_key(request: Request):
    """Protect administrative endpoints when ADMIN_API_KEY is configured."""
    if not AppConfig.ADMIN_API_KEY:
        return
    provided_key = request.headers.get("x-api-key") or _get_bearer_token(request.headers.get("authorization"))
    if provided_key != AppConfig.ADMIN_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing admin API key")


def _safe_upload_filename(filename: str | None) -> str:
    safe_name = os.path.basename(filename or "upload")
    return safe_name or "upload"


async def save_validated_upload(file: UploadFile, destination: str) -> int:
    """Validate and save an uploaded image without loading the whole file into memory."""
    content_type = (file.content_type or "").split(";", 1)[0].strip().lower()
    if content_type not in AppConfig.ALLOWED_UPLOAD_CONTENT_TYPES:
        raise ValueError(f"不支持的文件类型: {file.content_type or 'unknown'}")

    max_bytes = AppConfig.MAX_UPLOAD_FILE_MB * 1024 * 1024
    total_bytes = 0
    async with aiofiles.open(destination, "wb") as f:
        while True:
            chunk = await file.read(_UPLOAD_CHUNK_SIZE)
            if not chunk:
                break
            total_bytes += len(chunk)
            if total_bytes > max_bytes:
                raise ValueError(f"文件过大，单文件不能超过 {AppConfig.MAX_UPLOAD_FILE_MB}MB")
            await f.write(chunk)
    return total_bytes


def resolve_task_file_path(task_id: str, filename: str) -> str | None:
    task_dir = Path(DirectoryConfig.TASKS_DIR, task_id).resolve()
    file_path = Path(task_dir, filename).resolve()
    try:
        file_path.relative_to(task_dir)
    except ValueError:
        return None
    return str(file_path)


@app.get("/")
async def root():
    return JSONResponse({"service": "AI Image Generation API", "status": "ok", "docs": "/docs"})


@app.get("/health")
async def health_check():
    return JSONResponse({
        "status": "ok",
        "version": app.version,
        "api_provider": APIConfig.IMAGE_GENERATION_PROVIDER,
        "configured_providers": len([p for p in APIConfig.get_available_providers() if p["configured"]]),
    })

# 设置环境变量
os.environ["REPLICATE_API_TOKEN"] = APIConfig.REPLICATE_API_TOKEN

# 轻量提示：根据API服务提供商打印相关信息
if APIConfig.IMAGE_GENERATION_PROVIDER == "flux_bfl" and not APIConfig.BFL_API_KEY:
    print("⚠️ 未检测到 BFL_API_KEY，生图调用将失败，请在 .env 中设置 BFL_API_KEY")
elif APIConfig.IMAGE_GENERATION_PROVIDER == "flux_replicate" and not APIConfig.REPLICATE_API_TOKEN:
    print("⚠️ 未检测到 REPLICATE_API_TOKEN，生图调用将失败，请在 .env 中设置 REPLICATE_API_TOKEN")
elif APIConfig.IMAGE_GENERATION_PROVIDER == "gemini-nanobanana_google" and not APIConfig.GOOGLE_GEMINI_API_KEY:
    print("⚠️ 未检测到 GOOGLE_GEMINI_API_KEY，生图调用将失败，请在 .env 中设置 GOOGLE_GEMINI_API_KEY")
elif APIConfig.IMAGE_GENERATION_PROVIDER == "gemini-nanobanana_replicate" and not APIConfig.REPLICATE_API_TOKEN:
    print("⚠️ 未检测到 REPLICATE_API_TOKEN，生图调用将失败，请在 .env 中设置 REPLICATE_API_TOKEN")

# 种子管理函数 - 从tasks目录读取
def get_last_seed_from_tasks():
    """从最近的生图任务中获取seed（优先读缓存文件，避免全目录扫描）"""
    try:
        cache_file = os.path.join(DirectoryConfig.TASKS_DIR, "_last_seed.txt")
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                val = f.read().strip()
            if val.isdigit():
                return int(val)

        # 缓存不存在时回退到目录扫描（兼容历史数据）
        if not os.path.exists(DirectoryConfig.TASKS_DIR):
            return None
        task_dirs = sorted(
            [
                (item, os.path.join(DirectoryConfig.TASKS_DIR, item))
                for item in os.listdir(DirectoryConfig.TASKS_DIR)
                if os.path.isdir(os.path.join(DirectoryConfig.TASKS_DIR, item))
                and os.path.exists(os.path.join(DirectoryConfig.TASKS_DIR, item, "params.json"))
            ],
            reverse=True
        )
        for task_id, task_path in task_dirs:
            try:
                with open(os.path.join(task_path, "params.json"), 'r', encoding='utf-8') as f:
                    params = json.load(f)
                if 'art_style' in params and 'model' not in params:
                    return params.get('extracted_seed') or params.get('input_seed')
            except:
                continue
        return None
    except Exception:
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
    except Exception:
        return {"seeds": [], "count": 0}

# ==================== 异步高并发端点 ====================

@app.post("/generate-async/")
async def generate_image_async(
    provider: ProviderEnum | None = Form(None),  # 覆盖默认服务提供商；Swagger 中显示为下拉选择
    prompt: str = Form(...),
    aspect_ratio: AspectRatioEnum = Form(AspectRatioEnum.ratio_3_4),
    output_format: OutputFormatEnum = Form(OutputFormatEnum.png),
    art_style: ArtStyleEnum = Form(ArtStyleEnum.flux_realistic),
    seed: str = Form(None),  # 接受字符串类型的seed
    use_last_seed: bool = Form(False),
    description: str = Form(""),
    auto_upscale: bool = Form(False),
    upscale_face_enhance: bool = Form(False),
    input_image_url: str = Form(None),
    width: int = Form(None),
    height: int = Form(None),
    size: str = Form("2K"),
    sequential_image_generation: str = Form("disabled"),
    enable_human_check: bool = Form(False),  # 是否开启真人检测，True时上传图片必须包含清晰人脸
    files: list[UploadFile] = File([])
):
    """异步图像生成API - 立即返回任务ID，支持1个并发处理"""
    
    # -- 参数兼容性处理 --
    # 1. flux_model_variant: 固定为 pro
    flux_model_variant = FluxModelEnum.pro

    # 2. provider: 确定本次请求实际使用的服务商并做运行时校验
    provider_value = provider.value if isinstance(provider, ProviderEnum) else provider
    effective_provider = (provider_value or "").strip() or APIConfig.IMAGE_GENERATION_PROVIDER
    try:
        APIConfig.validate_provider(effective_provider)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    
    # 3. seed: 前端可能发送空字符串，需转换为 int 或 None
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
            if len(files) > AppConfig.MAX_UPLOAD_FILES:
                raise ValueError(f"最多只能上传 {AppConfig.MAX_UPLOAD_FILES} 个文件")
            for file in files:
                input_filename = _safe_upload_filename(file.filename)
                input_image_path = os.path.join(task_dir, input_filename)
                await save_validated_upload(file, input_image_path)
                input_image_paths.append(input_image_path)
                input_filenames.append(input_filename)
        
        # 允许仅传 URL 情况（无文件）
        if not input_image_paths and not input_image_url:
            raise ValueError("必须提供 files 或 input_image_url 其中之一")
        
        # 2.5. 真人检测（仅当 enable_human_check=True 且有本地文件时执行）
        if enable_human_check and input_image_paths:
            loop = asyncio.get_event_loop()
            for img_path in input_image_paths:
                face_result = await loop.run_in_executor(None, contains_human, img_path)
                if not face_result["valid"]:
                    return JSONResponse(
                        {
                            "error": "human_check_failed",
                            "message": face_result["message"],
                            "task_id": task_id,
                        },
                        status_code=422,
                    )
        
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
            "width": width,
            "height": height,
            "size": size,
            "sequential_image_generation": sequential_image_generation,
            "api_provider": effective_provider  # 记录本次请求实际使用的服务提供商
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
        _t = asyncio.create_task(
            process_generation_background(
                task_id, task_dir, final_prompt, input_image_paths,
                flux_model_variant, aspect_ratio, output_format, final_seed, params,
                effective_provider
            )
        )
        _background_tasks.add(_t)
        _t.add_done_callback(_background_tasks.discard)
        
        # 8. 立即返回任务ID
        return JSONResponse({
            "task_id": task_id,
            "status": "submitted",
            "message": "任务已提交，正在后台处理",
            "estimated_time": "60-180秒",
            "status_url": f"/task-status/{task_id}",
            "created_at": timestamp,
            "api_provider": effective_provider,  # 返回本次请求实际使用的服务提供商
            "concurrent_improvement": "支持1个并发处理，内存优化版本"
        })
        
    except Exception as e:
        task_manager.set_task_failed(task_id, str(e))
        return JSONResponse({
            "error": f"任务创建失败: {str(e)}",
            "task_id": task_id
        }, status_code=500)


@app.get("/providers")
async def get_providers():
    """返回所有 provider 及其是否已配置 API Key。"""
    return JSONResponse({"providers": APIConfig.get_available_providers()})

async def process_generation_background(
    task_id: str, task_dir: str, prompt: str, input_image_paths: list[str],
    flux_model_variant, aspect_ratio, output_format, seed: int, params: dict,
    provider: str = None
):
    """后台异步处理生成任务"""
    
    try:
        # 使用并发限制执行
        await task_manager.execute_with_concurrency_limit(
            task_id,
            _do_generation_work(
                task_id, task_dir, prompt, input_image_paths,
                flux_model_variant, aspect_ratio, output_format, seed, params,
                provider
            )
        )
    except Exception as e:
        print(f"❌ 后台任务 {task_id} 失败: {e}")
        task_manager.set_task_failed(task_id, str(e))

async def _do_generation_work(
    task_id: str, task_dir: str, prompt: str, input_image_paths: list[str],
    flux_model_variant, aspect_ratio, output_format, seed: int, params: dict,
    provider: str = None
):
    """实际的生成工作"""
    
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
            width=params.get("width"),
            height=params.get("height"),
            size=params.get("size"),
            sequential_image_generation=params.get("sequential_image_generation"),
            task_id=task_id,  # 传递任务ID
            provider=provider
        )
        
        # 处理seed逻辑差异 - Gemini不返回实际使用的seed
        extracted_seed = result.get("extracted_seed")
        if (provider or APIConfig.IMAGE_GENERATION_PROVIDER) in ["gemini-nanobanana_google", "gemini-nanobanana_replicate"]:
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
        # 生产环境用紧凑 JSON 格式加快序列化
        async with aiofiles.open(response_file, 'w', encoding='utf-8') as f:
            await f.write(json.dumps(result_to_save, ensure_ascii=False, separators=(',', ':')))
        
        # 更新params.json，添加API相关信息和seed信息（合并为单次异步读写操作）
        try:
            loop = asyncio.get_running_loop()
            params_path = os.path.join(task_dir, "params.json")
            existing_params = {}
            
            # 异步读取
            try:
                async with aiofiles.open(params_path, 'r', encoding='utf-8') as pf:
                    content = await pf.read()
                    existing_params = json.loads(content)
            except Exception:
                pass
            
            # 批量更新
            existing_params.update({
                f"{api_type}_id": result.get("id"),
                "stage": "submitted",
                **({
                    "extracted_seed": extracted_seed
                } if extracted_seed is not None else {})
            })
            
            # 单次异步写入（紧凑格式）
            async with aiofiles.open(params_path, 'w', encoding='utf-8') as pf:
                await pf.write(json.dumps(existing_params, ensure_ascii=False, separators=(',', ':')))
            
            # 异步更新 last_seed 缓存
            if extracted_seed is not None:
                try:
                    cache_file = os.path.join(DirectoryConfig.TASKS_DIR, "_last_seed.txt")
                    async with aiofiles.open(cache_file, 'w') as cf:
                        await cf.write(str(extracted_seed))
                except Exception:
                    pass
        except Exception as werr:
            # 仅在真正失败时打印警告
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
        current_files = await get_output_files_async(task_dir, task_id)
        task_manager.update_task(task_id, 
            status=TaskStatus.PROCESSING, 
            progress=75,
            result={
                "image_url": image_url,
                "output_files": current_files
            }
        )
        
        # 创建主图像生成的定位文件夹（后台异步，不打印日志）
        prediction_id = result.get("id")
        if prediction_id:
            main_task_lookup_dir = f"{task_dir}_{prediction_id}"
            try:
                if not os.path.exists(main_task_lookup_dir):
                    os.makedirs(main_task_lookup_dir)
            except Exception:
                pass
        
        # 🚀 关键优化：立即完成普通图片任务，放大处理移到后台
        task_manager.set_task_completed(task_id, {
            "image_url": image_url,
            "output_files": current_files,
            "stage": "regular_completed"
        })
        
        # 自动进行2倍放大处理（如果用户选择启用）- 后台异步处理
        auto_upscale_value = params.get("auto_upscale", False)
        
        if auto_upscale_value:
            _t = asyncio.create_task(
                process_upscale_background(task_id, task_dir, params, image_url)
            )
            _background_tasks.add(_t)
            _t.add_done_callback(_background_tasks.discard)
        
        return
    else:
        error_msg = result.get("error", "未知错误")
        task_manager.set_task_failed(task_id, error_msg)

async def download_and_process_image_async_optimized(task_id: str, task_dir: str, image_url: str, filename: str):
    """优化版本：分阶段更新状态的异步下载和处理图片（最小化日志输出）"""
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
        try:
            response = await app.state.http_client.get(image_url)
            response.raise_for_status()
            content_bytes = response.content
        except Exception as e:
            raise ValueError(f"图片下载失败: {e}")
    
    # 异步保存原图
    try:
        async with aiofiles.open(ORIGINAL_FILE, "wb") as f:
            await f.write(content_bytes)
    except Exception as e:
        raise ValueError(f"原图保存失败: {e}")
    
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
    try:
        global _watermark_logo_created
        loop = asyncio.get_running_loop()
        if not _watermark_logo_created:
            # 在 await 前置位，防止并发任务同时通过检查重复创建（asyncio 单线程，此处无竞态）
            _watermark_logo_created = True
            await loop.run_in_executor(None, create_logo_watermark)
        await loop.run_in_executor(None, add_logo_watermark, ORIGINAL_FILE, WATERMARK_FILE)
    except Exception:
        # 水印失败不中断主流程
        pass

async def process_upscale_background(task_id: str, task_dir: str, params: dict, image_url: str):
    """后台处理放大任务，不阻塞主流程"""
    try:
        
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
            
            # 下载放大后的图片
            upscale_output_url = upscale_result["output_url"]
            
            # 使用算法值生成文件名
            from utils.utils_upscale import generate_upscale_filename
            upscale_filename = generate_upscale_filename("output_reference.png", 2)
            upscale_output_path = os.path.join(task_dir, upscale_filename)
            
            loop = asyncio.get_running_loop()
            dl_ok = await loop.run_in_executor(
                None, download_upscaled_image, upscale_output_url, upscale_output_path
            )
            if dl_ok:
                # 为放大图片添加水印（水印文件不包含算法值）
                upscaled_watermark_path = os.path.join(task_dir, "output_reference_upscaled_2x_watermark.png")
                await loop.run_in_executor(None, add_logo_watermark, upscale_output_path, upscaled_watermark_path)
                
                # 保存放大任务信息 - 修正：使用 main_input_filename
                save_upscale_task_info(
                    task_dir, task_id, params.get("time", ""), "upscaler", 2, 
                    params.get("upscale_face_enhance", False), main_input_filename, upscale_result["full_response"]
                )
                
                # 创建放大任务定位文件夹
                prediction_id = upscale_result.get("prediction_id")
                if prediction_id:
                    create_upscale_lookup_folder(task_dir, prediction_id)
    except Exception:
        pass

async def get_output_files_async(task_dir: str, task_id: str) -> list:
    """获取输出文件列表（异步执行，不阻塞事件循环）"""
    files = []
    try:
        loop = asyncio.get_running_loop()
        # 将 os.listdir 拷贝到线程池，避免阻塞事件循环
        filenames = await loop.run_in_executor(None, os.listdir, task_dir)
        for filename in filenames:
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                files.append(f"/taskfile/{task_id}/{filename}")
    except Exception:
        pass
    return files

@app.get("/task-status/{task_id}")
async def get_task_status_async(task_id: str):
    """异步获取任务状态"""
    from datetime import datetime
    
    task_info = task_manager.get_task(task_id)
    if not task_info:
        return JSONResponse({"error": "任务不存在"}, status_code=404)
    
    # 把 Unix 时间戳转换为 ISO 格式字符串（保持 API 兼容性）
    def ts_to_iso(ts):
        if isinstance(ts, (int, float)):
            return datetime.utcfromtimestamp(ts).isoformat() + "Z"
        return ts
    
    # 构建基础响应
    response_data = {
        "task_id": task_id,
        "status": task_info["status"],
        "progress": task_info["progress"],
        "created_at": ts_to_iso(task_info["created_at"]),
        "updated_at": ts_to_iso(task_info["updated_at"]),
        "error": task_info.get("error"),
        "prediction_id": task_info.get("prediction_id"),  # 当前任务的ID
        "api_provider": APIConfig.IMAGE_GENERATION_PROVIDER  # 返回使用的API服务提供商
    }
    # 检查是否有文件
    # 任务已完成时直接用内存缓存，避免每次轮询都 os.listdir
    cached_files = (task_info.get("result") or {}).get("output_files")
    if cached_files:
        response_data["files"] = cached_files
    else:
        task_dir = os.path.join(DirectoryConfig.TASKS_DIR, task_id)
        if os.path.exists(task_dir):
            output_files = await get_output_files_async(task_dir, task_id)
            if output_files:
                response_data["files"] = output_files
    
    # 如果任务完成，添加额外信息
    if task_info["status"] == TaskStatus.COMPLETED and task_info.get("result"):
        response_data.update({
            "completed_at": ts_to_iso(task_info["updated_at"]),
            "main_prediction_id": task_info.get("prediction_id"),  # 主图像生成ID
            "upscale_prediction_id": task_info.get("upscale_prediction_id")  # 放大任务ID
        })
    
    return JSONResponse(response_data)

@app.get("/system-stats", dependencies=[Depends(require_admin_api_key)])
async def get_system_stats():
    """获取系统统计信息"""
    stats = task_manager.get_statistics()
    return JSONResponse({
        "message": "AI Image Generation API - 异步并发统计",
        "concurrent_capacity": "1个同时处理",
        "performance_improvement": "2000%+ 相比阻塞版本",
        "api_version": "2.0.0",
        "api_provider": APIConfig.IMAGE_GENERATION_PROVIDER,  # 返回当前使用的API服务提供商
        **stats
    })

# ==================== 原有端点保持不变 ====================

@app.get("/taskfile/{task_id}/{filename}")
def get_task_file(task_id: str, filename: str, request: Request):
    file_path = resolve_task_file_path(task_id, filename)
    
    if file_path and os.path.exists(file_path):
        # 跳过每次访问都写日志的同步 IO 操作，改为仅记录在内存中
        # 如需完整日志，建议用异步日志系统

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
        elif value.startswith("seedream-4_"):
            return (2, value)
        else:
            return (3, value)
    
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


@app.post("/api/check-photo")
@app.post("/check-photo/")
async def check_photo(
    file: UploadFile = File(...),
    client_city: str = Form("")
):
    """执行即时人脸检测，并保存上传照片到当前任务目录。"""
    task_id, task_dir, timestamp = generate_task_dir(DirectoryConfig.TASKS_DIR)
    original_name = os.path.basename(file.filename or "upload.jpg")
    if not original_name:
        original_name = "upload.jpg"

    input_path = os.path.join(task_dir, original_name)

    try:
        async with aiofiles.open(input_path, "wb") as buffer:
            content = await file.read()
            await buffer.write(content)

        params = {
            "task_id": task_id,
            "time": timestamp,
            "task_type": "face_check",
            "input_images": [original_name],
            "client_city": client_city,
        }
        save_params(params, task_dir)

        face_check = contains_human(input_path)
        response_data = {
            "task_id": task_id,
            "status": "success" if face_check["valid"] else "error",
            "message": face_check["message"],
            "client_city": client_city,
            "user_photo_url": f"/taskfile/{task_id}/{original_name}",
        }

        if face_check.get("face"):
            response_data["face"] = {
                "x": face_check["face"][0],
                "y": face_check["face"][1],
                "w": face_check["face"][2],
                "h": face_check["face"][3],
            }
        if face_check.get("img_size"):
            response_data["image_size"] = {
                "width": face_check["img_size"][0],
                "height": face_check["img_size"][1],
            }

        return JSONResponse(response_data)
    except Exception as e:
        return JSONResponse(
            {
                "task_id": task_id,
                "status": "error",
                "message": f"人脸检测处理失败: {str(e)}",
                "client_city": client_city,
            },
            status_code=500,
        )

@app.get("/tasks/", dependencies=[Depends(require_admin_api_key)])
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

        # 异步保存上传图片
        input_filename = _safe_upload_filename(file.filename)
        input_image_path = os.path.join(task_dir, input_filename)
        await save_validated_upload(file, input_image_path)

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
        output_filename = generate_upscale_filename(input_filename, scale)
        output_path = os.path.join(task_dir, output_filename)
        
        # 下载图片
        if download_upscaled_image(output_url, output_path):
            print(f"✅ 图片放大完成: {output_filename}")
            
            # 保存任务信息
            save_upscale_task_info(
                task_dir, task_id, timestamp, model, scale, 
                face_enhance, input_filename, full_response
            )
            
            # 创建定位文件夹
            create_upscale_lookup_folder(task_dir, prediction_id)
            
            return JSONResponse({
                "task_id": task_id,
                "original_image": f"/taskfile/{task_id}/{input_filename}",
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

@app.get("/token-usage/", dependencies=[Depends(require_admin_api_key)])
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
            client = app.state.long_http_client
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

