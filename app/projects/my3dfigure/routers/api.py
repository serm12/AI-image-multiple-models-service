import os
import shutil
# import replicate  # 移除：生图流程改为统一API客户端
import json
import asyncio
import aiofiles
import time
from contextlib import asynccontextmanager
from fastapi import APIRouter, File, UploadFile, Form, Request, Depends
from fastapi.responses import JSONResponse

# 导入配置和核心模块
from app.projects.my3dfigure.core.config import (
    DirectoryConfig, APIConfig, AppConfig,
    FluxModelEnum, AspectRatioEnum, OutputFormatEnum, ArtStyleEnum, ProviderEnum,
    STYLE_PROMPTS
)
from app.core.version import APP_VERSION, get_version_info
from app.services.security import (
    get_request_client_ip,
    get_request_country,
    require_admin_api_key,
)
from app.projects.my3dfigure.services.task_files import (
    resolve_task_file_path,
    safe_upload_filename,
    save_validated_upload,
)
from app.projects.my3dfigure.services.task_storage import generate_task_dir, save_params, generate_output_filenames
from app.utils.time_utils import CHINA_TIMEZONE_NAME
from app.projects.my3dfigure.services.upscale_service import (
    upscale_image_with_replicate,
    download_upscaled_image,
    generate_upscale_filename,
    save_upscale_task_info,
    create_upscale_lookup_folder,
    get_upscale_models_info
)
from app.projects.my3dfigure.services.face_detection import (
    contains_human,
    get_usable_face_reference_boxes,
)
from app.projects.my3dfigure.services.generation_face_anchors import (
    create_face_identity_anchors,
)
from app.projects.my3dfigure.services.pet_detection import contains_single_pet
from app.projects.my3dfigure.services.direct_character_prompt_service import (
    build_my3d_connected_pair_prompt,
    build_my3d_direct_prompt,
    build_my3d_pet_prompt,
    MY3D_PROMPT_VERSION,
    MY3D_PET_PROMPT_VERSION,
)
from app.projects.my3dfigure.core.i18n import get_message, normalize_locale

# 异步组件导入
from app.projects.my3dfigure.services.async_task_manager import task_manager, TaskStatus

# 导入统一API客户端
from app.projects.my3dfigure.clients.unified_api_client import api_client
from app.services.runtime_state import get_http_client, get_long_http_client

# 保存后台任务引用，防止 GC 过早回收未完成的 Task
_background_tasks: set = set()


async def _periodic_my3dfigure_task_cleanup():
    """Keep My3dFigure task records bounded without coupling to the shared app loop."""
    while True:
        try:
            await asyncio.sleep(300)
            removed = task_manager.cleanup_old_tasks(max_age_hours=6)
            if removed:
                print(f"🧹 My3dFigure 定期清理：已移除 {removed} 个过期任务记录")
        except asyncio.CancelledError:
            break
        except Exception as exc:
            print(f"⚠️ My3dFigure 任务清理出错: {exc}")


@asynccontextmanager
async def my3dfigure_router_lifespan(_):
    cleanup_task = asyncio.create_task(_periodic_my3dfigure_task_cleanup())
    try:
        yield
    finally:
        cleanup_task.cancel()
        try:
            await cleanup_task
        except asyncio.CancelledError:
            pass


router = APIRouter(
    prefix="/api/my3dfigure",
    tags=["My3dFigure"],
    lifespan=my3dfigure_router_lifespan,
)


def _resolve_checked_upload(upload_id: str) -> tuple[str, str]:
    """Return the one validated source image saved by the private check route."""
    params_path = resolve_task_file_path(upload_id, "params.json")
    if not params_path or not os.path.isfile(params_path):
        raise ValueError("checked upload was not found")
    try:
        with open(params_path, "r", encoding="utf-8") as file:
            params = json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("checked upload metadata is unreadable") from exc
    if params.get("task_type") != "face_check":
        raise ValueError("upload is not a checked My3dFigure photo")
    filenames = params.get("input_images") or []
    if len(filenames) != 1:
        raise ValueError("checked upload has no unique source image")
    filename = safe_upload_filename(filenames[0])
    image_path = resolve_task_file_path(upload_id, filename)
    if not image_path or not os.path.isfile(image_path):
        raise ValueError("checked upload image was not found")
    return image_path, filename


def _get_checked_upload_validation_mode(upload_id: str) -> str:
    """Read the validated mode from the source task, never trust a new form value."""
    params_path = resolve_task_file_path(upload_id, "params.json")
    if not params_path or not os.path.isfile(params_path):
        raise ValueError("checked upload was not found")
    try:
        with open(params_path, "r", encoding="utf-8") as file:
            params = json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("checked upload metadata could not be read") from exc
    return "pet" if str(params.get("validation_mode", "human")).strip().lower() == "pet" else "human"


def _get_checked_upload_face_boxes(upload_id: str, expected_face_count: int) -> list[tuple[float, float, float, float]]:
    """Read the private face evidence captured by the successful check-photo task."""
    params_path = resolve_task_file_path(upload_id, "params.json")
    if not params_path or not os.path.isfile(params_path):
        return []
    try:
        with open(params_path, "r", encoding="utf-8") as file:
            params = json.load(file)
    except (OSError, json.JSONDecodeError):
        return []
    expected = 2 if expected_face_count == 2 else 1
    boxes = params.get("usable_face_boxes")
    if not isinstance(boxes, list) or len(boxes) != expected:
        return []
    normalized = []
    for box in boxes:
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            return []
        try:
            x, y, width, height = (float(value) for value in box)
        except (TypeError, ValueError):
            return []
        if width <= 0 or height <= 0:
            return []
        normalized.append((x, y, width, height))
    return normalized


def _link_or_copy_checked_upload(source_path: str, destination_path: str) -> None:
    """Reuse the upload without another browser transfer; copy only if linking fails."""
    try:
        os.link(source_path, destination_path)
    except OSError:
        shutil.copyfile(source_path, destination_path)







# ==================== 异步高并发端点 ====================

@router.post("/generate-async/")
async def generate_image_async(
    request: Request,
    provider: str | None = Form(None),  # 可选；空值回退到 IMAGE_GENERATION_PROVIDER
    prompt: str = Form(...),
    aspect_ratio: AspectRatioEnum = Form(AspectRatioEnum.ratio_3_4),
    output_format: OutputFormatEnum = Form(OutputFormatEnum.png),
    art_style: str | None = Form(None),
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
    enhance_prompt: bool = Form(True),
    enable_human_check: bool = Form(False),  # 是否开启真人检测，True时上传图片必须包含清晰人脸
    locale: str = Form(""),
    validation_profile: str = Form("strict"),
    validation_mode: str = Form("human"),
    prompt_profile: str = Form("my3d_character"),
    figure_composition: str = Form("single_person"),
    expected_face_count: int = Form(1),
    wardrobe_profile: str = Form(""),
    wardrobe_palette: str = Form(""),
    upper_garment_profile: str = Form(""),
    footwear_profile: str = Form(""),
    files: list[UploadFile] = File([]),
    upload_id: str = Form(""),
):
    """异步图像生成API - 立即返回任务ID，支持1个并发处理"""

    # -- 参数兼容性处理 --
    # 1. flux_model_variant: 固定为 pro
    flux_model_variant = FluxModelEnum.pro

    # 2. provider: 确定本次请求实际使用的服务商并做运行时校验
    try:
        effective_provider = APIConfig.resolve_provider(provider)
        APIConfig.validate_aspect_ratio(effective_provider, aspect_ratio)
    except ValueError as e:
        return JSONResponse({
            "code": "GENERATION_REQUEST_INVALID",
            "message": get_message("GENERATION_REQUEST_INVALID", locale),
        }, status_code=400)

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
        source_validation_mode = "pet" if str(validation_mode or "").strip().lower() == "pet" else "human"

        # 2. Save one browser upload, or reuse a source that check-photo has
        # already validated and stored. The latter prevents a second image
        # upload after the front end finishes its detection animation.
        if upload_id and files:
            return JSONResponse({
                "code": "GENERATION_REQUEST_INVALID",
                "message": get_message("GENERATION_REQUEST_INVALID", locale),
            }, status_code=400)
        if upload_id:
            try:
                source_path, input_filename = _resolve_checked_upload(upload_id)
                source_validation_mode = _get_checked_upload_validation_mode(upload_id)
            except ValueError:
                return JSONResponse({
                    "code": "GENERATION_INPUT_REQUIRED",
                    "message": get_message("GENERATION_INPUT_REQUIRED", locale),
                }, status_code=400)
            input_image_path = os.path.join(task_dir, input_filename)
            await asyncio.to_thread(_link_or_copy_checked_upload, source_path, input_image_path)
            input_image_paths.append(input_image_path)
            input_filenames.append(input_filename)
        elif files:
            if len(files) > AppConfig.MAX_UPLOAD_FILES:
                raise ValueError(f"最多只能上传 {AppConfig.MAX_UPLOAD_FILES} 个文件")
            for file in files:
                input_filename = safe_upload_filename(file.filename)
                input_image_path = os.path.join(task_dir, input_filename)
                await save_validated_upload(file, input_image_path)
                input_image_paths.append(input_image_path)
                input_filenames.append(input_filename)

        # aiapiroute GPT-image 系列支持纯文生图；其他 provider 仍要求文件或 URL 输入。
        allows_text_to_image = effective_provider in APIConfig.AIAPIROUTE_PROVIDER_MODEL_MAP
        if not input_image_paths and not input_image_url and not allows_text_to_image:
            return JSONResponse({
                "code": "GENERATION_INPUT_REQUIRED",
                "message": get_message("GENERATION_INPUT_REQUIRED", locale),
            }, status_code=400)

        # 2.5. 真人检测（仅当 enable_human_check=True 且有本地文件时执行）
        if enable_human_check and input_image_paths:
            loop = asyncio.get_event_loop()
            for img_path in input_image_paths:
                face_result = await loop.run_in_executor(
                    None,
                    contains_human,
                    img_path,
                    locale,
                    validation_profile,
                )
                if not face_result["valid"]:
                    return JSONResponse(
                        {
                            "code": face_result.get("code", "FACE_DETECTION_FAILED"),
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
        normalized_art_style = ""
        style_prompt = ""
        art_style_value = (art_style or "").strip()
        if art_style_value:
            try:
                recognized_art_style = ArtStyleEnum(art_style_value)
            except ValueError:
                recognized_art_style = None
            if recognized_art_style is not None:
                normalized_art_style = recognized_art_style.value
                style_prompt = STYLE_PROMPTS.get(recognized_art_style, "")
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

        normalized_figure_composition = (figure_composition or "single_person").strip().lower()
        if normalized_figure_composition not in {"single_person", "connected_pair"}:
            normalized_figure_composition = "single_person"
        normalized_expected_face_count = 2 if expected_face_count == 2 else 1
        # Defense in depth: a stale storefront field must not silently turn a
        # validated two-face connected request into a single-person prompt.
        if normalized_expected_face_count == 2 and normalized_figure_composition == "single_person":
            normalized_figure_composition = "connected_pair"

        params = {
            "prompt": final_prompt,
            "original_prompt": prompt,
            "art_style": normalized_art_style,
            "aspect_ratio": aspect_ratio.value,
            "output_format": output_format.value,
            "flux_model_variant": flux_model_variant.value,
            "input_images": input_filenames,
            "checked_upload_id": upload_id or None,
            "input_image_url": input_image_url,
            "task_id": task_id,
            "time": timestamp,
            "time_zone": CHINA_TIMEZONE_NAME,
            "input_seed": final_seed,
            "use_last_seed": use_last_seed,
            "description": description,
            "auto_upscale": auto_upscale,
            "upscale_face_enhance": upscale_face_enhance,
            "width": width,
            "height": height,
            "size": size,
            "sequential_image_generation": sequential_image_generation,
            "enhance_prompt": enhance_prompt,
            "locale": normalize_locale(locale),
            "validation_profile": validation_profile,
            "validation_mode": source_validation_mode,
            "prompt_profile": prompt_profile,
            "figure_composition": normalized_figure_composition,
            "expected_face_count": normalized_expected_face_count,
            "wardrobe_profile": wardrobe_profile,
            "wardrobe_palette": wardrobe_palette,
            "upper_garment_profile": upper_garment_profile,
            "footwear_profile": footwear_profile,
            "api_provider": effective_provider,  # 记录本次请求实际使用的服务提供商
            "request_url": str(request.url),
            "client_ip": get_request_client_ip(request),
            "client_country": get_request_country(request),
            "user_agent": request.headers.get("user-agent", "")[:500],
            "started_at_epoch": time.time(),
        }

        # 5. 异步保存参数
        async with aiofiles.open(os.path.join(task_dir, "params.json"), "w") as f:
            await f.write(json.dumps(params, ensure_ascii=False, indent=2))

        # 6. 创建任务记录
        task_manager.create_task(
            task_id,
            prompt=final_prompt,
            api_provider=effective_provider,
            locale=normalize_locale(locale),
            flux_model_variant=flux_model_variant.value,
            estimated_time="60-180秒"
        )

        # 7. 启动后台处理任务（不等待）
        # 注意：process_generation_background 需要适配多图路径
        background_task = asyncio.create_task(
            process_generation_background(
                task_id, task_dir, final_prompt, input_image_paths,
                flux_model_variant, aspect_ratio, output_format, final_seed, params,
                effective_provider
            )
        )
        _background_tasks.add(background_task)
        background_task.add_done_callback(_background_tasks.discard)

        # 8. 立即返回任务ID
        return JSONResponse({
            "task_id": task_id,
            "status": "submitted",
            "message": get_message("GENERATION_SUBMITTED", locale),
            "estimated_time": "60-180秒",
            "status_url": f"/api/my3dfigure/task-status/{task_id}",
            "created_at": timestamp,
            "api_provider": effective_provider,  # 返回本次请求实际使用的服务提供商
            "concurrent_improvement": "支持1个并发处理，内存优化版本"
        })

    except Exception as e:
        task_manager.set_task_failed(task_id, str(e))
        return JSONResponse({
            "code": "GENERATION_CREATE_FAILED",
            "message": get_message("GENERATION_CREATE_FAILED", locale),
            "task_id": task_id,
        }, status_code=500)





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
    finally:
        completed_at = time.time()
        started_at = params.get("started_at_epoch", completed_at)
        try:
            duration = max(0.0, completed_at - float(started_at))
        except (TypeError, ValueError):
            duration = 0.0
        params_path = os.path.join(task_dir, "params.json")
        try:
            async with aiofiles.open(params_path, "r", encoding="utf-8") as file:
                stored_params = json.loads(await file.read())
        except Exception:
            stored_params = dict(params)
        stored_params.update(
            {
                "completed_at_epoch": completed_at,
                "generation_duration_seconds": round(duration, 2),
            }
        )
        async with aiofiles.open(params_path, "w", encoding="utf-8") as file:
            await file.write(
                json.dumps(stored_params, ensure_ascii=False, separators=(",", ":"))
            )

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
        effective_prompt = prompt
        request_images = input_image_paths
        if params.get("prompt_profile") == "my3d_character":
            if provider not in {"gpt-image-2_aiapiroute", "gpt-image-2_fal"}:
                raise ValueError("my3d_character prompt profile requires a GPT Image 2 provider")
            if len(input_image_paths or []) != 1:
                raise ValueError("my3d_character prompt profile requires one uploaded reference image")

            # The browser upload is the single source of truth for both validation
            # and generation. Do not rotate, crop, resize, re-encode or select a
            # sub-panel before sending it to the project client.
            source_context = "general"
            identity_anchor_paths = []
            anchor_instructions = ""
            if params.get("validation_mode") != "pet":
                # Request form values are serialized as strings. Normalize before
                # selecting the validated-face anchor count so a double-person
                # request with "2" cannot be treated as a single-person request.
                expected_face_count = 2 if str(params.get("expected_face_count")).strip() == "2" else 1
                checked_upload_id = str(params.get("checked_upload_id") or "").strip()
                if checked_upload_id:
                    anchor_source_path, _ = _resolve_checked_upload(checked_upload_id)
                    face_boxes = _get_checked_upload_face_boxes(checked_upload_id, expected_face_count)
                    if not face_boxes:
                        face_boxes = await asyncio.to_thread(
                            get_usable_face_reference_boxes,
                            anchor_source_path,
                            expected_face_count,
                        )
                else:
                    anchor_source_path = input_image_paths[0]
                    face_boxes = await asyncio.to_thread(
                        get_usable_face_reference_boxes,
                        anchor_source_path,
                        expected_face_count,
                    )
                identity_anchor_paths = await asyncio.to_thread(
                    create_face_identity_anchors,
                    anchor_source_path,
                    face_boxes,
                    task_dir,
                )
                if len(identity_anchor_paths) != expected_face_count:
                    raise ValueError("Could not prepare identity anchors from the validated usable faces")
                request_images = [*input_image_paths, *identity_anchor_paths]
                anchor_instructions = (
                    f"\nIDENTITY ANCHORS — ABSOLUTE: Image 1 is the original upload and remains the only source "
                    f"for scene, pose, clothing, body and objects. Images 2 through {expected_face_count + 1} are "
                    f"the {expected_face_count} usable-face identity anchors extracted from Image 1. Render exactly "
                    f"one figure for each anchor face. Use anchor images only to lock facial identity, hairline and "
                    f"hairstyle; never copy clothing, pose, body, objects or background from an anchor. No person "
                    f"visible only in Image 1 without a matching anchor may appear in the output."
                )
            if params.get("validation_mode") == "pet":
                effective_prompt = build_my3d_pet_prompt()
                params["prompt_branch"] = "pet"
                params["prompt_version"] = MY3D_PET_PROMPT_VERSION
            elif params.get("figure_composition") == "connected_pair":
                effective_prompt = build_my3d_connected_pair_prompt(
                    params.get("wardrobe_profile"),
                    params.get("wardrobe_palette"),
                    upper_garment_profile=params.get("upper_garment_profile"),
                    footwear_profile=params.get("footwear_profile"),
                    source_context=source_context,
                )
                params["prompt_branch"] = "connected_pair"
            else:
                effective_prompt = build_my3d_direct_prompt(
                    params.get("wardrobe_profile"),
                    params.get("wardrobe_palette"),
                    upper_garment_profile=params.get("upper_garment_profile"),
                    footwear_profile=params.get("footwear_profile"),
                    source_context=source_context,
                )
                params["prompt_branch"] = source_context
            effective_prompt = f"{effective_prompt}{anchor_instructions}"
            params["generation_mode"] = "direct_no_gemini"
            params.setdefault("prompt_version", MY3D_PROMPT_VERSION)
            params["effective_prompt"] = effective_prompt
            params["reference_transport"] = "uploaded_master_direct"
            params["identity_anchor_count"] = len(identity_anchor_paths)
            params["identity_anchor_files"] = [os.path.basename(path) for path in identity_anchor_paths]
            params["identity_anchor_source"] = "validated_upload" if params.get("checked_upload_id") else "task_upload"
            async with aiofiles.open(
                os.path.join(task_dir, "params.json"), "w", encoding="utf-8"
            ) as file:
                await file.write(json.dumps(params, ensure_ascii=False, indent=2))

        # 使用统一API客户端生成图像
        # 注意：api_client.generate_image 需要适配多图路径
        result = await api_client.generate_image(
            prompt=effective_prompt,
            input_image_paths=request_images if request_images else None,
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
            enhance_prompt=params.get("enhance_prompt", True),
            task_id=task_id,  # 传递任务ID
            provider=provider
        )
        actual_provider = result.get("api_provider", provider)
        params["api_provider"] = actual_provider
        params["provider_fallback_chain"] = result.get("provider_fallback_chain", [actual_provider])
        params["provider_fallback_attempts"] = result.get("provider_fallback_attempts", [])
        task_manager.update_task(task_id, api_provider=actual_provider)

        # 处理seed逻辑差异 - Gemini不返回实际使用的seed
        extracted_seed = result.get("extracted_seed")
        if provider in ["gemini-nanobanana_google", "gemini-nanobanana_replicate"]:
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
                "api_provider": actual_provider,
                "provider_fallback_chain": result.get("provider_fallback_chain", [actual_provider]),
                "provider_fallback_attempts": result.get("provider_fallback_attempts", []),
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
        current_files = await save_generated_image_outputs(task_id, task_dir, image_url, main_input_filename)
        # Never retain a multi-megabyte data URL in the in-memory task registry.
        # The generated file is already persisted and is the canonical public URL.
        persisted_image_url = current_files[0] if current_files else None

        # 水印处理完成后，最终更新75%状态包含完整文件列表
        task_manager.update_task(task_id,
            status=TaskStatus.PROCESSING,
            progress=75,
            result={
                "image_url": persisted_image_url,
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
            "image_url": persisted_image_url,
            "output_files": current_files,
            "stage": "regular_completed"
        })

        # 自动进行2倍放大处理（如果用户选择启用）- 后台异步处理
        auto_upscale_value = params.get("auto_upscale", False)

        if auto_upscale_value:
            background_task = asyncio.create_task(
                process_upscale_background(task_id, task_dir, params)
            )
            _background_tasks.add(background_task)
            background_task.add_done_callback(_background_tasks.discard)

        return
    else:
        error_msg = result.get("error", "未知错误")
        task_manager.set_task_failed(task_id, error_msg)

async def save_generated_image_outputs(task_id: str, task_dir: str, image_url: str, filename: str):
    """Download a generated image and save the original output."""
    # 生成文件名
    original_file = generate_output_filenames(
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
            response = await get_http_client().get(image_url)
            response.raise_for_status()
            content_bytes = response.content
        except Exception as e:
            raise ValueError(f"图片下载失败: {e}")

    # 异步保存原图
    try:
        async with aiofiles.open(original_file, "wb") as f:
            await f.write(content_bytes)
    except Exception as e:
        raise ValueError(f"原图保存失败: {e}")

    output_files = [f"/api/my3dfigure/taskfile/{task_id}/{os.path.basename(original_file)}"]

    # 🚀 关键优化：原图保存完成后立即更新状态 (72%)，让前端能立即获取文件
    task_manager.update_task(task_id,
        status=TaskStatus.PROCESSING,
        progress=72,
        result={
            "image_url": output_files[0],
            "output_files": output_files
        }
    )

    return output_files

async def process_upscale_background(task_id: str, task_dir: str, params: dict):
    """后台处理放大任务，不阻塞主流程"""
    try:

        # 修正：从 'input_images' 列表中获取主输入文件名
        main_input_filename = params.get("input_images")[0] if params.get("input_images") else "reference.png"

        # 获取原图路径
        original_file = generate_output_filenames(
            task_dir, main_input_filename, "png"
        )

        # 调用放大功能
        upscale_success, upscale_message, upscale_result = await upscale_image_with_replicate(
            original_file,  # 使用原图进行放大
            "upscaler",  # 使用upscaler模型
            2,  # 2倍放大
            params.get("upscale_face_enhance", False)  # 使用用户选择的面部增强设置
        )

        if upscale_success:

            # 下载放大后的图片
            upscale_output_url = upscale_result["output_url"]

            # 使用算法值生成文件名
            upscale_filename = generate_upscale_filename("output_reference.png", 2)
            upscale_output_path = os.path.join(task_dir, upscale_filename)

            loop = asyncio.get_running_loop()
            dl_ok = await loop.run_in_executor(
                None, download_upscaled_image, upscale_output_url, upscale_output_path
            )
            if dl_ok:
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
                files.append(f"/api/my3dfigure/taskfile/{task_id}/{filename}")
    except Exception:
        pass
    return files

@router.get("/task-status/{task_id}")
async def get_task_status_async(task_id: str, locale: str = ""):
    """异步获取任务状态"""
    from datetime import datetime

    task_info = task_manager.get_task(task_id)
    if not task_info:
        return JSONResponse({
            "code": "TASK_NOT_FOUND",
            "message": get_message("TASK_NOT_FOUND", locale),
        }, status_code=404)

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
        "api_provider": task_info.get("api_provider")
    }
    if task_info["status"] in {TaskStatus.FAILED, TaskStatus.CANCELLED}:
        response_data["code"] = "GENERATION_FAILED"
        response_data["message"] = get_message("GENERATION_FAILED", task_info.get("locale"))
    # 检查是否有文件
    # 任务已完成时直接用内存缓存，避免每次轮询都 os.listdir
    cached_files = (task_info.get("result") or {}).get("output_files")
    if cached_files:
        response_data["files"] = cached_files
    elif task_info["progress"] >= 70 or task_info["status"] in {
        TaskStatus.DOWNLOADING,
        TaskStatus.COMPLETED,
        TaskStatus.FAILED,
    }:
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


# ==================== 原有端点保持不变 ====================

@router.get("/taskfile/{task_id}/{filename}")
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








@router.post("/check-photo")
@router.post("/check-photo/")
async def check_photo(
    request: Request,
    file: UploadFile = File(...),
    client_city: str = Form(""),
    locale: str = Form(""),
    validation_profile: str = Form("strict"),
    expected_face_count: int = Form(1),
    validation_mode: str = Form("human"),
):
    """Validate a human or pet upload and save it to the current task directory."""
    task_id, task_dir, timestamp = generate_task_dir(DirectoryConfig.TASKS_DIR)
    original_name = os.path.basename(file.filename or "upload.jpg")
    if not original_name:
        original_name = "upload.jpg"

    input_path = os.path.join(task_dir, original_name)

    try:
        await save_validated_upload(file, input_path)

        params = {
            "task_id": task_id,
            "time": timestamp,
            "task_type": "face_check",
            "input_images": [original_name],
            "client_city": client_city,
            "locale": normalize_locale(locale),
            "validation_profile": validation_profile,
            "expected_face_count": expected_face_count,
            "validation_mode": validation_mode,
            "request_url": str(request.url),
            "client_ip": get_request_client_ip(request),
            "user_agent": request.headers.get("user-agent", "")[:500],
        }
        save_params(params, task_dir)

        normalized_validation_mode = str(validation_mode or "human").strip().lower()
        if normalized_validation_mode == "pet":
            face_check = contains_single_pet(input_path, locale)
        else:
            face_check = contains_human(
                input_path,
                locale,
                validation_profile,
                expected_face_count,
            )
        if face_check.get("valid") and normalized_validation_mode != "pet":
            usable_face_boxes = face_check.get("usable_face_boxes")
            if isinstance(usable_face_boxes, list):
                params["usable_face_boxes"] = usable_face_boxes
                save_params(params, task_dir)
        response_data = {
            "task_id": task_id,
            "upload_id": task_id if face_check["valid"] else None,
            "status": "success" if face_check["valid"] else "error",
            "message": face_check["message"],
            "client_city": client_city,
            "user_photo_url": f"/api/my3dfigure/taskfile/{task_id}/{original_name}",
        }
        if face_check.get("code"):
            response_data["code"] = face_check["code"]
        if face_check.get("usable_face_count") is not None:
            response_data["usable_face_count"] = face_check["usable_face_count"]
        if face_check.get("pet_count") is not None:
            response_data["pet_count"] = face_check["pet_count"]

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
        if face_check.get("image_size"):
            response_data["image_size"] = {
                "width": face_check["image_size"][0],
                "height": face_check["image_size"][1],
            }
        if face_check.get("pet"):
            response_data["pet"] = {
                "x": face_check["pet"][0],
                "y": face_check["pet"][1],
                "w": face_check["pet"][2],
                "h": face_check["pet"][3],
            }
        if face_check.get("pet_species"):
            response_data["pet_species"] = face_check["pet_species"]

        return JSONResponse(response_data, status_code=500 if face_check.get("code") == "PET_DETECTION_FAILED" else 200)
    except Exception as e:
        return JSONResponse(
            {
                "task_id": task_id,
                "status": "error",
                "code": "FACE_DETECTION_FAILED",
                "message": get_message("FACE_DETECTION_FAILED", locale),
                "client_city": client_city,
            },
            status_code=500,
        )





# 根据API服务提供商提供特定的端点
