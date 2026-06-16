import os
import json
from contextlib import ExitStack
import requests
import replicate
from datetime import datetime

# 支持的放大模型（使用完整的版本ID）
UPSCALE_MODELS = {
    "real-esrgan": "nightmareai/real-esrgan:f121d640bd286e1fdc67f9799164c1d5be36ff74576ee11c803ae5b665dd46aa",
    "gfpgan": "tencentarc/gfpgan:9283608cc6b7be6b65a8e44983db012355fde4132009bf99d976b2f0896856a3", 
    "upscaler": "google/upscaler"
}

def validate_upscale_params(model: str, scale: int, face_enhance: bool) -> tuple[bool, str]:
    """验证放大参数"""
    if model not in UPSCALE_MODELS:
        return False, f"不支持的模型: {model}"
    
    if scale not in [2, 4]:
        return False, "只支持2x或4x放大"
    
    return True, ""

def build_upscale_input_params(input_file, model: str, scale: int, face_enhance: bool) -> dict:
    """构建放大输入参数"""
    input_params = {}
    
    # 不同模型使用不同的参数名称
    if model == "gfpgan":
        input_params["img"] = input_file
    elif model == "upscaler":
        # google/upscaler 模型使用不同的参数
        input_params["image"] = input_file
        input_params["upscale_factor"] = f"x{scale}"
        input_params["compression_quality"] = 100
    else:
        input_params["image"] = input_file
        input_params["scale"] = scale
    
    # 只有real-esrgan模型支持face_enhance参数
    if model == "real-esrgan":
        input_params["face_enhance"] = face_enhance
        print(f"👤 启用面部增强: {face_enhance}")
    elif face_enhance:
        print(f"⚠️ 警告: {model} 模型不支持面部增强，将忽略face_enhance参数")
    
    return input_params

async def upscale_image_with_replicate(
    input_image_path: str, 
    model: str, 
    scale: int, 
    face_enhance: bool = False
) -> tuple[bool, str, dict]:
    """
    使用Replicate API放大图片
    
    Args:
        input_image_path: 输入图片路径
        model: 放大模型名称
        scale: 放大倍数
        face_enhance: 是否启用面部增强
    
    Returns:
        (success, message, result_data)
        - success: 是否成功
        - message: 结果消息
        - result_data: 包含output_url, full_response等数据
    """
    import asyncio

    # 验证参数
    is_valid, error_msg = validate_upscale_params(model, scale, face_enhance)
    if not is_valid:
        return False, error_msg, {}
    
    print(f"🔄 开始放大图片，模型: {model}, 倍数: {scale}x, 面部增强: {face_enhance}")
    
    try:
        with ExitStack() as stack:
            input_file = stack.enter_context(open(input_image_path, "rb"))
            input_params = build_upscale_input_params(input_file, model, scale, face_enhance)

            # 使用replicate.Client()获取完整的预测信息
            try:
                # 创建replicate客户端
                client = replicate.Client()
                
                # 异步运行阻塞的 create 和 wait
                def run_prediction():
                    prediction = client.predictions.create(
                        version=UPSCALE_MODELS[model],
                        input=input_params
                    )
                    print(f"📥 收到放大预测响应，ID: {prediction.id}")
                    print(f"预测状态: {prediction.status}")
                    
                    # 等待预测完成
                    print("⏳ 等待放大预测完成...")
                    prediction.wait()
                    print(f"✅ 放大预测完成，最终状态: {prediction.status}")
                    return prediction

                prediction = await asyncio.to_thread(run_prediction)
                
                # 获取完整的预测信息
                full_response = prediction.dict()
                print(f"📄 获取到完整放大预测信息，ID: {prediction.id}")
                
                # 获取图片URL
                output_url = str(prediction.output)
                print("最终放大图片URL:", output_url)
                
                result_data = {
                    "output_url": output_url,
                    "full_response": full_response,
                    "prediction_id": prediction.id,
                    "success": True
                }
                
                return True, "放大成功", result_data
                
            except Exception as e:
                print(f"⚠️ 使用client.predictions.create()失败: {e}")
                print("🔄 回退到replicate.run()方法...")
                
                # 回退到原来的方法 - 同样异步化
                output_url = await asyncio.to_thread(
                    replicate.run,
                    UPSCALE_MODELS[model],
                    input=input_params
                )
                
                # 创建基本响应记录
                full_response = {
                    "completed_at": datetime.now().isoformat(),
                    "created_at": datetime.now().isoformat(),
                    "data_removed": False,
                    "error": None,
                    "id": None,
                    "input": {
                        "model": model,
                        "scale": scale,
                        "face_enhance": face_enhance
                    },
                    "logs": f"Using {model} model with scale {scale}x",
                    "metrics": {
                        "image_count": 1,
                        "predict_time": 0,
                        "total_time": 0
                    },
                    "output": output_url,
                    "started_at": datetime.now().isoformat(),
                    "status": "succeeded",
                    "urls": {
                        "stream": "",
                        "cancel": "",
                        "get": "",
                        "web": ""
                    },
                    "version": "hidden"
                }
                
                result_data = {
                    "output_url": output_url,
                    "full_response": full_response,
                    "prediction_id": None,
                    "success": True
                }
                
                return True, "放大成功（回退模式）", result_data
            
    except Exception as e:
        error_msg = f"放大图片时出错: {str(e)}"
        print(f"❌ {error_msg}")
        return False, error_msg, {}

def download_upscaled_image(output_url: str, output_path: str) -> bool:
    """下载放大后的图片"""
    try:
        response = requests.get(output_url, timeout=60)
        if response.status_code == 200:
            with open(output_path, "wb") as f:
                f.write(response.content)
            print(f"✅ 图片下载完成: {output_path}")
            return True
        else:
            print(f"❌ 下载放大图片失败: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 下载图片时出错: {e}")
        return False

def generate_upscale_filename(original_filename: str, scale: int) -> str:
    """生成放大后的文件名"""
    # 从配置中获取算法值
    from config import AlgorithmConfig
    algorithm_value = AlgorithmConfig.get_algorithm_value()
    
    filename_without_ext = os.path.splitext(original_filename)[0]
    return f"{filename_without_ext}_upscaled_{scale}x_{algorithm_value}.png"

def save_upscale_task_info(
    task_dir: str, 
    task_id: str, 
    timestamp: str,
    model: str, 
    scale: int, 
    face_enhance: bool, 
    original_filename: str,
    full_response: dict
) -> None:
    """保存放大任务信息"""
    # 保存放大任务参数
    upscale_params = {
        "task_id": task_id,
        "time": timestamp,
        "model": model,
        "scale": scale,
        "face_enhance": face_enhance,
        "input_image": original_filename,
        "description": f"图片放大任务 - {model} {scale}x",
        "replicate_full_response": full_response
    }
    
    # 保存参数文件
    params_file = os.path.join(task_dir, "params.json")
    with open(params_file, 'w', encoding='utf-8') as f:
        json.dump(upscale_params, f, ensure_ascii=False, indent=2)
    
    # 保存完整响应到 JSON 文件
    response_file = os.path.join(task_dir, "replicate_response.json")
    with open(response_file, 'w', encoding='utf-8') as f:
        json.dump(full_response, f, ensure_ascii=False, indent=2)
    
    print(f"📄 任务信息已保存到: {task_dir}")

def create_upscale_lookup_folder(task_dir: str, prediction_id: str) -> str:
    """创建放大任务定位文件夹"""
    if prediction_id:
        # 使用短ID来避免路径过长问题
        short_id = prediction_id[:8] if len(prediction_id) > 8 else prediction_id
        new_task_dir = f"{task_dir}_up_{short_id}"
        if not os.path.exists(new_task_dir):
            os.makedirs(new_task_dir)
            print(f"📁 创建放大任务定位文件夹: {os.path.basename(new_task_dir)}")
        else:
            print(f"📁 放大任务定位文件夹已存在: {os.path.basename(new_task_dir)}")
        return new_task_dir
    return task_dir

def get_upscale_models_info() -> dict:
    """获取支持的放大模型信息"""
    return {
        "models": UPSCALE_MODELS,
        "supported_scales": [2, 4],
        "face_enhance_support": {
            "real-esrgan": True,
            "gfpgan": False,
            "codeformer": False
        }
    } 
