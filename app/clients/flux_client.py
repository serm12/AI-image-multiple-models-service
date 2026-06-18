#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Flux API客户端
支持BFL和Replicate两种Flux API提供商
"""

import os
import json
import base64
import mimetypes
import httpx
import asyncio
from app.core.config import APIConfig
from app.core.helpers import create_response_record
from PIL import Image

class FluxClient:
    """Flux API客户端，支持BFL、Replicate和Fireworks三种提供商"""
    
    def __init__(self, bfl_api_key=None, replicate_token=None, fireworks_api_key=None):
        self.bfl_api_key = bfl_api_key or APIConfig.BFL_API_KEY
        self.replicate_token = replicate_token or APIConfig.REPLICATE_API_TOKEN
        self.fireworks_api_key = fireworks_api_key or APIConfig.FIREWORKS_API_KEY
        self.bfl_base_url = APIConfig.BFL_BASE_URL

    async def generate_image(self, provider, prompt, input_image_path=None, input_image_url=None, 
                           flux_model_variant=None, aspect_ratio=None, output_format=None, seed=None):
        """
        统一的Flux图像生成接口
        
        Args:
            provider: API提供商 ('flux_bfl', 'flux_replicate', 'flux_fireworks')
            prompt: 提示词
            input_image_path: 输入图片路径（本地文件）
            input_image_url: 输入图片URL
            flux_model_variant: Flux模型变体
            aspect_ratio: 宽高比
            output_format: 输出格式
            seed: 随机种子
            
        Returns:
            dict: 生成结果
        """
        if provider == "flux_bfl":
            return await self.generate_with_bfl(
                prompt, input_image_path, input_image_url, 
                flux_model_variant, aspect_ratio, output_format, seed
            )
        elif provider == "flux_replicate":
            return await self.generate_with_replicate(
                prompt, input_image_path, input_image_url,
                flux_model_variant, aspect_ratio, output_format, seed
            )
        elif provider == "flux_fireworks":
            return await self.generate_with_fireworks(
                prompt, input_image_path, input_image_url,
                flux_model_variant, aspect_ratio, output_format, seed
            )
        else:
            raise ValueError(f"不支持的Flux提供商: {provider}")

    async def generate_with_fireworks(self, prompt, input_image_path=None, input_image_url=None,
                                    flux_model_variant=None, aspect_ratio=None, output_format=None, seed=None):
        """
        使用Fireworks AI API生成图像，严格按照官方参考代码的异步轮询流程。
        """
        if not self.fireworks_api_key:
            raise ValueError("Fireworks AI API key not configured.")

        # 根据模型变体选择专属的提交URL
        submit_url_map = {
            "black-forest-labs/flux-kontext-max": "https://api.fireworks.ai/inference/v1/workflows/accounts/fireworks/models/flux-kontext-max",
            "black-forest-labs/flux-kontext-pro": "https://api.fireworks.ai/inference/v1/workflows/accounts/fireworks/models/flux-kontext-pro",
            "black-forest-labs/flux-kontext-dev": "https://api.fireworks.ai/inference/v1/workflows/accounts/fireworks/models/flux-kontext-dev",
        }
        submit_url = submit_url_map.get(flux_model_variant.value)
        if not submit_url:
            raise ValueError(f"Unsupported Fireworks model variant or URL not configured: {flux_model_variant.value}")

        # 动态构建正确的轮询URL
        result_url = f"{submit_url}/get_result"

        # 准备请求体，严格遵循参考代码
        payload = {
            "prompt": prompt,
            "aspect_ratio": aspect_ratio.value,
            "prompt_upsampling": False,
            "safety_tolerance": 2,
        }
        if seed:
            payload["seed"] = seed
        else:
            payload["seed"] = -1

        # 处理参考图
        if input_image_path and os.path.exists(input_image_path):
            print(f"📷 Processing reference image for Fireworks: {input_image_path}")
            with open(input_image_path, "rb") as f:
                b64_img = base64.b64encode(f.read()).decode("utf-8")
            payload["input_image"] = f"data:image/jpeg;base64,{b64_img}"
        elif input_image_url:
            raise NotImplementedError("URL as a reference image is not supported. Please use a local file path.")

        headers = {
            "Authorization": f"Bearer {self.fireworks_api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                # 1. 提交任务
                print(f"🚀 Submitting task to Fireworks AI endpoint: {submit_url}")
                submit_resp = await client.post(submit_url, json=payload, headers=headers)
                submit_resp.raise_for_status()
                submit_data = submit_resp.json()
                if not submit_data:
                    raise ValueError("Fireworks did not return a valid JSON response for submission.")
                request_id = submit_data.get("request_id")
                if not request_id:
                    raise ValueError(f"Fireworks did not return a request_id. Response: {submit_data}")
                print(f"✅ Task submitted with ID: {request_id}")

                # 2. 轮询结果
                max_wait_sec, interval, waited = 300, 5, 0
                final_data, output_url = {}, None

                while waited <= max_wait_sec:
                    print(f"⏳ Polling for result at {result_url}...")
                    await asyncio.sleep(interval)
                    waited += interval
                    
                    poll_resp = await client.post(result_url, headers=headers, json={"id": request_id})
                    poll_resp.raise_for_status()
                    
                    if poll_resp.status_code == 202:
                        print("... Task is still running, continuing to poll.")
                        continue

                    poll_data = poll_resp.json()
                    status = poll_data.get("status")

                    if status in ["Ready", "Complete", "Finished"]:
                        output_url = poll_data.get("result", {}).get("sample")
                        final_data = poll_data
                        break
                    elif status in ["Failed", "Error"]:
                        raise ValueError(f"Fireworks task failed with status '{status}': {poll_data.get('details') or poll_data.get('detail')}")

                if not output_url:
                    raise ValueError(f"Fireworks polling timed out. Last status: {final_data.get('status')}")

                return {
                    "status": "succeeded", "output": output_url, "id": request_id,
                    "logs": f"Using seed: {seed}" if seed else "Using random seed",
                    "raw": final_data, "api_type": "fireworks", "extracted_seed": seed
                }
        except Exception as e:
            raise ValueError(f"Fireworks AI call failed: {e}")
    
    async def generate_with_bfl(self, prompt, input_image_path=None, input_image_url=None,
                              flux_model_variant=None, aspect_ratio=None, output_format=None, seed=None):
        """使用BFL API生成图像"""
        
        # 模型路由映射
        model_route_map = {
            "black-forest-labs/flux-kontext-max": "flux-kontext-max",
            "black-forest-labs/flux-kontext-pro": "flux-kontext-pro",
            "black-forest-labs/flux-kontext-dev": "flux-kontext-dev",
        }
        route = model_route_map.get(flux_model_variant.value, "flux-kontext-pro")
        url = f"{self.bfl_base_url}/{route}"
        
        # 准备负载
        payload = {
            "prompt": prompt,
            "seed": seed if seed else None,
            "aspect_ratio": aspect_ratio.value,
            "output_format": ("jpeg" if output_format.value == "jpg" else output_format.value),
            "prompt_upsampling": False,
            "safety_tolerance": 2
        }
        
        # 处理输入图像文件 - 添加压缩优化
        if input_image_url:
            payload["input_image"] = input_image_url
        elif input_image_path and os.path.exists(input_image_path):
            print(f"📷 处理输入图像: {input_image_path}")
            # 压缩图片以加快上传速度
            compressed_image_path = self._compress_image_for_flux(input_image_path, target_width=512)
            
            with open(compressed_image_path, "rb") as f:
                image_data = f.read()
            
            # 转换为Base64编码
            b64_img = base64.b64encode(image_data).decode("utf-8")
            mime_type, _ = mimetypes.guess_type(compressed_image_path)
            if not mime_type:
                mime_type = "image/jpeg"
            payload["input_image"] = f"data:{mime_type};base64,{b64_img}"
            
            # 清理临时文件（如果是压缩生成的）
            if compressed_image_path != input_image_path and os.path.exists(compressed_image_path):
                try:
                    os.remove(compressed_image_path)
                    print(f"🗑️ 清理临时文件: {compressed_image_path}")
                except:
                    pass
        
        # 移除None字段
        payload = {k: v for k, v in payload.items() if v is not None}
        headers = {
            "x-key": self.bfl_api_key or "",
            "Content-Type": "application/json"
        }
        
        print(f"BFL API Request Payload: {json.dumps(payload, indent=2)}")
        
        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                resp = await client.post(url, json=payload, headers=headers)
                resp.raise_for_status()
                bfl_json = resp.json()
                prediction_id = bfl_json.get("id")
                polling_url = bfl_json.get("polling_url")
                
                output_url = None
                raw_final = bfl_json
                
                # 如果返回了 polling_url，则轮询直到拿到结果
                if polling_url:
                    max_wait_sec = 300
                    interval = 2
                    waited = 0
                    while waited <= max_wait_sec:
                        poll_resp = await client.get(polling_url, headers=headers)
                        poll_resp.raise_for_status()
                        data = poll_resp.json()
                        raw_final = data
                        
                        # 尝试常见键获取图片 URL
                        for key in ["url", "image_url", "output", "result", "data"]:
                            val = data.get(key)
                            if isinstance(val, str) and (val.startswith("http") or val.startswith("data:image/")):
                                output_url = val
                                break
                            if isinstance(val, dict):
                                for k2 in ["url", "image", "image_url", "sample"]:
                                    if isinstance(val.get(k2), str) and (val.get(k2).startswith("http") or val.get(k2).startswith("data:image/")):
                                        output_url = val.get(k2)
                                        break
                            if isinstance(val, list) and val:
                                first = val[0]
                                if isinstance(first, str) and (first.startswith("http") or first.startswith("data:image/")):
                                    output_url = first
                                    break
                        
                        if output_url:
                            break
                        
                        status_val = str(data.get("status") or data.get("state") or "").lower()
                        if status_val in ["failed", "error", "canceled"]:
                            raise ValueError(f"BFL 任务失败: {data}")
                        
                        await asyncio.sleep(interval)
                        waited += interval
                    
                    if not output_url:
                        raise ValueError(f"BFL 轮询超时或未返回图片 URL，可用字段: {list(raw_final.keys())}")
                else:
                    # 无轮询链接，直接从初始响应解析
                    for key in ["url", "image_url", "output", "result", "data"]:
                        val = bfl_json.get(key)
                        if isinstance(val, str) and (val.startswith("http") or val.startswith("data:image/")):
                            output_url = val
                            break
                        if isinstance(val, dict):
                            for k2 in ["url", "image", "image_url", "sample"]:
                                if isinstance(val.get(k2), str) and (val.get(k2).startswith("http") or val.get(k2).startswith("data:image/")):
                                    output_url = val.get(k2)
                                    break
                        if isinstance(val, list) and val and isinstance(val[0], str) and (val[0].startswith("http") or val[0].startswith("data:image/")):
                            output_url = val[0]
                            break
                    if not output_url:
                        raise ValueError(f"BFL 响应未包含图片 URL，可用字段: {list(bfl_json.keys())}")
                    raw_final = bfl_json
                
                return {
                    "status": "succeeded",
                    "output": output_url,
                    "id": prediction_id,
                    "logs": f"Using seed: {seed}" if seed else "Using random seed",
                    "raw": raw_final,
                    "api_type": "bfl",
                    "extracted_seed": seed  # BFL返回输入的seed
                }
                
        except Exception as e:
            raise ValueError(f"BFL 调用失败: {e}")
        
    async def generate_with_replicate(self, prompt, input_image_path=None, input_image_url=None,
                                    flux_model_variant=None, aspect_ratio=None, output_format=None, seed=None):
        """使用Replicate API生成图像"""
        try:
            import replicate
            
            # 准备输入数据
            input_dict = {
                "prompt": prompt,
                "aspect_ratio": aspect_ratio.value,
                "output_format": output_format.value
            }
            
            # Replicate API不接受None值的seed，只有存在时才添加
            if seed:
                input_dict["seed"] = seed
            
            # 处理输入图片 - 添加压缩优化
            if input_image_url:
                input_dict["input_image"] = input_image_url
                # 如果有URL，直接使用replicate.run
                result = await asyncio.to_thread(
                    replicate.run,
                    flux_model_variant.value,
                    input=input_dict
                )
                return {
                    "status": "succeeded", 
                    "output": result,
                    "id": None,
                    "logs": f"Using seed: {seed}" if seed else "Using random seed",
                    "api_type": "replicate",
                    "extracted_seed": seed
                }

            elif input_image_path and os.path.exists(input_image_path):
                print(f"📷 处理输入图像: {input_image_path}")
                # 压缩图片以加快上传速度
                compressed_image_path = self._compress_image_for_flux(input_image_path, target_width=512)
                
                def run_prediction():
                    with open(compressed_image_path, "rb") as img_file:
                        input_dict["input_image"] = img_file
                        
                        # 使用Replicate客户端
                        client_sync = replicate.Client()
                        prediction = client_sync.predictions.create(
                            version=flux_model_variant.value,
                            input=input_dict
                        )
                        
                        # 等待预测完成
                        prediction.wait()
                        return prediction

                prediction = await asyncio.to_thread(run_prediction)
                result = prediction.dict()

                # 清理临时文件（如果是压缩生成的）
                if compressed_image_path != input_image_path and os.path.exists(compressed_image_path):
                    try:
                        os.remove(compressed_image_path)
                        print(f"🗑️ 清理临时文件: {compressed_image_path}")
                    except:
                        pass
                
                return {
                    "status": "succeeded",
                    "output": result.get("output"),
                    "id": prediction.id,
                    "logs": f"Using seed: {seed}" if seed else "Using random seed",
                    "raw": result,
                    "api_type": "replicate",
                    "extracted_seed": seed
                }
            else:
                # 如果没有本地图片或URL，也使用replicate.run
                result = await asyncio.to_thread(
                    replicate.run,
                    flux_model_variant.value,
                    input=input_dict
                )
                return {
                    "status": "succeeded", 
                    "output": result,
                    "id": None,
                    "logs": f"Using seed: {seed}" if seed else "Using random seed",
                    "api_type": "replicate",
                    "extracted_seed": seed
                }
                
        except Exception as e:
            raise ValueError(f"Replicate 调用失败: {e}")

    def _compress_image_for_flux(self, image_path, target_width=512):
        """
        将图片压缩到指定宽度以加快上传速度
        
        Args:
            image_path: 图片路径
            target_width: 目标宽度(默认512px，适合Flux模型)
        
        Returns:
            压缩后的图片文件路径
        """
        try:
            with Image.open(image_path) as img:
                # 获取原始尺寸
                orig_width, orig_height = img.size
                print(f"📷 原始图片: {orig_width}x{orig_height}")
                
                # 如果图片宽度已经小于等于目标宽度，直接返回原路径
                if orig_width <= target_width:
                    print(f"✅ 图片已经够小，无需压缩")
                    return image_path
                
                # 计算缩放比例
                scale_factor = target_width / orig_width
                new_height = int(orig_height * scale_factor)
                
                # 转换为RGB模式（如果需要）
                if img.mode in ('RGBA', 'P'):
                    img = img.convert('RGB')
                
                # 缩放图片
                resized_img = img.resize((target_width, new_height), Image.Resampling.LANCZOS)
                
                # 生成压缩后的文件路径
                import tempfile
                temp_dir = tempfile.gettempdir()
                compressed_path = os.path.join(temp_dir, f"flux_compressed_{os.path.basename(image_path)}.jpg")
                
                # 保存压缩后的图片
                resized_img.save(compressed_path, 'JPEG', quality=80, optimize=True)
                
                # 计算压缩效果
                original_size = os.path.getsize(image_path)
                compressed_size = os.path.getsize(compressed_path)
                compression_ratio = (1 - compressed_size / original_size) * 100
                
                # 计算详细的压缩统计
                size_reduction = original_size - compressed_size
                
                print(f"📦 Flux图片压缩完成:")
                print(f"   📏 尺寸: {orig_width}x{orig_height} → {target_width}x{new_height}")
                print(f"   📁 文件大小: {original_size:,} bytes → {compressed_size:,} bytes")
                print(f"   📉 压缩比例: {compression_ratio:.1f}% (减少 {size_reduction:,} bytes)")
                print(f"   ⚡ 上传速度提升预估: ~{compression_ratio:.1f}%")
                
                return compressed_path
                
        except Exception as e:
            print(f"⚠️ 图片压缩失败，使用原图: {e}")
            return image_path

# 导出实例
flux_client = FluxClient() 
