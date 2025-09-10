"""
Seedream-4 Fal.ai API客户端 (Corrected with native async SDK)
"""

import os
from datetime import datetime
from config import APIConfig
import fal
import httpx
import base64

class Seedream4FalClient:
    """Seedream-4 Fal.ai API客户端"""
    
    def __init__(self, api_key=None):
        """初始化客户端"""
        self.api_key = api_key or APIConfig.FAL_API_KEY
        if not self.api_key:
            raise ValueError("FAL_API_KEY not found in environment variables")
        
        # The `fal` library automatically uses this environment variable.
        os.environ["FAL_KEY"] = self.api_key
        self.model_id = "fal-ai/bytedance/seedream/v4/edit"

    def _convert_aspect_ratio_to_image_size(self, aspect_ratio_str: str) -> str:
        """将 aspect_ratio (e.g., '3:4') 转换为 fal.ai 的 image_size (e.g., 'portrait_3_4')"""
        
        # 特殊处理match_input_image
        if aspect_ratio_str == "match_input_image":
            return "portrait_4_3"
            
        # 对于其他情况，检查是否包含冒号
        if not aspect_ratio_str or ':' not in aspect_ratio_str:
            return "portrait_4_3"

        mapping = {
            "1:1": "square_hd", "16:9": "landscape_16_9", "9:16": "portrait_16_9",
            "4:3": "landscape_4_3", "3:4": "portrait_4_3", "3:2": "landscape_4_3",
            "2:3": "portrait_4_3"
        }
        return mapping.get(aspect_ratio_str, "portrait_4_3")

    async def _upload_local_image(self, image_path):
        """
        上传本地图片到fal.ai并返回URL
        """
        if not os.path.exists(image_path):
            raise ValueError(f"Image file not found: {image_path}")
        
        # 直接使用HTTP上传图片到fal.ai
        return await self._upload_image_via_http(image_path)

    async def _upload_image_via_http(self, image_path):
        """
        通过HTTP上传图片到fal.ai
        """
        # 直接使用HTTP上传方法
        url = "https://upload.fal.ai/upload"
        headers = {
            "Authorization": f"Key {self.api_key}",
            "Accept": "application/json"
        }
        
        async with httpx.AsyncClient() as client:
            with open(image_path, "rb") as f:
                files = {"file": f}
                try:
                    response = await client.post(url, files=files, headers=headers)
                    response.raise_for_status()
                    result = response.json()
                    return result["url"]
                except Exception as e:
                    # 如果上传失败，尝试将图片转换为base64编码直接传递
                    print(f"HTTP上传失败: {e}, 尝试使用base64编码")
                    import base64
                    import mimetypes
                    
                    # 获取文件的MIME类型
                    mime_type, _ = mimetypes.guess_type(image_path)
                    if not mime_type:
                        mime_type = "image/jpeg"
                    
                    # 读取文件并转换为base64
                    with open(image_path, "rb") as f:
                        image_data = f.read()
                        b64_img = base64.b64encode(image_data).decode("utf-8")
                        return f"data:{mime_type};base64,{b64_img}"

    async def generate_image(self, prompt, input_image_paths=None, seed=None, art_style=None, aspect_ratio="1:1", **kwargs):
        """
        使用Seedream-4 Fal.ai生成图像
        """
        if not input_image_paths:
            raise ValueError("Seedream-4 Fal.ai 模型需要至少一个参考图像。")

        print("Seedream-4 Fal.ai 生成图像 (使用原生 async SDK)...")
        # 简化日志输出，避免显示过长的base64代码
        simplified_paths = []
        for path in input_image_paths:
            if isinstance(path, str) and path.startswith("data:image/"):
                simplified_paths.append(f"data:image/...({len(path)} chars)")
            else:
                simplified_paths.append(path)
        print(f"输入参数: prompt={prompt[:50]}..., input_image_paths={simplified_paths}, seed={seed}")
        
        # 处理输入图像路径，将本地路径转换为URL
        processed_image_urls = []
        for image_path in input_image_paths:
            if isinstance(image_path, str):
                if os.path.exists(image_path):
                    # 本地文件，需要上传
                    print(f"上传本地图片: {image_path}")
                    url = await self._upload_local_image(image_path)
                    processed_image_urls.append(url)
                    print(f"上传完成: {url}")
                elif image_path.startswith(("http://", "https://")):
                    # 已经是URL，直接使用
                    processed_image_urls.append(image_path)
                    print(f"使用URL: {image_path}")
                else:
                    # 其他情况，尝试直接使用
                    processed_image_urls.append(image_path)
                    print(f"使用其他路径: {image_path}")
            else:
                # 其他类型，尝试直接使用
                processed_image_urls.append(image_path)
                print(f"使用其他类型: {image_path}")
        
        print(f"开始转换图像尺寸: 输入比例='{aspect_ratio}' (类型: {type(aspect_ratio)})")
        image_size = self._convert_aspect_ratio_to_image_size(aspect_ratio)
        print(f"图像尺寸转换完成: 输入比例='{aspect_ratio}' -> Fal.ai参数='{image_size}'")
        
        # 使用 fal.apps.submit 异步提交任务
        print("提交任务到Fal.ai...")
        handler = fal.apps.submit(
            self.model_id,
            arguments={
                "prompt": prompt,
                "image_urls": processed_image_urls,
                "image_size": image_size,
                "seed": seed
            }
        )

        # 异步等待结果
        print("等待Fal.ai响应...")
        try:
            # 使用handler.get()获取结果
            result = await handler.get()
            # 简化响应输出，避免显示过长的内容
            if isinstance(result, dict) and "images" in result:
                simplified_result = {
                    "status": result.get("status", "unknown"),
                    "images_count": len(result.get("images", [])),
                    "seed": result.get("seed")
                }
                print(f"Fal.ai响应摘要: {simplified_result}")
            else:
                print(f"Fal.ai响应长度: {len(str(result))} 字符")
        except Exception as e:
            print(f"获取Fal.ai响应时出错: {e}")
            # 如果get()方法失败，尝试其他方法
            try:
                # 尝试使用fetch_result方法
                result = handler.fetch_result()
                # 简化响应输出
                if isinstance(result, dict) and "images" in result:
                    simplified_result = {
                        "status": result.get("status", "unknown"),
                        "images_count": len(result.get("images", [])),
                        "seed": result.get("seed")
                    }
                    print(f"通过fetch_result获取Fal.ai响应摘要: {simplified_result}")
                else:
                    print(f"通过fetch_result获取Fal.ai响应长度: {len(str(result))} 字符")
            except Exception as e2:
                print(f"通过fetch_result获取Fal.ai响应时出错: {e2}")
                # 如果所有方法都失败，抛出异常
                raise ValueError(f"无法从Fal.ai获取响应: {e}")
            
        # 确保result是一个字典
        if not isinstance(result, dict):
            raise ValueError(f"Fal.ai API返回了意外的响应类型: {type(result)}")
            
        if not result or "images" not in result or not result["images"]:
            raise ValueError("Fal.ai API 未返回有效的图像数据")
        
        image_url = result["images"][0]["url"]
        print(f"Seedream-4 Fal.ai 图像生成成功")
        print(f"图像URL: {image_url}")

        prediction_id = f"seedream4_fal_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{seed or 'na'}"
        
        return {
            "id": prediction_id,
            "status": "succeeded",
            "output": image_url,
            "logs": f"Seedream-4 Fal.ai 生成 - 种子: {seed}",
            "input": {
                "prompt": prompt,
                "image_size": image_size,
                "seed": seed
            },
            "api_type": "seedream4_fal",
            "extracted_seed": seed
        }
