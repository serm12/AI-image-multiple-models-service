"""
Seedream-4 Replicate API客户端
基于Replicate平台的 bytedance/seedream-4 模型
"""

import os
import asyncio
import replicate
from datetime import datetime
from app.core.config import APIConfig

class Seedream4ReplicateClient:
    """Seedream-4 Replicate API客户端"""
    
    def __init__(self, api_key=None):
        """初始化客户端"""
        self.api_key = api_key or APIConfig.REPLICATE_API_TOKEN
        if not self.api_key:
            raise ValueError("REPLICATE_API_TOKEN not found in environment variables")
        
        os.environ["REPLICATE_API_TOKEN"] = self.api_key
        self.model_name = "bytedance/seedream-4"
    
    async def _upload_image_to_replicate(self, image_path):
        """将本地图像上传到Replicate并返回URL"""
        try:
            print(f"📤 正在上传参考图像到Replicate: {image_path}")
            
            def upload_sync():
                with open(image_path, "rb") as image_file:
                    return replicate.files.create(image_file)

            uploaded_file = await asyncio.to_thread(upload_sync)
            uploaded_image_url = uploaded_file.urls['get']
            
            print(f"✅ 图像上传成功: {uploaded_image_url}")
            return uploaded_image_url
            
        except Exception as e:
            print(f"❌ 图像上传失败: {e}")
            return None

    async def generate_image(self, prompt, input_image_paths=None, seed=None, art_style=None, size="2K", width=None, height=None, aspect_ratio="3:4", sequential_image_generation="disabled", **kwargs):
        """
        使用Seedream-4 Replicate生成图像
        """
        try:
            print(f"🎨 Seedream-4 Replicate生成图像")
            print(f"📝 提示词: {prompt}")
            
            actual_seed = seed if seed is not None else 12345
            print(f"🎯 使用种子: {actual_seed}")
            
            if not input_image_paths:
                raise ValueError("Seedream-4 模型需要至少一个参考图像。")

            uploaded_image_urls = []
            for image_path in input_image_paths:
                print(f"📷 使用参考图像: {image_path}")
                reference_image_url = await self._upload_image_to_replicate(image_path)
                if not reference_image_url:
                    print(f"⚠️ 参考图像上传失败，跳过: {image_path}")
                    continue
                uploaded_image_urls.append(reference_image_url)

            if not uploaded_image_urls:
                raise ValueError("所有参考图像都上传失败。")

            input_params = {
                "prompt": prompt,
                "image_input": uploaded_image_urls,
                "seed": actual_seed,
                "size": size,
                "aspect_ratio": aspect_ratio,
                "sequential_image_generation": sequential_image_generation,
            }

            if size == "custom":
                if width:
                    input_params["width"] = width
                if height:
                    input_params["height"] = height
            
            print("🎨 正在使用Replicate seedream-4模型生成图像...")
            
            output = await asyncio.to_thread(
                replicate.run,
                self.model_name,
                input=input_params
            )
            
            if not output:
                raise ValueError("Seedream-4 Replicate API未返回有效内容")
            
            image_url = None
            if isinstance(output, list) and len(output) > 0:
                first_item = output[0]
                if isinstance(first_item, str):
                    image_url = first_item
                elif hasattr(first_item, 'url'):
                    image_url = str(first_item.url)
                else:
                    image_url = str(first_item)
            else:
                raise ValueError(f"预期的输出格式是列表，但得到: {type(output)}")

            print(f"✅ Seedream-4 Replicate图像生成成功")
            print(f"📥 图像URL: {image_url}")
            
            prediction_id = f"seedream4_replicate_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{actual_seed}"
            
            return {
                "id": prediction_id,
                "status": "succeeded",
                "output": image_url,
                "logs": f"Seedream-4 Replicate生成 - 种子: {actual_seed}",
                "input": {
                    "prompt": prompt,
                    "seed": actual_seed,
                    "style": art_style
                },
                "api_type": "seedream4_replicate",
                "extracted_seed": actual_seed
            }
            
        except Exception as e:
            print(f"❌ Seedream-4 Replicate生成图像失败: {e}")
            raise ValueError(f"Seedream-4 Replicate API调用失败: {e}")
