"""
Seedream-4 Fal.ai API客户端 (Corrected with native async SDK)
"""

import os
from datetime import datetime
from config import APIConfig
import os
from datetime import datetime
from config import APIConfig
import fal_client
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
        
        # 使用fal.toolkit.File.from_path上传图片到fal.ai
        try:
            print(f"使用fal_client.upload_file上传图片: {image_path}")
            # 注意：新版 fal_client 可能使用 upload 或 upload_file
            if hasattr(fal_client, 'upload_file'):
                url = fal_client.upload_file(image_path)
            elif hasattr(fal_client, 'upload'):
                url = fal_client.upload(image_path)
            else:
                raise ImportError("fal_client module has no upload function")
            return url
        except Exception as e:
            print(f"fal.toolkit.File上传失败: {e}, 尝试使用base64编码")
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
                    if url.startswith("data:"):
                        print(f"上传完成 (base64): data:image/...({len(url)} chars)")
                    else:
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
        
        # 使用 fal_client.submit 异步提交任务
        print("提交任务到Fal.ai...")
        # 新版 fal_client.submit(id, arguments={...}) 或直接传参
        # 根据官方文档通常是 fal_client.submit("model/id", arguments={...}) 
        # 但报错说 unexpected keyword argument 'arguments'，
        # 那么可能是直接 fal_client.submit("model/id", prompt="...", ...) 
        # 或者 fal_client.submit("model/id", {...}) 作为位置参数
        
        arguments_dict = {
            "prompt": prompt,
            "image_urls": processed_image_urls,
            "image_size": image_size,
            "seed": seed
        }
        
        # 尝试直接传递字典作为第二个位置参数 (common pattern for new SDKs)
        try:
            handler = fal_client.submit(
                self.model_id,
                arguments_dict
            )
        except TypeError:
            # 如果失败，尝试作为 kwargs 解包
             handler = fal_client.submit(
                self.model_id,
                **arguments_dict
            )

        # 异步等待结果
        print("等待Fal.ai响应...")
        try:
            # 使用handler.get()获取结果 (注意: fal_client 的 handler.get() 是同步阻塞的，
            # 但在这个特定的客户端实现中，我们暂时接受它，或者使用 run_in_executor 将其包装为非阻塞)
            # 为了更好的异步兼容性，最好是在 executor 中运行它
            import asyncio
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, handler.get)

            # 简化响应输出，避免显示过长的内容
            if isinstance(result, dict) and "images" in result:
                simplified_result = {
                    "status": "succeeded", # fal-client 返回字典通常意味着成功
                    "images_count": len(result.get("images", [])),
                    "seed": result.get("seed")
                }
                print(f"Fal.ai响应摘要: {simplified_result}")
            else:
                print(f"Fal.ai响应长度: {len(str(result))} 字符")
        except Exception as e:
            print(f"获取Fal.ai响应时出错: {e}")
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
