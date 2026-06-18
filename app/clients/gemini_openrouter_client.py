import os
import base64
import asyncio
import aiohttp
import json
import time
from PIL import Image
from io import BytesIO
import logging
from typing import List, Optional, Dict, Any, Union

class GeminiOpenRouterClient:
    """
    OpenRouter Gemini 2.0 Flash Image Preview 客户端
    通过 OpenRouter 平台使用 Google Gemini 2.0 Flash Image Preview 模型生成图像
    """
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv('OPENROUTER_API_KEY')
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        self.model_name = "google/gemini-2.5-flash-image-preview"
        self.max_image_size = 1024 * 1024  # 1MB
        self.logger = logging.getLogger(__name__)
        
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY is required")
    
    def compress_image(self, image_data: bytes, max_size: int = None) -> bytes:
        """
        压缩图像到指定大小
        
        Args:
            image_data: 原始图像数据
            max_size: 最大文件大小（字节），默认使用 self.max_image_size
        
        Returns:
            压缩后的图像数据
        """
        if max_size is None:
            max_size = self.max_image_size
            
        if len(image_data) <= max_size:
            return image_data
        
        try:
            # 打开图像
            image = Image.open(BytesIO(image_data))
            
            # 转换为RGB（如果需要）
            if image.mode in ('RGBA', 'LA', 'P'):
                background = Image.new('RGB', image.size, (255, 255, 255))
                if image.mode == 'P':
                    image = image.convert('RGBA')
                background.paste(image, mask=image.split()[-1] if image.mode in ('RGBA', 'LA') else None)
                image = background
            elif image.mode != 'RGB':
                image = image.convert('RGB')
            
            # 二分法查找合适的质量
            quality = 95
            min_quality = 10
            max_quality = 95
            
            while min_quality <= max_quality:
                output = BytesIO()
                image.save(output, format='JPEG', quality=quality, optimize=True)
                compressed_data = output.getvalue()
                
                if len(compressed_data) <= max_size:
                    min_quality = quality + 1
                else:
                    max_quality = quality - 1
                
                quality = (min_quality + max_quality) // 2
            
            # 使用找到的最佳质量
            output = BytesIO()
            image.save(output, format='JPEG', quality=max_quality, optimize=True)
            final_data = output.getvalue()
            
            self.logger.info(f"图像压缩: {len(image_data):,} bytes -> {len(final_data):,} bytes (质量: {max_quality})")
            return final_data
            
        except Exception as e:
            self.logger.error(f"图像压缩失败: {e}")
            return image_data
    
    def estimate_tokens(self, prompt: str, images: List[str] = None) -> int:
        """
        估算 token 数量
        
        Args:
            prompt: 文本提示词
            images: 图像列表（base64 或 data URL）
        
        Returns:
            估算的 token 数量
        """
        # 文本 token 估算（粗略估算：4个字符约等于1个token）
        text_tokens = len(prompt) // 4
        
        # 图像 token 估算（每张图像约 258 tokens）
        image_tokens = len(images) * 258 if images else 0
        
        total_tokens = text_tokens + image_tokens
        self.logger.info(f"Token 估算: 文本={text_tokens}, 图像={image_tokens}, 总计={total_tokens}")
        
        return total_tokens
    
    def _prepare_image_data(self, image_path_or_data: Union[str, bytes]) -> str:
        """
        准备图像数据为 data URL 格式
        
        Args:
            image_path_or_data: 图像文件路径或图像数据
        
        Returns:
            data URL 格式的图像数据
        """
        try:
            if isinstance(image_path_or_data, str):
                # 如果是文件路径
                if os.path.exists(image_path_or_data):
                    with open(image_path_or_data, 'rb') as f:
                        image_data = f.read()
                else:
                    # 可能是 data URL 或 base64 字符串
                    if image_path_or_data.startswith('data:image'):
                        return image_path_or_data
                    else:
                        # 假设是 base64 字符串
                        image_data = base64.b64decode(image_path_or_data)
            else:
                # 如果是字节数据
                image_data = image_path_or_data
            
            # 压缩图像
            compressed_data = self.compress_image(image_data)
            
            # 转换为 base64
            base64_data = base64.b64encode(compressed_data).decode('utf-8')
            
            # 检测图像格式
            try:
                image = Image.open(BytesIO(compressed_data))
                format_lower = image.format.lower() if image.format else 'jpeg'
                if format_lower == 'jpeg':
                    mime_type = 'image/jpeg'
                elif format_lower == 'png':
                    mime_type = 'image/png'
                elif format_lower == 'webp':
                    mime_type = 'image/webp'
                else:
                    mime_type = 'image/jpeg'  # 默认
            except:
                mime_type = 'image/jpeg'  # 默认
            
            return f"data:{mime_type};base64,{base64_data}"
            
        except Exception as e:
            self.logger.error(f"准备图像数据失败: {e}")
            raise
    
    def _extract_image_from_response(self, response_data: Dict[str, Any]) -> Optional[str]:
        """
        从 OpenRouter API 响应中提取图像数据 (最终修正版)
        
        Args:
            response_data: API 响应数据
        
        Returns:
            图像数据（base64 data URL 或 HTTP URL）
        """
        if not response_data:
            self.logger.error("response_data 为空，无法提取图像")
            return None
            
        try:
            self.logger.info("🔍 开始解析JSON响应...")
            choice = response_data.get("choices", [{}])[0]
            message = choice.get("message", {})

            if not message:
                self.logger.error("❌ 响应的 'choices' 中没有 'message' 字段")
                return None

            # --- 方案一: 检查 message['images'] (根据日志，这是最主要的路径) ---
            if "images" in message and isinstance(message["images"], list):
                self.logger.info(f"📝 发现 'images' 列表 (长度: {len(message['images'])})")
                for item in message["images"]:
                    if isinstance(item, dict) and item.get("type") == "image_url":
                        image_url = item.get("image_url", {}).get("url")
                        if image_url and image_url.startswith("data:image"):
                            self.logger.info("✅ 在 'images' 列表中成功提取 base64 图像数据")
                            return image_url
                self.logger.warning("⚠️ 'images' 列表存在，但未找到有效的图像URL")

            # --- 方案二: 检查 message['content'] 是否为列表 ---
            content = message.get("content")
            if isinstance(content, list):
                self.logger.info(f"📝 'images'未找到，正在检查 'content' 列表 (长度: {len(content)})")
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "image_url":
                        image_url = item.get("image_url", {}).get("url")
                        if image_url and image_url.startswith("data:image"):
                            self.logger.info("✅ 在 'content' 列表中成功提取 base64 图像数据")
                            return image_url

            # --- 方案三: 检查 message['content'] 是否为字符串 ---
            elif isinstance(content, str) and content.startswith("data:image"):
                 self.logger.info("✅ 在 'content' 字符串中发现 data:image 格式")
                 return content

            self.logger.error(f"❌ 在 'message' 中未找到任何可识别的图像数据。响应预览: {str(response_data)[:500]}")
            return None

        except Exception as e:
            self.logger.error(f"提取图像数据时发生意外错误: {e}")
            return None
    
    async def generate_image(self, 
                           prompt: str, 
                           reference_images: List[Union[str, bytes]] = None,
                           seed: Optional[int] = None,
                           **kwargs) -> Optional[str]:
        """
        异步生成图像
        
        Args:
            prompt: 文本提示词
            reference_images: 参考图像列表（文件路径、字节数据或 data URL）
            seed: 随机种子
            **kwargs: 其他参数
        
        Returns:
            生成的图像数据（base64 data URL 或 HTTP URL）
        """
        response_data = None  # 初始化response_data变量
        try:
            # 准备参考图像
            image_data_urls = []
            if reference_images:
                for img in reference_images:
                    try:
                        data_url = self._prepare_image_data(img)
                        image_data_urls.append(data_url)
                    except Exception as e:
                        self.logger.warning(f"处理参考图像失败，跳过: {e}")
                        continue
            
            if not image_data_urls:
                self.logger.error("没有可用的参考图像")
                return None
            
            # 估算 tokens
            estimated_tokens = self.estimate_tokens(prompt, image_data_urls)
            
            # 构建消息内容
            message_content = []
            
            # 添加文本提示词
            if seed is not None:
                full_prompt = f"{prompt}\n\nSeed: {seed}"
            else:
                full_prompt = prompt
            
            message_content.append({
                "type": "text",
                "text": full_prompt
            })
            
            # 添加参考图像
            for data_url in image_data_urls:
                message_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": data_url
                    }
                })
            
            # 构建请求数据
            request_data = {
                "model": self.model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": message_content
                    }
                ],
                "max_tokens": 4000,
                "temperature": 0.7
            }
            
            # 请求头
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/your-repo",  # 可选，用于统计
                "X-Title": "AI Image Generation Service"  # 可选，用于统计
            }
            
            self.logger.info(f"发送 OpenRouter API 请求，估算 tokens: {estimated_tokens}")
            
            # 发送异步请求
            timeout = aiohttp.ClientTimeout(total=120)  # 2分钟超时
            
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    self.api_url,
                    headers=headers,
                    json=request_data
                ) as response:
                    
                    if response.status == 429:
                        # 速率限制
                        retry_after = response.headers.get('Retry-After', '60')
                        wait_time = int(retry_after)
                        self.logger.warning(f"遇到速率限制，等待 {wait_time} 秒后重试")
                        await asyncio.sleep(wait_time)
                        
                        # 重试一次
                        async with session.post(
                            self.api_url,
                            headers=headers,
                            json=request_data
                        ) as retry_response:
                            # 替换: 使用更稳健的 text 读取和 json 解析
                            raw_text = await retry_response.text()
                            self.logger.info(f"完整API响应文本 (重试): {raw_text}") # <--- 添加日志
                            if not raw_text:
                                self.logger.error("API 响应体为空 (重试)")
                                return None
                            try:
                                response_data = json.loads(raw_text)
                            except json.JSONDecodeError:
                                self.logger.error(f"无法从响应中解析JSON (重试): {raw_text[:500]}")
                                return None
                    
                    elif response.status != 200:
                        error_text = await response.text()
                        self.logger.error(f"API 请求失败: {response.status} - {error_text}")
                        return None
                    
                    else:
                        # 替换: 使用更稳健的 text 读取和 json 解析
                        raw_text = await response.text()
                        self.logger.info(f"完整API响应文本: {raw_text}") # <--- 添加日志
                        if not raw_text:
                            self.logger.error("API 响应体为空")
                            return None
                        try:
                            response_data = json.loads(raw_text)
                        except json.JSONDecodeError:
                            self.logger.error(f"无法从响应中解析JSON: {raw_text[:500]}")
                            return None
            
            # 提取图像数据
            image_data = self._extract_image_from_response(response_data)
            
            if image_data:
                self.logger.info("图像生成成功")
                
                # 清理原始响应中的 base64 数据
                cleaned_response_data = response_data.copy() # 创建副本以避免修改原始数据
                try:
                    if 'choices' in cleaned_response_data and cleaned_response_data['choices']:
                        message = cleaned_response_data['choices'][0].get('message', {})
                        if 'images' in message and message['images']:
                            for img_item in message['images']:
                                if 'image_url' in img_item and 'url' in img_item['image_url']:
                                    img_item['image_url']['url'] = "base64_data_removed_for_brevity"
                except Exception as e:
                    self.logger.warning(f"移除 base64 数据时出错: {e}")

                # 返回的字典中，output 字段也只保留提示信息
                output_for_json = "base64_data_removed_for_brevity"

                return {
                    "status": "succeeded",
                    "output": image_data,  # 实际的 base64 数据，用于后续处理
                    "output_for_json": output_for_json, # 清理后的字段，用于保存
                    "id": cleaned_response_data.get("id"),
                    "api_type": "gemini_openrouter",
                    "raw": cleaned_response_data
                }
            else:
                self.logger.error("从响应中提取图像数据失败")
                return None
                
        except asyncio.TimeoutError:
            self.logger.error("请求超时")
            return None
        except Exception as e:
            self.logger.error(f"生成图像时出错: {e}")
            return None
    
    def generate_image_sync(self, 
                          prompt: str, 
                          reference_images: List[Union[str, bytes]] = None,
                          seed: Optional[int] = None,
                          **kwargs) -> Optional[str]:
        """
        同步生成图像（包装异步方法）
        
        Args:
            prompt: 文本提示词
            reference_images: 参考图像列表
            seed: 随机种子
            **kwargs: 其他参数
        
        Returns:
            生成的图像数据
        """
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(
            self.generate_image(prompt, reference_images, seed, **kwargs)
        )
