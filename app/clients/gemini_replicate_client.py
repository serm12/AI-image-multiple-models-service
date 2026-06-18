"""
Gemini Replicate API客户端
基于Replicate平台的Google Nano Banana模型
"""

import os
import json
import asyncio
import aiofiles
import aiohttp
from datetime import datetime
from PIL import Image
import io
import replicate
from app.core.config import APIConfig, ArtStyleEnum

class GeminiReplicateClient:
    """Gemini Replicate API客户端"""
    
    def __init__(self, api_key=None):
        """初始化客户端"""
        self.api_key = api_key or APIConfig.REPLICATE_API_TOKEN
        if not self.api_key:
            raise ValueError("REPLICATE_API_TOKEN not found in environment variables")
        
        # 设置Replicate API密钥
        os.environ["REPLICATE_API_TOKEN"] = self.api_key
        # 注意：需要确认正确的Replicate模型名称
        # 可能的模型名称：google/imagen-3.0, google/imagen, 或其他Gemini相关模型
        self.model_name = "google/nano-banana"  # 请确认这是正确的模型名称
    
    def _get_gemini_replicate_style_prompt(self, art_style):
        """获取Gemini Replicate风格提示词"""
        # 直接使用 Gemini Google 的风格提示词
        style_mapping = {
            "oil_painting": ArtStyleEnum.gemini_oil_painting,
            "flux_oil_painting": ArtStyleEnum.gemini_oil_painting,
            "illustration": ArtStyleEnum.gemini_illustration,
            "watercolor": ArtStyleEnum.gemini_watercolor,
            "flux_watercolor": ArtStyleEnum.gemini_watercolor,
            "acrylic": ArtStyleEnum.gemini_acrylic,
            "flux_acrylic": ArtStyleEnum.gemini_acrylic,
        }
        
        mapped_style = style_mapping.get(art_style, ArtStyleEnum.gemini_oil_painting)
        return mapped_style.value
    
    def _compress_image_for_gemini_replicate(self, image_path, target_width=512):
        """压缩图片用于Gemini Replicate上传"""
        try:
            # 获取原始文件大小
            original_file_size = os.path.getsize(image_path)
            
            with Image.open(image_path) as img:
                # 计算新的高度，保持宽高比
                width, height = img.size
                new_height = int((target_width / width) * height)
                
                # 调整图片大小
                resized_img = img.resize((target_width, new_height), Image.Resampling.LANCZOS)
                
                # 转换为RGB模式（JPEG不支持RGBA）
                if resized_img.mode == 'RGBA':
                    # 创建白色背景
                    rgb_img = Image.new('RGB', resized_img.size, (255, 255, 255))
                    rgb_img.paste(resized_img, mask=resized_img.split()[-1])  # 使用alpha通道作为mask
                    resized_img = rgb_img
                
                # 创建临时文件
                temp_path = f"{image_path}_compressed_{target_width}.jpg"
                resized_img.save(temp_path, "JPEG", quality=85, optimize=True)
                
                # 计算压缩后的大小和压缩比例
                compressed_file_size = os.path.getsize(temp_path)
                compression_ratio = (1 - compressed_file_size / original_file_size) * 100
                size_reduction = original_file_size - compressed_file_size
                
                print(f"📦 Gemini Replicate图片压缩完成:")
                print(f"   📏 尺寸: {width}x{height} → {target_width}x{new_height}")
                print(f"   📁 文件大小: {original_file_size:,} bytes → {compressed_file_size:,} bytes")
                print(f"   📉 压缩比例: {compression_ratio:.1f}% (减少 {size_reduction:,} bytes)")
                print(f"   🚀 上传速度提升预估: ~{compression_ratio:.1f}%")
                
                return temp_path
        except Exception as e:
            print(f"⚠️ 图片压缩失败: {e}")
            original_file_size = os.path.getsize(image_path)
            print(f"📁 使用原始文件: {original_file_size:,} bytes (未压缩)")
            return image_path
    
    async def _upload_image_to_replicate(self, image_path):
        """将本地图像上传到Replicate并返回URL"""
        try:
            print(f"📤 正在上传参考图像到Replicate: {image_path}")
            
            # 异步运行阻塞的文件上传
            def upload_sync():
                with open(image_path, "rb") as image_file:
                    return replicate.files.create(image_file)

            uploaded_file = await asyncio.to_thread(upload_sync)
            
            # 获取上传文件的URL
            uploaded_image_url = uploaded_file.urls['get']
            
            print(f"✅ 图像上传成功: {uploaded_image_url}")
            return uploaded_image_url
            
        except Exception as e:
            print(f"❌ 图像上传失败: {e}")
            return None
    
    async def generate_image(self, prompt, input_image_paths=None, input_image_urls=None, seed=None, art_style=None):
        """
        使用Gemini Replicate生成图像
        
        Args:
            prompt: 提示词
            input_image_paths: 本地参考图像路径列表
            input_image_urls: 在线参考图像URL列表
            seed: 随机种子（Gemini Replicate支持）
            art_style: 艺术风格
            
        Returns:
            dict: 生成结果
        """
        try:
            # 处理艺术风格
            if art_style is None:
                art_style = "gemini_replicate_oil_painting"
            
            style_value = art_style if isinstance(art_style, str) else art_style.value
            style_prompt = self._get_gemini_replicate_style_prompt(style_value)
            
            # 完整提示词
            full_prompt = prompt + style_prompt
            
            print(f"🎨 Gemini Replicate生成图像")
            print(f"📝 提示词: {prompt}")
            print(f"🎨 风格: {style_prompt}")
            
            # 处理seed
            actual_seed = seed if seed is not None else 12345
            print(f"🎯 使用种子: {actual_seed}")
            
            # 处理参考图像 - 支持多张图片
            reference_data_list = []
            
            if input_image_paths and isinstance(input_image_paths, list):
                print(f"📷 使用 {len(input_image_paths)} 张参考图像")
                
                for i, img_path in enumerate(input_image_paths):
                    if os.path.exists(img_path):
                        print(f"   📸 处理参考图 {i+1}: {os.path.basename(img_path)}")
                        # 压缩图片
                        compressed_path = self._compress_image_for_gemini_replicate(img_path, target_width=512)
                        # 上传到Replicate
                        reference_data = await self._upload_image_to_replicate(compressed_path)
                        if reference_data:
                            reference_data_list.append(reference_data)
                        
                        # 清理临时压缩文件
                        if compressed_path != img_path and os.path.exists(compressed_path):
                            try:
                                os.remove(compressed_path)
                                print(f"      🧹 已清理临时压缩文件: {compressed_path}")
                            except:
                                pass
                        print(f"      ✅ 添加参考图 {i+1}: {os.path.basename(img_path)}")
                    else:
                        print(f"   ⚠️ 参考图 {i+1} 不存在: {img_path}")
                        
            elif input_image_paths and not isinstance(input_image_paths, list):
                # 向后兼容：如果传入的是单个路径字符串
                print(f"📷 使用单张参考图像: {input_image_paths}")
                if os.path.exists(input_image_paths):
                    compressed_path = self._compress_image_for_gemini_replicate(input_image_paths, target_width=512)
                    reference_data = await self._upload_image_to_replicate(compressed_path)
                    if reference_data:
                        reference_data_list.append(reference_data)
                    
                    # 清理临时压缩文件
                    if compressed_path != input_image_paths and os.path.exists(compressed_path):
                        try:
                            os.remove(compressed_path)
                            print(f"🧹 已清理临时压缩文件: {compressed_path}")
                        except:
                            pass
            
            if input_image_urls:
                print("⚠️ Gemini Replicate客户端暂不支持input_image_urls，将忽略该参数")
            
            # 准备输入参数
            input_params = {
                "prompt": full_prompt,
            }
            
            # 添加参考图像
            if reference_data_list:
                input_params["image_input"] = reference_data_list
            
            # 添加seed
            if actual_seed is not None:
                input_params["seed"] = actual_seed
            
            print("🎨 正在使用Replicate nano-banana模型生成图像...")
            
            # 异步运行阻塞的replicate.run
            output = await asyncio.to_thread(
                replicate.run,
                self.model_name,
                input=input_params
            )
            
            print(f"🔍 Replicate API 返回类型: {type(output)}")
            print(f"🔍 Replicate API 返回内容: {output}")
            
            if not output:
                raise ValueError("Gemini Replicate API未返回有效内容")
            
            # 处理输出结果
            image_url = None
            if isinstance(output, str):
                print("📝 输出是字符串类型")
                image_url = output
            elif isinstance(output, list) and len(output) > 0:
                print(f"📝 输出是列表类型，长度: {len(output)}")
                print(f"📝 列表第一个元素类型: {type(output[0])}")
                print(f"📝 列表第一个元素内容: {output[0]}")
                
                # 处理列表中的元素
                first_item = output[0]
                if isinstance(first_item, str):
                    image_url = first_item
                elif hasattr(first_item, 'url'):
                    if callable(first_item.url):
                        image_url = first_item.url()
                    else:
                        image_url = str(first_item.url)
                else:
                    image_url = str(first_item)
                    
            elif hasattr(output, 'url') and callable(output.url):
                print("📝 输出有url方法")
                image_url = output.url()
            elif hasattr(output, 'url'):
                print("📝 输出有url属性")
                image_url = str(output.url)
            else:
                print(f"❌ 未识别的输出格式: {type(output)}")
                print(f"❌ 输出属性: {dir(output) if hasattr(output, '__dict__') else 'No attributes'}")
                # 尝试转换为字符串
                try:
                    image_url = str(output)
                    print(f"🔄 尝试转换为字符串: {image_url}")
                except:
                    raise ValueError("Gemini Replicate API响应中未找到图像数据")
            
            print(f"✅ Gemini Replicate图像生成成功")
            print(f"📥 图像URL: {image_url}")
            
            # 生成任务ID
            prediction_id = f"gemini_replicate_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{actual_seed}"
            
            return {
                "id": prediction_id,
                "status": "succeeded",
                "output": image_url,
                "logs": f"Gemini Replicate生成 - 种子: {actual_seed}",
                "input": {
                    "prompt": full_prompt,
                    "seed": actual_seed,
                    "style": style_value
                },
                "api_type": "gemini_replicate",
                "extracted_seed": actual_seed  # Gemini Replicate返回实际使用的seed
            }
            
        except Exception as e:
            print(f"❌ Gemini Replicate生成图像失败: {e}")
            raise ValueError(f"Gemini Replicate API调用失败: {e}")

# 测试函数
if __name__ == "__main__":
    async def test():
        gemini_replicate_client = GeminiReplicateClient()
        result = await gemini_replicate_client.generate_image(
            prompt="A beautiful landscape",
            art_style="gemini_replicate_oil_painting",
            seed=12345
        )
        print(result)
    
    asyncio.run(test()) 
