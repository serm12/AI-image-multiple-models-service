#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gemini API客户端
基于提供的Gemini代码集成到统一API框架中
"""

import os
import base64
import json
import random
import asyncio
from datetime import datetime
from google import genai
from google.genai import types
from PIL import Image
import io
from app.core.config import APIConfig, STYLE_PROMPTS, ArtStyleEnum

class GeminiClient:
    """Gemini API客户端"""
    
    def __init__(self, api_key=None):
        # 优先使用传入的api_key，否则使用配置文件中的
        self.api_key = api_key or APIConfig.GOOGLE_GEMINI_API_KEY
        if not self.api_key:
            raise ValueError("GOOGLE_GEMINI_API_KEY not found in environment variables or passed as parameter")
        
        # 配置客户端
        self.client = genai.Client(api_key=self.api_key)
        self.model_name = "gemini-2.5-flash-image-preview"
    
    def _get_gemini_style_prompt(self, art_style):
        """获取Gemini风格提示词"""
        # 如果是Gemini风格，直接使用
        if hasattr(ArtStyleEnum, art_style) and art_style.startswith('gemini_'):
            return STYLE_PROMPTS.get(getattr(ArtStyleEnum, art_style), "")
        
        # 如果是传统风格名称，映射到对应的Gemini风格
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
        return STYLE_PROMPTS.get(mapped_style, "")
    
    def _compress_image_for_gemini(self, image_path, target_width=256):
        """
        将图片压缩到指定宽度以减少Token消耗
        
        Args:
            image_path: 图片路径
            target_width: 目标宽度(默认256px，适合头像识别)
        
        Returns:
            压缩后的图片二进制数据
        """
        try:
            with Image.open(image_path) as img:
                # 获取原始尺寸
                orig_width, orig_height = img.size
                print(f"📷 原始图片: {orig_width}x{orig_height}")
                
                # 如果图片宽度已经小于等于目标宽度，直接使用
                if orig_width <= target_width:
                    print(f"✅ 图片已经够小，无需压缩")
                    with open(image_path, "rb") as f:
                        return f.read()
                
                # 计算缩放比例
                scale_factor = target_width / orig_width
                new_height = int(orig_height * scale_factor)
                
                # 转换为RGB模式（如果需要）
                if img.mode in ('RGBA', 'P'):
                    img = img.convert('RGB')
                
                # 缩放图片
                resized_img = img.resize((target_width, new_height), Image.Resampling.LANCZOS)
                
                # 保存到内存缓冲区
                buffer = io.BytesIO()
                resized_img.save(buffer, format='JPEG', quality=85, optimize=True)
                compressed_data = buffer.getvalue()
                
                # 计算压缩效果
                original_size = os.path.getsize(image_path)
                compressed_size = len(compressed_data)
                compression_ratio = (1 - compressed_size / original_size) * 100
                
                # 计算详细的压缩统计
                size_reduction = original_size - compressed_size
                token_savings = ((original_size - compressed_size) // 1024) * 13  # 估算每KB约13个token
                
                print(f"📦 Gemini Google图片压缩完成:")
                print(f"   📏 尺寸: {orig_width}x{orig_height} → {target_width}x{new_height}")
                print(f"   📁 文件大小: {original_size:,} bytes → {compressed_size:,} bytes") 
                print(f"   📉 压缩比例: {compression_ratio:.1f}% (减少 {size_reduction:,} bytes)")
                print(f"   💰 Token节省预估: ~{token_savings:,} tokens (节省约 ${token_savings * 0.000125:.4f})")
                
                return compressed_data
                
        except Exception as e:
            print(f"⚠️ 图片压缩失败，使用原图: {e}")
            # 如果压缩失败，返回原图
            with open(image_path, "rb") as f:
                return f.read()

    def _estimate_token_usage(self, compressed_data, prompt_text):
        """
        估算Token使用量
        
        Args:
            compressed_data: 压缩后的图片数据
            prompt_text: 提示词文本
            
        Returns:
            dict: Token使用详情
        """
        # 图片Token估算：Base64编码后，每KB约13个token
        image_size_kb = len(compressed_data) / 1024
        base64_size_kb = image_size_kb * 1.33  # Base64增加约33%
        image_tokens = int(base64_size_kb * 13)
        
        # 文本Token估算：大约每4个字符1个token
        text_tokens = len(prompt_text) // 4
        
        total_tokens = image_tokens + text_tokens
        
        return {
            "image_size_kb": round(image_size_kb, 2),
            "image_tokens": image_tokens,
            "text_tokens": text_tokens,
            "total_tokens": total_tokens,
            "estimated_cost": total_tokens * 0.000001  # $0.000001 per token
        }

    def _log_token_usage(self, token_info, task_id=None):
        """
        记录Token使用情况到日志文件
        
        Args:
            token_info: Token使用详情
            task_id: 任务ID（可选）
        """
        import json
        from datetime import datetime
        
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "task_id": task_id,
            "model": "gemini-2.5-flash-image-preview",
            "image_size_kb": token_info["image_size_kb"],
            "image_tokens": token_info["image_tokens"],
            "text_tokens": token_info["text_tokens"],
            "total_tokens": token_info["total_tokens"],
            "estimated_cost": token_info["estimated_cost"]
        }
        
        # 写入日志文件
        log_file = "gemini_token_usage.json"
        try:
            # 读取现有日志
            if os.path.exists(log_file):
                with open(log_file, 'r', encoding='utf-8') as f:
                    logs = json.load(f)
            else:
                logs = []
            
            # 添加新记录
            logs.append(log_entry)
            
            # 保存日志（保留最近100条）
            if len(logs) > 100:
                logs = logs[-100:]
                
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(logs, f, indent=2, ensure_ascii=False)
                
            print(f"💾 Token使用记录已保存到: {log_file}")
            
        except Exception as e:
            print(f"⚠️ Token日志记录失败: {e}")

    async def generate_image(self, prompt, input_image_paths=None, input_image_urls=None, 
                           art_style=None, seed=None):
        """
        使用Gemini生成图像
        
        Args:
            prompt: 基础提示词
            input_image_paths: 输入图片路径列表（本地文件）
            input_image_urls: 输入图片URL列表（暂不支持）
            art_style: 艺术风格
            seed: 随机种子（Gemini会自动生成）
            
        Returns:
            dict: 生成结果
        """
        
        try:
            # 获取风格提示词
            style_prompt = ""
            if art_style:
                # 如果是枚举类型，获取其值
                if hasattr(art_style, 'value'):
                    style_value = art_style.value
                else:
                    style_value = str(art_style)
                style_prompt = self._get_gemini_style_prompt(style_value)
            
            # 组合完整提示词
            full_prompt = prompt + style_prompt
            
            print(f"🎨 Gemini生成图像")
            print(f"📝 完整提示词: {full_prompt}")
            
            # Gemini自己生成seed，我们只是记录输入的seed用于日志
            if seed is None:
                actual_seed = random.randint(1, 999999)
                print(f"🎯 生成随机种子用于记录: {actual_seed}")
            else:
                actual_seed = seed
                print(f"🎯 记录种子: {actual_seed} (注意：Gemini会自己生成实际使用的seed)")
            
            # 准备内容部分
            content_parts = [types.Part.from_text(text=full_prompt)]
            
            # 处理参考图像 - 支持多张图片输入
            total_image_tokens = 0
            total_image_size = 0
            
            if input_image_paths and isinstance(input_image_paths, list):
                print(f"📷 使用 {len(input_image_paths)} 张参考图像")
                
                for i, img_path in enumerate(input_image_paths):
                    if os.path.exists(img_path):
                        print(f"   📸 处理参考图 {i+1}: {os.path.basename(img_path)}")
                        # 自动压缩图片以降低Token消耗
                        reference_data = self._compress_image_for_gemini(img_path, target_width=256)
                        
                        # 估算Token使用
                        token_info = self._estimate_token_usage(reference_data, "")
                        total_image_tokens += token_info['image_tokens']
                        total_image_size += token_info['image_size_kb']
                        
                        content_parts.append(
                            types.Part.from_bytes(data=reference_data, mime_type="image/jpeg")
                        )
                        print(f"      ✅ 添加参考图 {i+1}: {os.path.basename(img_path)}")
                    else:
                        print(f"   ⚠️ 参考图 {i+1} 不存在: {img_path}")
                
                # 显示总体Token使用情况
                total_tokens = total_image_tokens + self._estimate_token_usage(b"", full_prompt)['text_tokens']
                total_cost = (total_tokens / 1000000) * 0.0005  # Gemini定价
                
                print(f"📊 多图Token使用统计:")
                print(f"   图片总数: {len([p for p in input_image_paths if os.path.exists(p)])}")
                print(f"   图片Token: {total_image_tokens} tokens ({total_image_size:.1f}KB)")
                print(f"   文本Token: {self._estimate_token_usage(b'', full_prompt)['text_tokens']} tokens")
                print(f"   总计Token: {total_tokens} tokens")
                print(f"   估算成本: ${total_cost:.6f}")
                
            elif input_image_paths and not isinstance(input_image_paths, list):
                # 向后兼容：如果传入的是单个路径字符串
                print(f"📷 使用单张参考图像: {input_image_paths}")
                if os.path.exists(input_image_paths):
                    reference_data = self._compress_image_for_gemini(input_image_paths, target_width=256)
                    token_info = self._estimate_token_usage(reference_data, full_prompt)
                    print(f"📊 估算Token使用:")
                    print(f"   图片: {token_info['image_tokens']} tokens ({token_info['image_size_kb']}KB)")
                    print(f"   文本: {token_info['text_tokens']} tokens")
                    print(f"   总计: {token_info['total_tokens']} tokens")
                    print(f"   估算成本: ${token_info['estimated_cost']:.6f}")
                    
                    content_parts.append(
                        types.Part.from_bytes(data=reference_data, mime_type="image/jpeg")
                    )
            
            if input_image_urls:
                print("⚠️ Gemini客户端暂不支持input_image_urls，将忽略该参数")
            
            print("🎨 正在生成图像...")
            
            # 配置参数 - 注意：Gemini的seed是自动生成的
            config_params = {}
            if actual_seed is not None:
                config_params["seed"] = actual_seed
            
            # 调用API - 将同步调用移到线程中执行
            response = await asyncio.to_thread(
                self.client.models.generate_content,
                model=self.model_name,
                contents=[types.Content(role="user", parts=content_parts)],
                config=types.GenerateContentConfig(**config_params),
            )
            
            # 处理响应
            if not response.candidates or not response.candidates[0].content.parts:
                raise ValueError("Gemini API未返回有效内容")
            
            # 查找图像数据
            image_data = None
            for part in response.candidates[0].content.parts:
                if part.inline_data:
                    image_data = part.inline_data.data
                    break
            
            if not image_data:
                raise ValueError("Gemini API响应中未找到图像数据")
            
            # 处理图像数据编码
            binary_data = None
            if isinstance(image_data, bytes):
                # 检查是否为Base64编码的字节数据
                try:
                    test_str = image_data[:100].decode('ascii', errors='ignore')
                    if test_str.startswith('iVBOR') or test_str.startswith('/9j/') or test_str.startswith('UklG'):
                        # Base64编码数据
                        binary_data = base64.b64decode(image_data)
                    else:
                        # 直接二进制数据
                        binary_data = image_data
                except Exception:
                    binary_data = image_data
            elif isinstance(image_data, str):
                # 字符串格式的Base64
                binary_data = base64.b64decode(image_data)
            
            if not binary_data:
                raise ValueError("无法解析图像数据")
            
            # 转换为data URL格式返回
            b64_encoded = base64.b64encode(binary_data).decode('utf-8')
            data_url = f"data:image/png;base64,{b64_encoded}"
            
            print(f"✅ Gemini图像生成成功")
            print(f"📊 图像数据大小: {len(binary_data):,} 字节")
            
            # 记录Token使用到日志文件
            if 'token_info' in locals():
                self._log_token_usage(token_info, task_id=f"gemini_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{actual_seed}")
            
            # 创建响应记录
            prediction_id = f"gemini_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{actual_seed}"
            
            return {
                "status": "succeeded",
                "output": data_url,
                "id": prediction_id,
                "logs": f"Gemini生成 - 记录种子: {actual_seed} (实际种子由Gemini自动生成)",
                "raw": {
                    "model": self.model_name,
                    "input_seed": actual_seed,  # 用户输入的种子（如果有）
                    "gemini_generated_seed": True,  # 标识这是Gemini自动生成的种子
                    "full_prompt": full_prompt,
                    "art_style": art_style.value if hasattr(art_style, 'value') else str(art_style),
                    "image_size_bytes": len(binary_data)
                },
                "api_type": "gemini_google",
                "extracted_seed": None  # Gemini不返回实际使用的seed
            }
            
        except Exception as e:
            print(f"❌ Gemini生成图像失败: {e}")
            raise ValueError(f"Gemini API调用失败: {e}")

# 导出实例
gemini_client = GeminiClient() 
