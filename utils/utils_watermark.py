from PIL import Image, ImageDraw, ImageFont
import math
import os
import numpy as np
import random
from config import WatermarkConfig

# 添加logo缓存
_logo_cache = {}

def blend_images(background, overlay, mode="normal"):
    """
    图像混合模式处理函数
    
    Args:
        background: 背景图像 (PIL Image, RGBA)
        overlay: 叠加图像 (PIL Image, RGBA) 
        mode: 混合模式 ("normal", "screen", "multiply", "overlay")
    
    Returns:
        PIL Image: 混合后的图像
    """
    if mode == "normal":
        return Image.alpha_composite(background, overlay)
    
    # 转换为numpy数组进行混合运算
    bg_array = np.array(background, dtype=np.float32) / 255.0
    ov_array = np.array(overlay, dtype=np.float32) / 255.0
    
    # 获取alpha通道
    bg_alpha = bg_array[:, :, 3:4]
    ov_alpha = ov_array[:, :, 3:4]
    
    # RGB通道
    bg_rgb = bg_array[:, :, :3]
    ov_rgb = ov_array[:, :, :3]
    
    # 应用混合模式
    if mode == "screen":
        # 滤色模式: 1 - (1 - bg) * (1 - ov)
        blended_rgb = 1 - (1 - bg_rgb) * (1 - ov_rgb)
    elif mode == "multiply":
        # 正片叠底模式: bg * ov
        blended_rgb = bg_rgb * ov_rgb
    elif mode == "overlay":
        # 叠加模式: 根据背景亮度选择正片叠底或滤色
        mask = bg_rgb < 0.5
        multiply_result = 2 * bg_rgb * ov_rgb
        screen_result = 1 - 2 * (1 - bg_rgb) * (1 - ov_rgb)
        blended_rgb = np.where(mask, multiply_result, screen_result)
    else:
        # 默认为normal模式
        blended_rgb = bg_rgb
    
    # Alpha混合
    result_alpha = bg_alpha + ov_alpha * (1 - bg_alpha)
    
    # 防止除零
    alpha_mask = result_alpha > 0
    result_rgb = np.where(
        alpha_mask,
        (bg_rgb * bg_alpha * (1 - ov_alpha) + blended_rgb * ov_alpha) / result_alpha,
        blended_rgb
    )
    
    # 合并RGB和Alpha通道
    result_array = np.concatenate([result_rgb, result_alpha], axis=2)
    
    # 转换回PIL图像
    result_array = np.clip(result_array * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(result_array, mode="RGBA")

def add_invisible_watermark_layer(image, text, font_path, font_size):
    """添加近乎不可见但难以去除的水印层"""
    if not WatermarkConfig.ENABLE_RANDOM_PATTERNS:
        return image
        
    watermark = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(watermark)
    
    try:
        font = ImageFont.truetype(font_path, font_size // 2)
    except:
        font = ImageFont.load_default()
    
    # 超低透明度文字
    for y in range(0, image.height, 300):
        for x in range(0, image.width, 300):
            # 极低透明度，几乎不可见但存在
            draw.text((x, y), text, font=font, fill=(255, 255, 255, 3))
    
    # 使用multiply模式，几乎不影响视觉效果
    return blend_images(image, watermark, "multiply")

def add_text_watermark(
    input_path,
    output_path,
    text=None,
    font_path=None,
    font_size=None,
    color=None,
    angle=None,
    step=None,
    alpha=30,
    blend_mode=None,
    resize_scale=None,
    add_brand=True
):
    # 使用配置文件的默认值
    text = text or WatermarkConfig.DEFAULT_TEXT
    font_path = font_path or WatermarkConfig.DEFAULT_FONT_PATH
    font_size = font_size or WatermarkConfig.DEFAULT_FONT_SIZE
    color = color or WatermarkConfig.DEFAULT_COLOR
    angle = angle or WatermarkConfig.DEFAULT_ANGLE
    step = step or WatermarkConfig.DEFAULT_STEP
    blend_mode = blend_mode or WatermarkConfig.DEFAULT_BLEND_MODE
    resize_scale = resize_scale or WatermarkConfig.get_resize_scale()
    image = Image.open(input_path).convert("RGBA")
    width, height = image.size
    watermark = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(watermark)
    try:
        font = ImageFont.truetype(font_path, font_size)
    except Exception as e:
        font = ImageFont.load_default()
        print("使用默认字体，字号不可调，错误信息：", e)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    diagonal = int(math.hypot(width, height))
    step_x = text_width + font_size // 8  # 减小X方向间距，增加密度
    step_y = text_height + font_size // 8  # 减小Y方向间距，增加密度
    for y in range(-diagonal, diagonal, step_y):
        for x in range(-diagonal, diagonal, step_x):
            draw.text((x, y), text, font=font, fill=(color[0], color[1], color[2], alpha))
    watermark = watermark.rotate(angle, expand=1)
    watermark = watermark.crop((0, 0, width, height))
    watermarked = blend_images(image, watermark, blend_mode)
    
    if add_brand:
        # 保存临时文件以添加品牌logo
        temp_path = output_path.replace(".png", "_temp.png").replace(".jpg", "_temp.jpg")
        if output_path.lower().endswith("jpg"):
            watermarked = watermarked.convert("RGB")
        else:
            watermarked = watermarked.convert("RGBA")
        watermarked.save(temp_path)
        
        # 添加右下角品牌logo，但不直接保存到最终输出路径
        temp_final_path = output_path.replace(".png", "_temp_final.png").replace(".jpg", "_temp_final.jpg")
        add_brand_logo(temp_path, temp_final_path)
        
        # 应用缩放
        if resize_scale != 1.0:
            final_image = Image.open(temp_final_path)
            original_width, original_height = final_image.size
            new_width = int(original_width * resize_scale)
            new_height = int(original_height * resize_scale)
            
            resized_image = final_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            if output_path.lower().endswith((".jpg", ".jpeg")):
                resized_image = resized_image.convert("RGB")
            
            resized_image.save(output_path)
            print(f"图片已缩放至 {resize_scale*100}% ({new_width}x{new_height})")
        else:
            import shutil
            shutil.move(temp_final_path, output_path)
        
        # 删除临时文件
        if os.path.exists(temp_path):
            os.remove(temp_path)
        if os.path.exists(temp_final_path):
            os.remove(temp_final_path)
    else:
        # 直接保存，应用缩放
        if resize_scale != 1.0:
            original_width, original_height = watermarked.size
            new_width = int(original_width * resize_scale)
            new_height = int(original_height * resize_scale)
            
            watermarked = watermarked.resize((new_width, new_height), Image.Resampling.LANCZOS)
            print(f"图片已缩放至 {resize_scale*100}% ({new_width}x{new_height})")
        
        if output_path.lower().endswith("jpg"):
            watermarked = watermarked.convert("RGB")
        else:
            watermarked = watermarked.convert("RGBA")
        watermarked.save(output_path)

def create_logo_watermark(
    text=None,
    font_path=None,
    font_size=None,
    padding=None,
    color=None,
    out_path=None
):
    # 使用配置文件的默认值
    text = text or WatermarkConfig.DEFAULT_TEXT
    font_path = font_path or WatermarkConfig.DEFAULT_FONT_PATH
    font_size = font_size or WatermarkConfig.DEFAULT_FONT_SIZE
    padding = padding or WatermarkConfig.DEFAULT_PADDING
    color = color or WatermarkConfig.DEFAULT_COLOR
    out_path = out_path or WatermarkConfig.LOGO_PATH
    if os.path.exists(out_path):
        print(f"水印logo已存在: {out_path}")
        return
    try:
        font = ImageFont.truetype(font_path, font_size)
        print(f"使用自带字体: {font_path}, 字号: {font_size}")
    except IOError as e:
        font = ImageFont.load_default()
        print("使用默认字体，字号不可调，错误信息：", e)
    dummy_img = Image.new("RGBA", (1, 1))
    dummy_draw = ImageDraw.Draw(dummy_img)
    bbox = dummy_draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    logo_width = text_width + 2 * padding
    logo_height = text_height + 2 * padding
    logo = Image.new("RGBA", (logo_width, logo_height), (0,0,0,0))
    draw = ImageDraw.Draw(logo)
    draw.text((padding, padding), text, font=font, fill=color)
    logo.save(out_path)
    print(f"logo水印已保存为: {out_path}")

def draw_rounded_rectangle(draw, coords, radius, fill=None, outline=None, width=0):
    """绘制圆角矩形"""
    x1, y1, x2, y2 = coords
    
    # 绘制背景（如果指定）
    if fill:
        # 绘制主体矩形
        draw.rectangle([x1 + radius, y1, x2 - radius, y2], fill=fill)
        draw.rectangle([x1, y1 + radius, x2, y2 - radius], fill=fill)
        
        # 绘制四个圆角
        draw.pieslice([x1, y1, x1 + 2*radius, y1 + 2*radius], 180, 270, fill=fill)  # 左上
        draw.pieslice([x2 - 2*radius, y1, x2, y1 + 2*radius], 270, 360, fill=fill)  # 右上
        draw.pieslice([x1, y2 - 2*radius, x1 + 2*radius, y2], 90, 180, fill=fill)   # 左下
        draw.pieslice([x2 - 2*radius, y2 - 2*radius, x2, y2], 0, 90, fill=fill)     # 右下
    
    # 绘制边框（如果指定）
    if outline and width > 0:
        # 绘制四条边
        draw.line([x1 + radius, y1, x2 - radius, y1], fill=outline, width=width)  # 上边
        draw.line([x1 + radius, y2, x2 - radius, y2], fill=outline, width=width)  # 下边
        draw.line([x1, y1 + radius, x1, y2 - radius], fill=outline, width=width)  # 左边
        draw.line([x2, y1 + radius, x2, y2 - radius], fill=outline, width=width)  # 右边
        
        # 绘制四个圆角边框
        draw.arc([x1, y1, x1 + 2*radius, y1 + 2*radius], 180, 270, fill=outline, width=width)  # 左上
        draw.arc([x2 - 2*radius, y1, x2, y1 + 2*radius], 270, 360, fill=outline, width=width)  # 右上
        draw.arc([x1, y2 - 2*radius, x1 + 2*radius, y2], 90, 180, fill=outline, width=width)   # 左下
        draw.arc([x2 - 2*radius, y2 - 2*radius, x2, y2], 0, 90, fill=outline, width=width)     # 右下

def add_corner_label(
    input_image_path, 
    output_image_path, 
    text=None,
    font_path=None,
    font_size=None,
    padding=None,
    margin=None,
    text_color=None,
    bg_color=None,
    border_color=None,
    border_width=None,
    corner_radius=None
):
    """在图片左上角添加带圆角外框的白色文字"""
    # 使用配置文件的默认值
    text = text or WatermarkConfig.CORNER_LABEL_TEXT
    font_path = font_path or WatermarkConfig.CORNER_LABEL_FONT_PATH
    font_size = font_size or WatermarkConfig.CORNER_LABEL_FONT_SIZE
    padding = padding or WatermarkConfig.CORNER_LABEL_PADDING
    margin = margin or WatermarkConfig.CORNER_LABEL_MARGIN
    text_color = text_color or WatermarkConfig.CORNER_LABEL_TEXT_COLOR
    bg_color = bg_color or WatermarkConfig.CORNER_LABEL_BG_COLOR
    border_color = border_color or WatermarkConfig.CORNER_LABEL_BORDER_COLOR
    border_width = border_width or WatermarkConfig.CORNER_LABEL_BORDER_WIDTH
    corner_radius = corner_radius or WatermarkConfig.CORNER_LABEL_CORNER_RADIUS
    
    # 打开图片
    image = Image.open(input_image_path).convert("RGBA")
    width, height = image.size
    
    # 创建绘图对象
    draw = ImageDraw.Draw(image)
    
    # 加载字体
    try:
        font = ImageFont.truetype(font_path, font_size)
    except Exception as e:
        font = ImageFont.load_default()
        print(f"使用默认字体，字号不可调，错误信息：{e}")
    
    # 计算文字尺寸
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    # 计算外框尺寸和位置（左上角）
    box_width = text_width + 2 * padding
    box_height = text_height + 2 * padding
    box_x = margin
    box_y = margin
    
    # 绘制圆角矩形（背景和边框）
    box_coords = [box_x, box_y, box_x + box_width, box_y + box_height]
    draw_rounded_rectangle(
        draw, 
        box_coords, 
        corner_radius, 
        fill=bg_color, 
        outline=border_color, 
        width=border_width
    )
    
    # 计算文字位置（在框内居中）
    text_x = box_x + padding
    text_y = box_y + padding
    
    # 绘制文字
    draw.text((text_x, text_y), text, font=font, fill=text_color)
    
    # 保存图片
    if output_image_path.lower().endswith(".jpg") or output_image_path.lower().endswith(".jpeg"):
        image = image.convert("RGB")
    
    image.save(output_image_path)
    print(f"左上角标识添加完成: {output_image_path}")

def add_logo_watermark(input_image_path, output_image_path, logo_path=None, step=None, angle=None, blend_mode=None, resize_scale=None):
    # 使用配置文件的默认值
    logo_path = logo_path or WatermarkConfig.LOGO_PATH
    step = step or WatermarkConfig.DEFAULT_STEP
    angle = angle or WatermarkConfig.DEFAULT_ANGLE
    blend_mode = blend_mode or WatermarkConfig.DEFAULT_BLEND_MODE
    resize_scale = resize_scale or WatermarkConfig.get_resize_scale()
    
    image = Image.open(input_image_path).convert("RGBA")
    logo = Image.open(logo_path).convert("RGBA")
    
    # 创建多层随机化水印
    watermark_layer = Image.new("RGBA", image.size, (0,0,0,0))
    
    # 生成随机种子（基于图片内容，确保同图片结果一致）
    random.seed(hash(str(image.size)) % 2147483647)
    
    if WatermarkConfig.ENABLE_RANDOM_PATTERNS:
        # 随机化处理
        for y in range(0, image.height, step):
            for x in range(0, image.width, step):
                # 随机角度
                random_angle = angle + random.randint(-WatermarkConfig.RANDOM_ANGLE_RANGE, WatermarkConfig.RANDOM_ANGLE_RANGE)
                
                # 随机尺寸
                size_factor = 1.0 + random.uniform(-WatermarkConfig.RANDOM_SIZE_RANGE, WatermarkConfig.RANDOM_SIZE_RANGE)
                new_size = (int(logo.width * size_factor), int(logo.height * size_factor))
                
                # 随机透明度
                opacity_factor = 1.0 + random.uniform(-WatermarkConfig.RANDOM_OPACITY_RANGE/255, WatermarkConfig.RANDOM_OPACITY_RANGE/255)
                
                # 处理logo
                temp_logo = logo.resize(new_size, Image.Resampling.LANCZOS)
                temp_logo = temp_logo.rotate(random_angle, expand=1)
                
                # 调整透明度
                if temp_logo.mode == 'RGBA':
                    alpha = temp_logo.split()[-1]
                    alpha = alpha.point(lambda p: max(0, min(255, int(p * opacity_factor))))
                    temp_logo.putalpha(alpha)
                
                # 随机位置微调
                offset_x = random.randint(-10, 10)
                offset_y = random.randint(-10, 10)
                
                watermark_layer.paste(temp_logo, (x + offset_x, y + offset_y), temp_logo)
    else:
        # 标准处理
        logo = logo.rotate(angle, expand=1)
        for y in range(0, image.height, step):
            for x in range(0, image.width, step):
                watermark_layer.paste(logo, (x, y), logo)
    
    # 随机选择混合模式
    if WatermarkConfig.ENABLE_RANDOM_PATTERNS and hasattr(WatermarkConfig, 'MULTIPLE_BLEND_MODES'):
        blend_mode = random.choice(WatermarkConfig.MULTIPLE_BLEND_MODES)
    
    watermarked = blend_images(image, watermark_layer, blend_mode)
    
    # 添加隐形防护层
    watermarked = add_invisible_watermark_layer(watermarked, WatermarkConfig.DEFAULT_TEXT, WatermarkConfig.DEFAULT_FONT_PATH, WatermarkConfig.DEFAULT_FONT_SIZE)
    
    # 添加左上角标识
    temp_path1 = output_image_path.replace(".png", "_temp1.png")
    watermarked.convert("RGB").save(temp_path1, "PNG")
    
    # 在水印图上添加左上角标识
    temp_path2 = output_image_path.replace(".png", "_temp2.png")
    add_corner_label(temp_path1, temp_path2)
    
    # 添加右下角品牌logo
    temp_path_final = output_image_path.replace(".png", "_temp_final.png")
    add_brand_logo(temp_path2, temp_path_final)
    
    # 缩放图片到指定比例
    if resize_scale != 1.0:
        final_image = Image.open(temp_path_final)
        original_width, original_height = final_image.size
        new_width = int(original_width * resize_scale)
        new_height = int(original_height * resize_scale)
        
        # 使用高质量的重采样算法
        resized_image = final_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # 保存最终结果
        if output_image_path.lower().endswith((".jpg", ".jpeg")):
            resized_image = resized_image.convert("RGB")
        
        resized_image.save(output_image_path)
        print(f"图片已缩放至 {resize_scale*100}% ({new_width}x{new_height})")
    else:
        # 如果不需要缩放，直接重命名
        import shutil
        shutil.move(temp_path_final, output_image_path)
    
    # 删除临时文件
    if os.path.exists(temp_path1):
        os.remove(temp_path1)
    if os.path.exists(temp_path2):
        os.remove(temp_path2)
    if os.path.exists(temp_path_final):
        os.remove(temp_path_final)
    
    print(f"水印叠加完成: {output_image_path}") 

def add_brand_logo(
    input_image_path, 
    output_image_path, 
    logo_path=None,
    margin=None,
    scale=None,
    min_size=None,
    max_size=None,
    opacity=None
):
    """在图片右下角添加品牌logo（优化版本，包含缓存）"""
    # 使用配置文件的默认值
    logo_path = logo_path or WatermarkConfig.BRAND_LOGO_PATH
    margin = margin or WatermarkConfig.BRAND_LOGO_MARGIN
    scale = scale or WatermarkConfig.BRAND_LOGO_SCALE
    min_size = min_size or WatermarkConfig.BRAND_LOGO_MIN_SIZE
    max_size = max_size or WatermarkConfig.BRAND_LOGO_MAX_SIZE
    opacity = opacity or WatermarkConfig.BRAND_LOGO_OPACITY
    
    # 检查logo文件是否存在
    if not os.path.exists(logo_path):
        print(f"品牌logo文件不存在: {logo_path}")
        # 如果logo不存在，直接复制原图
        if input_image_path != output_image_path:
            import shutil
            shutil.copy2(input_image_path, output_image_path)
        return
    
    # 打开原图
    image = Image.open(input_image_path).convert("RGBA")
    img_width, img_height = image.size
    
    # 计算logo目标尺寸（基于图片宽度的比例）
    target_size = int(img_width * scale)
    target_size = max(min_size, min(max_size, target_size))  # 限制在最小和最大尺寸之间
    
    # 生成缓存键
    cache_key = f"{logo_path}_{target_size}_{opacity}"
    
    # 检查缓存中是否有已处理的logo
    if cache_key in _logo_cache:
        logo_resized = _logo_cache[cache_key]
        new_width, new_height = logo_resized.size
    else:
        # 加载并处理logo（只在缓存中没有时执行）
        logo = Image.open(logo_path).convert("RGBA")
        logo_width, logo_height = logo.size
        
        # 计算logo缩放比例（保持原始比例）
        logo_ratio = logo_width / logo_height
        if logo_ratio > 1:  # 宽度大于高度
            new_width = target_size
            new_height = int(target_size / logo_ratio)
        else:  # 高度大于等于宽度
            new_height = target_size
            new_width = int(target_size * logo_ratio)
        
        # 缩放logo
        logo_resized = logo.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # 调整透明度
        if opacity < 1.0:
            # 创建一个新的图层来调整透明度
            alpha = logo_resized.split()[-1]  # 获取alpha通道
            alpha = alpha.point(lambda p: int(p * opacity))  # 调整透明度
            logo_resized.putalpha(alpha)
        
        # 缓存处理后的logo（限制缓存大小）
        if len(_logo_cache) < 10:  # 最多缓存10个不同尺寸的logo
            _logo_cache[cache_key] = logo_resized.copy()
    
    # 计算logo在右下角的位置
    logo_x = img_width - new_width - margin
    logo_y = img_height - new_height - margin
    
    # 创建一个新的图层来放置logo
    watermark_layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    watermark_layer.paste(logo_resized, (logo_x, logo_y), logo_resized)
    
    # 合成图片
    result = Image.alpha_composite(image, watermark_layer)
    
    # 保存图片
    if output_image_path.lower().endswith((".jpg", ".jpeg")):
        result = result.convert("RGB")
    
    result.save(output_image_path)
    print(f"品牌logo添加完成: {output_image_path}")

def clear_logo_cache():
    """清理logo缓存"""
    global _logo_cache
    _logo_cache.clear()
    print("Logo缓存已清理")

def add_comprehensive_watermark(
    input_image_path,
    output_image_path,
    include_text_watermark=True,
    include_corner_label=True,
    include_brand_logo=True,
    **kwargs
):
    """
    综合水印处理函数
    
    Args:
        input_image_path: 输入图片路径
        output_image_path: 输出图片路径
        include_text_watermark: 是否包含文字水印
        include_corner_label: 是否包含左上角标识
        include_brand_logo: 是否包含右下角品牌logo
        **kwargs: 其他参数传递给各个水印函数
    """
    current_path = input_image_path
    temp_files = []
    
    try:
        # 1. 添加文字水印
        if include_text_watermark:
            temp_path = output_image_path.replace(".png", "_step1.png").replace(".jpg", "_step1.jpg")
            temp_files.append(temp_path)
            
            # 提取文字水印相关参数
            text_kwargs = {k: v for k, v in kwargs.items() 
                          if k in ['text', 'font_path', 'font_size', 'color', 'angle', 'step', 'alpha']}
            text_kwargs['add_brand'] = False  # 在综合处理中不单独添加品牌logo
            add_text_watermark(current_path, temp_path, **text_kwargs)
            current_path = temp_path
        
        # 2. 添加左上角标识
        if include_corner_label:
            temp_path = output_image_path.replace(".png", "_step2.png").replace(".jpg", "_step2.jpg")
            temp_files.append(temp_path)
            
            # 提取左上角标识相关参数
            corner_kwargs = {k: v for k, v in kwargs.items() 
                           if k in ['text', 'font_path', 'font_size', 'padding', 'margin', 
                                   'text_color', 'bg_color', 'border_color', 'border_width', 'corner_radius']}
            # 注意：这里需要临时禁用add_corner_label内部的品牌logo添加
            temp_add_corner_label(current_path, temp_path, **corner_kwargs)
            current_path = temp_path
        
        # 3. 添加右下角品牌logo
        if include_brand_logo:
            # 提取品牌logo相关参数
            brand_kwargs = {k: v for k, v in kwargs.items() 
                          if k in ['logo_path', 'margin', 'scale', 'min_size', 'max_size', 'opacity']}
            add_brand_logo(current_path, output_image_path, **brand_kwargs)
        else:
            # 如果不需要品牌logo，直接复制当前结果
            if current_path != output_image_path:
                import shutil
                shutil.copy2(current_path, output_image_path)
    
    finally:
        # 清理临时文件
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)
    
    print(f"综合水印处理完成: {output_image_path}")

def temp_add_corner_label(
    input_image_path, 
    output_image_path, 
    text=None,
    font_path=None,
    font_size=None,
    padding=None,
    margin=None,
    text_color=None,
    bg_color=None,
    border_color=None,
    border_width=None,
    corner_radius=None
):
    """临时的左上角标识函数，不包含品牌logo处理"""
    # 使用配置文件的默认值
    text = text or WatermarkConfig.CORNER_LABEL_TEXT
    font_path = font_path or WatermarkConfig.CORNER_LABEL_FONT_PATH
    font_size = font_size or WatermarkConfig.CORNER_LABEL_FONT_SIZE
    padding = padding or WatermarkConfig.CORNER_LABEL_PADDING
    margin = margin or WatermarkConfig.CORNER_LABEL_MARGIN
    text_color = text_color or WatermarkConfig.CORNER_LABEL_TEXT_COLOR
    bg_color = bg_color or WatermarkConfig.CORNER_LABEL_BG_COLOR
    border_color = border_color or WatermarkConfig.CORNER_LABEL_BORDER_COLOR
    border_width = border_width or WatermarkConfig.CORNER_LABEL_BORDER_WIDTH
    corner_radius = corner_radius or WatermarkConfig.CORNER_LABEL_CORNER_RADIUS
    
    # 打开图片
    image = Image.open(input_image_path).convert("RGBA")
    width, height = image.size
    
    # 创建绘图对象
    draw = ImageDraw.Draw(image)
    
    # 加载字体
    try:
        font = ImageFont.truetype(font_path, font_size)
    except Exception as e:
        font = ImageFont.load_default()
        print(f"使用默认字体，字号不可调，错误信息：{e}")
    
    # 计算文字尺寸
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    # 计算外框尺寸和位置（左上角）
    box_width = text_width + 2 * padding
    box_height = text_height + 2 * padding
    box_x = margin
    box_y = margin
    
    # 绘制圆角矩形（背景和边框）
    box_coords = [box_x, box_y, box_x + box_width, box_y + box_height]
    draw_rounded_rectangle(
        draw, 
        box_coords, 
        corner_radius, 
        fill=bg_color, 
        outline=border_color, 
        width=border_width
    )
    
    # 计算文字位置（在框内居中）
    text_x = box_x + padding
    text_y = box_y + padding
    
    # 绘制文字
    draw.text((text_x, text_y), text, font=font, fill=text_color)
    
    # 保存图片
    if output_image_path.lower().endswith(".jpg") or output_image_path.lower().endswith(".jpeg"):
        image = image.convert("RGB")
    
    image.save(output_image_path) 