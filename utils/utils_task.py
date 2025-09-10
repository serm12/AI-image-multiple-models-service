import os
import uuid
import json
from datetime import datetime

def generate_task_dir(tasks_dir="tasks"):
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    task_id = f"{timestamp}_{uuid.uuid4().hex[:8]}"
    task_dir = os.path.join(tasks_dir, task_id)
    os.makedirs(task_dir, exist_ok=True)
    return task_id, task_dir, timestamp

def save_input_image(input_path, task_dir):
    input_image_filename = os.path.basename(input_path)
    task_input_image_path = os.path.join(task_dir, input_image_filename)
    with open(input_path, "rb") as src, open(task_input_image_path, "wb") as dst:
        dst.write(src.read())
    return input_image_filename, task_input_image_path

def save_params(params, task_dir):
    with open(os.path.join(task_dir, "params.json"), "w", encoding="utf-8") as f:
        json.dump(params, f, ensure_ascii=False, indent=2)

def generate_output_filenames(task_dir, input_image_filename, output_format):
    base_name = os.path.splitext(input_image_filename)[0]
    
    # 从配置中获取算法值
    from config import AlgorithmConfig
    algorithm_value = AlgorithmConfig.get_algorithm_value()
    
    # 生成原图文件名：output_cropped_original_算法值.png
    original_file = os.path.join(task_dir, f"output_cropped_original_{algorithm_value}.{output_format}")
    
    # 生成水印文件名（保持原来的命名方式，不加算法值）
    watermark_file = os.path.join(task_dir, f"output_cropped_watermark.{output_format}")
    
    # 不生成主输出文件（因为它和原图一样），直接返回原图文件作为主输出
    output_file = original_file
    
    return output_file, original_file, watermark_file 