import cv2
import numpy as np
import os
from pathlib import Path

def quantize_image(image, levels):
    """
    对单张图像进行量化处理
    :param image: 输入图像
    :param levels: 量化级别
    :return: 量化后的图像
    """
    step = 256 // levels
    quantized = (image // step) * step
    return np.clip(quantized, 0, 255).astype(np.uint8)

def process_folder(input_dir, output_dir, levels_list):
    """
    批量处理文件夹中的所有图像
    :param input_dir: 输入图像文件夹
    :param output_dir: 输出结果文件夹
    :param levels_list: 量化级别列表
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 支持的图像格式
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    
    # 获取所有图像文件路径
    image_files = [f for f in os.listdir(input_dir) 
                  if f.lower().endswith(image_extensions)]
    
    if not image_files:
        print(f"在 {input_dir} 中未找到任何图像文件")
        return
    
    print(f"找到 {len(image_files)} 张图像，开始量化处理...")
    
    # 处理每张图像
    for img_file in image_files:
        img_path = os.path.join(input_dir, img_file)
        image = cv2.imread(img_path)
        
        if image is None:
            print(f"警告：无法读取图像 {img_file}，已跳过")
            continue
        
        # 获取图像文件名（不含扩展名）
        img_name = Path(img_file).stem
        
        # 对当前图像应用所有量化级别
        for levels in levels_list:
            quantized_img = quantize_image(image, levels)
            
            # 构建输出文件名
            output_filename = f"{img_name}_quantized_L{levels}.png"
            output_path = os.path.join(output_dir, output_filename)
            
            # 保存量化后的图像
            cv2.imwrite(output_path, quantized_img)
        
        print(f"已处理：{img_file}")
    
    print(f"所有图像处理完成，结果保存在 {output_dir}")

if __name__ == "__main__":
    # 配置参数
    input_directory = "../813"  # 输入图像文件夹路径
    output_directory = "quantization_attack"  # 输出结果文件夹路径
    quantization_levels = [128, 64, 32, 16, 8, 4, 2]  # 量化级别
    
    # 检查输入目录是否存在
    if not os.path.exists(input_directory):
        print(f"错误：输入目录 {input_directory} 不存在")
    else:
        process_folder(input_directory, output_directory, quantization_levels)
    