import cv2
import numpy as np
import os
import random
from tqdm import tqdm

def usm_sharpen(image, amount=1.0, radius=1.0, threshold=0):
    """USM锐化实现，固定半径和阈值，调整强度"""
    # 确保参数有效范围
    amount = max(0.1, min(5.0, amount))  # 限制强度在0.1-5.0
    radius = max(0.5, min(2.0, radius))   # 固定半径范围（0.5-2.0）
    threshold = max(0, min(20, threshold)) # 固定阈值范围（0-20）
    
    # 生成高斯模糊掩模
    ksize = int(2 * round(radius) + 1)  # 计算核大小（确保为奇数）
    blurred = cv2.GaussianBlur(image, (ksize, ksize), sigmaX=radius)
    
    # 提取高频细节并应用阈值
    high_freq = image.astype(np.float32) - blurred.astype(np.float32)
    if threshold > 0:
        high_freq_abs = np.abs(high_freq)
        high_freq[high_freq_abs < threshold] = 0  # 过滤低对比度细节
    
    # 应用锐化强度并裁剪像素值范围
    sharpened = image.astype(np.float32) + amount * high_freq
    return np.clip(sharpened, 0, 255).astype(np.uint8)

def generate_leveled_amounts(num_weak, num_medium, num_strong):
    """
    分区间生成锐化强度（保留两位小数）
    - 弱锐化：0.10 ~ 0.50
    - 中锐化：0.60 ~ 2.00
    - 强锐化：2.10 ~ 5.00
    """
    amounts = []
    
    # 生成弱锐化强度（确保数量和唯一性）
    while len([a for a in amounts if 0.10 <= a <= 0.50]) < num_weak:
        a = round(random.uniform(0.10, 0.50), 2)
        if a not in amounts:
            amounts.append(a)
    
    # 生成中锐化强度
    while len([a for a in amounts if 0.60 <= a <= 2.00]) < num_medium:
        a = round(random.uniform(0.60, 2.00), 2)
        if a not in amounts:
            amounts.append(a)
    
    # 生成强锐化强度
    while len([a for a in amounts if 2.10 <= a <= 5.00]) < num_strong:
        a = round(random.uniform(2.10, 5.00), 2)
        if a not in amounts:
            amounts.append(a)
    
    return amounts

def process_images_with_leveled_amounts(input_dir, output_dir, total_samples=51):
    """批量处理图像，按强中弱区间生成锐化样本"""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有有效图像文件
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    image_files = [
        f for f in os.listdir(input_dir) 
        if f.lower().endswith(valid_exts)
    ]
    
    if not image_files:
        print(f"错误：在 {input_dir} 中未找到任何图像文件")
        return
    
    # 分配各区间样本数量（按比例）
    num_weak = int(total_samples * 0.2)    # 弱锐化：30%
    num_medium = int(total_samples * 0.5)  # 中锐化：50%（重点）
    num_strong = total_samples - num_weak - num_medium  # 强锐化：剩余20%
    
    # 打印参数信息
    print(f"===== 锐化样本生成配置 =====")
    print(f"输入目录：{input_dir}")
    print(f"输出目录：{output_dir}")
    print(f"图像数量：{len(image_files)} 张")
    print(f"单图样本数：{total_samples} 个（弱：{num_weak} | 中：{num_medium} | 强：{num_strong}）")
    print(f"强度区间（保留两位小数）：")
    print(f"  弱锐化：0.10 ~ 0.50")
    print(f"  中锐化：0.60 ~ 2.00")
    print(f"  强锐化：2.10 ~ 5.00")
    print("===========================")
    
    # 固定半径和阈值（可根据需求调整）
    fixed_radius = 1.0    # 高斯模糊半径
    fixed_threshold = 5   # 锐化阈值
    
    # 批量处理图像
    for img_file in tqdm(image_files, desc="处理进度"):
        img_path = os.path.join(input_dir, img_file)
        image = cv2.imread(img_path)
        
        if image is None:
            print(f"\n警告：无法读取图像 {img_file}，已跳过")
            continue
        
        # 生成分区间的锐化强度
        random_amounts = generate_leveled_amounts(num_weak, num_medium, num_strong)
        
        # 生成并保存锐化样本
        base_name, ext = os.path.splitext(img_file)
        for amount in random_amounts:
            # 执行锐化
            sharpened_img = usm_sharpen(image, amount, fixed_radius, fixed_threshold)
            
            # 确定区间标识（弱W/中M/强S）
            if 0.10 <= amount <= 0.50:
                level_tag = 'W'
            elif 0.60 <= amount <= 2.00:
                level_tag = 'M'
            else:
                level_tag = 'S'
            
            # 构建输出文件名
            output_name = f"{base_name}_usm_a{amount:.2f}{ext}"
            output_path = os.path.join(output_dir, output_name)
            
            # 保存图像
            cv2.imwrite(output_path, sharpened_img)
    
    print(f"\n处理完成！所有样本已保存至 {output_dir}")

if __name__ == "__main__":
    # 配置参数（可根据实际需求修改）
    input_directory = "../813"         # 输入图像所在目录
    output_directory = "ruihua_testsamples"  # 输出样本保存目录
    num_samples_per_image = 100         # 每张图像生成的样本总数
    
    # 检查输入目录是否存在
    if not os.path.exists(input_directory):
        print(f"错误：输入目录不存在 - {input_directory}")
    else:
        process_images_with_leveled_amounts(
            input_dir=input_directory,
            output_dir=output_directory,
            total_samples=num_samples_per_image
        )
