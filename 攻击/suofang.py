import os
from PIL import Image
import numpy as np

def scale_image(image, scale_factor):
    """仅缩放图像，不保持原始尺寸，不填充背景"""
    original_width, original_height = image.size
    
    # 计算缩放后的图像尺寸，确保至少为1x1
    scaled_width = max(1, int(original_width * scale_factor))
    scaled_height = max(1, int(original_height * scale_factor))
    
    # 缩放图像内容
    if scale_factor <= 0:
        # 处理缩放因子为0或负数的情况，返回1x1的图像
        scaled_img = Image.new(image.mode, (1, 1), color=image.getpixel((0, 0)) if original_width > 0 and original_height > 0 else 0)
    else:
        scaled_img = image.resize((scaled_width, scaled_height), Image.Resampling.LANCZOS)
    
    return scaled_img

def generate_scale_series(image, num_steps=10, min_scale=0.0, max_scale=2.0):
    """生成从min_scale到max_scale的缩放序列，返回缩放后的图像（不保持原尺寸）"""
    scales = np.linspace(min_scale, max_scale, num_steps, endpoint=True)
    scaled_images = []
    
    for scale in scales:
        scaled_img = scale_image(image, scale)
        scaled_images.append(scaled_img)
    
    return scales, scaled_images

def batch_generate_scale_series(input_folder, output_folder, num_steps=10, min_scale=0.0, max_scale=2.0):
    """批量处理图像，生成缩放序列（不填充背景，不保持原尺寸）"""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"创建输出主文件夹: {output_folder}")
    
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp')
    
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(image_extensions):
            input_path = os.path.join(input_folder, filename)
            base_name = os.path.splitext(filename)[0]
            image_output_folder = os.path.join(output_folder, base_name)
            os.makedirs(image_output_folder, exist_ok=True)
            
            try:
                with Image.open(input_path) as img:                    
                    scales, scaled_images = generate_scale_series(
                        img, num_steps, min_scale, max_scale
                    )
                    
                    for i, (scale, scaled_img) in enumerate(zip(scales, scaled_images)):
                        # 处理缩放比例的字符串表示，确保文件名合法
                        scale_str = f"{scale:.2f}".replace('.', '.')  # 使用p代替.避免潜在问题
                        ext = os.path.splitext(filename)[1]
                        output_filename = f"{base_name}_scale_{scale_str}{ext}"
                        output_path = os.path.join(image_output_folder, output_filename)
                        scaled_img.save(output_path)
                    
                    print(f"已生成 {num_steps} 张缩放图像: {filename}")
                    
            except Exception as e:
                print(f"处理 {filename} 时出错: {str(e)}")
    
    print("所有图像的缩放序列生成完成!")

if __name__ == "__main__":
    input_folder = "../813"       # 输入图像文件夹
    output_folder = "缩放490"  # 输出文件夹
    num_steps = 490                # 生成的步数
    min_scale = 0.1               # 最小缩放比例
    max_scale = 5                 # 最大缩放比例
    
    batch_generate_scale_series(
        input_folder=input_folder,
        output_folder=output_folder,
        num_steps=num_steps,
        min_scale=min_scale,
        max_scale=max_scale
    )

# import cv2
# import numpy as np
# import os
# import random
#
# def generate_random_scales(num_samples, lower=0.5, upper=3.0, low_range_ratio=0.3):
#     """
#     生成随机缩放因子，指定低范围区间的占比
#     :param num_samples: 生成的样本数量
#     :param lower: 总范围下限
#     :param upper: 总范围上限
#     :param low_range_ratio: 低范围区间(0.5-1.0)的样本占比
#     :return: 随机缩放因子列表（保留两位小数）
#     """
#     # 计算低范围和高范围的样本数量
#     low_count = int(num_samples * low_range_ratio)
#     high_count = num_samples - low_count
#
#     # 生成低范围(0.5-1.0)的随机因子
#     low_scales = [round(random.uniform(0.5, 1.0), 2) for _ in range(low_count)]
#
#     # 生成高范围(1.0-3.0)的随机因子
#     high_scales = [round(random.uniform(1.0, 3.0), 2) for _ in range(high_count)]
#
#     # 合并并打乱顺序
#     all_scales = low_scales + high_scales
#     random.shuffle(all_scales)
#
#     return all_scales
#
# # 输入图像路径（替换为你的图像路径）
# image_path = 'Peppers.png'  # 可为 PNG/JPG 等格式
# image = cv2.imread(image_path)
#
# if image is None:
#     raise ValueError(f"无法读取图像: {image_path}")
#
# # 生成随机缩放因子（例如生成30个样本）
# num_samples = 60  # 可根据需要调整样本数量
# random_scales = generate_random_scales(num_samples)
#
# # 输出目录
# output_dir = 'random_scaled_images'
# os.makedirs(output_dir, exist_ok=True)
#
# # 对每个随机缩放因子执行缩放操作
# for scale in random_scales:
#     # 按缩放因子缩放
#     scaled = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
#
#     # 恢复为原始图像尺寸
#     scaled_resized = cv2.resize(scaled, (image.shape[1], image.shape[0]))
#
#     # 保存图像（文件名保留两位小数）
#     filename = f'scaled_{scale:.2f}.png'
#     output_path = os.path.join(output_dir, filename)
#     cv2.imwrite(output_path, scaled_resized)
#     print(f"Saved: {output_path}")
#
# print(f"\n生成完成，共 {num_samples} 个样本，其中0.5-1.0范围占比约40%")
