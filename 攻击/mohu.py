import os
import cv2
import numpy as np
from tqdm import tqdm


def apply_blur(image, blur_type, kernel_size):
    """
    应用模糊效果
    :param image: 输入图像
    :param blur_type: 模糊类型 ('gaussian' 高斯模糊, 'median' 中值模糊, 'average' 平均模糊)
    :param kernel_size: 模糊核大小 (必须是奇数，值越大模糊越强)
    :return: 模糊处理后的图像
    """
    # 确保核大小为奇数且不小于3
    kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    kernel_size = max(3, kernel_size)  # 最小核大小为3x3

    if blur_type == 'gaussian':
        # 高斯模糊：模拟光学失焦
        blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    elif blur_type == 'median':
        # 中值模糊：同时模拟模糊和降噪
        blurred = cv2.medianBlur(image, kernel_size)
    elif blur_type == 'average':
        # 平均模糊：模拟简单平滑
        kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
        blurred = cv2.filter2D(image, -1, kernel)
    else:
        raise ValueError(f"不支持的模糊类型: {blur_type}")

    return blurred


def process_images(input_dir, output_dir, img_size=(128, 128)):
    """
    处理文件夹中的所有图像，仅生成固定强度的模糊样本
    :param input_dir: 输入文件夹路径
    :param output_dir: 输出文件夹路径
    :param img_size: 图像尺寸，固定为128x128
    """
    os.makedirs(output_dir, exist_ok=True)

    # 支持的图像格式
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    image_files = [f for f in os.listdir(input_dir)
                   if f.lower().endswith(valid_exts)]

    if not image_files:
        print(f"在文件夹 {input_dir} 中没有找到图像文件")
        return

    # 仅使用固定强度的模糊参数（取消随机部分）
    blur_types = ['gaussian', 'median', 'average']  # 三种模糊类型
    max_kernel = 15  # 最大核大小（15x15）
    min_kernel = 3   # 最小核大小（3x3）
    step_size = 2    # 核大小步长（只取奇数）
    
    # 生成固定强度级别（3x3, 5x5, ..., 15x15）
    fixed_kernels = list(range(min_kernel, max_kernel + 1, step_size))

    print(f"图像尺寸: {img_size[0]}x{img_size[1]}")
    print(f"模糊核大小范围: {min_kernel}x{min_kernel} 到 {max_kernel}x{max_kernel}")
    print(f"固定步长: {step_size}，共 {len(fixed_kernels)} 个固定级别")
    print(f"模糊类型: {blur_types}")

    # 处理每张图像
    for img_file in tqdm(image_files, desc="处理进度"):
        img_path = os.path.join(input_dir, img_file)
        img = cv2.imread(img_path)

        if img is None:
            print(f"无法读取图像: {img_path}")
            continue

        # 确保图像尺寸为128x128
        if img.shape[:2] != img_size:
            img = cv2.resize(img, img_size, interpolation=cv2.INTER_AREA)
            print(f"已将 {img_file} 调整为 {img_size[0]}x{img_size[1]}")

        base_name, ext = os.path.splitext(img_file)

        # 仅生成固定强度的模糊样本（取消随机生成部分）
        for kernel_size in fixed_kernels:
            for blur_type in blur_types:
                blurred = apply_blur(img, blur_type, kernel_size)
                
                # 构造输出文件名，包含原始文件名、模糊类型和核大小
                output_name = f"{base_name}_blur_{blur_type}_{kernel_size}x{kernel_size}{ext}"
                output_path = os.path.join(output_dir, output_name)
                cv2.imwrite(output_path, blurred)


if __name__ == "__main__":
    input_directory = "../813"  # 输入文件夹路径
    output_directory = "blur"  # 输出文件夹路径
    target_size = (128, 128)  # 固定图像尺寸

    if not os.path.exists(input_directory):
        print(f"输入文件夹不存在: {input_directory}")
    else:
        process_images(input_directory, output_directory, target_size)
        print(f"\n模糊样本生成完成，结果保存在 {output_directory}")