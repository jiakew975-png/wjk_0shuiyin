import os
import cv2
import numpy as np
from tqdm import tqdm  # 用于显示进度条


def add_salt_pepper_noise(image, noise_level):
    """
    为图像添加椒盐噪声
    :param image: 原始图像
    :param noise_level: 噪声强度（占总像素的百分比）
    :return: 添加噪声后的图像
    """
    h, w = image.shape[:2]
    # 计算噪声像素数量（避免噪声比例为0时的无效计算）
    if noise_level > 0:
        num_noise = int(h * w * noise_level / 100)

        # 添加胡椒噪声（黑色像素）
        coords = [np.random.randint(0, i - 1, num_noise) for i in [h, w]]
        image[coords[0], coords[1]] = 0

        # 添加盐噪声（白色像素）
        coords = [np.random.randint(0, i - 1, num_noise) for i in [h, w]]
        image[coords[0], coords[1]] = 255

    return image


def process_images(input_dir, output_dir):
    """
    批量为图像添加不同强度的椒盐噪声，按图片名创建子文件夹存储
    :param input_dir: 输入图像目录
    :param output_dir: 根输出目录（子文件夹会在此目录下创建）
    """
    # 创建根输出目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)

    # 有效图像扩展名
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')

    # 获取所有图像文件（过滤非图片文件）
    image_files = [f for f in os.listdir(input_dir)
                   if f.lower().endswith(valid_exts) and os.path.isfile(os.path.join(input_dir, f))]

    if not image_files:
        print(f"在 {input_dir} 中未找到任何图像文件")
        return

    # 噪声强度范围：0%到50%，每隔0.01%生成一个强度（保留两位小数）
    start_level = 0.00
    end_level = 50.00
    step = 0.1
    # 使用np.arange生成后保留两位小数，避免浮点数精度误差（如0.029999999999999）
    noise_levels = np.round(np.arange(start_level, end_level + step, step), 2).tolist()

    print(f"=== 预处理配置 ===")
    print(f"找到 {len(image_files)} 张图像")
    print(f"噪声强度范围: {start_level:.2f}% ~ {end_level:.2f}%")
    print(f"强度间隔: {step:.2f}%")
    print(f"总噪声等级数: {len(noise_levels)}")
    print(f"根输出目录: {os.path.abspath(output_dir)}")
    print(f"=================\n")

    # 处理每张图像（显示全局进度条）
    for img_file in tqdm(image_files, desc="全局处理进度", unit="张图片"):
        # 1. 读取原始图像
        img_path = os.path.join(input_dir, img_file)
        img = cv2.imread(img_path)
        if img is None:
            print(f"\n警告：无法读取图像 {img_file}，已跳过")
            continue

        # 2. 提取图片名（不含扩展名），创建该图片的独立子文件夹
        base_name, ext = os.path.splitext(img_file)  # base_name=图片名, ext=扩展名（如.jpg）
        img_subdir = os.path.join(output_dir, base_name)  # 子文件夹路径：根输出目录/图片名
        os.makedirs(img_subdir, exist_ok=True)  # 确保子文件夹存在

        # 3. 为当前图片添加所有强度的噪声（显示单张图片进度条）
        for level in tqdm(noise_levels, desc=f"处理 {base_name}", unit="个噪声等级", leave=False):
            # 复制原图，避免多次处理污染原始数据
            noisy_img = img.copy()
            # 添加椒盐噪声
            noisy_img = add_salt_pepper_noise(noisy_img, level)

            # 4. 构建输出文件名（保留两位小数，格式：图片名_jnoise_强度%_扩展名）
            # 处理强度字符串（如2.05%而非2.050000000001%）
            level_str = f"{level:.2f}"  # 强制保留两位小数，如0.01、5.20、50.00
            output_name = f"{base_name}_jynoise_{level_str}%{ext}"
            # 输出路径：子文件夹/输出文件名
            output_path = os.path.join(img_subdir, output_name)

            # 5. 保存噪声图像
            cv2.imwrite(output_path, noisy_img)

        # 单张图片处理完成提示
        tqdm.write(f"图片 {img_file} 处理完成，共生成 {len(noise_levels)} 个噪声样本，保存至：{os.path.abspath(img_subdir)}")


if __name__ == "__main__":
    # 输入输出目录设置（可根据实际路径修改）
    input_directory = "../813"        # 输入图像所在目录
    output_directory = "jyzs"  # 根输出目录（子文件夹会在此目录下创建）

    # 检查输入目录是否存在
    if not os.path.exists(input_directory):
        print(f"错误：输入目录不存在 - {input_directory}")
    else:
        # 执行图像处理
        process_images(input_directory, output_directory)
        print(f"\n=== 所有图像处理完成 ===")
        print(f"最终结果保存至根目录：{os.path.abspath(output_directory)}")
        print(f"目录结构：{output_directory}/[图片名]/[图片名_jnoise_xx.xx%.ext]")
# import os
# import cv2
# import numpy as np
# import random
# from tqdm import tqdm  # 用于显示进度条
#
# def add_salt_pepper_noise(image, noise_level):
#     """
#     为图像添加椒盐噪声
#     :param image: 原始图像
#     :param noise_level: 噪声强度（占总像素的百分比）
#     :return: 添加噪声后的图像
#     """
#     h, w = image.shape[:2]
#     # 计算噪声像素数量
#     num_noise = int(h * w * noise_level / 100)
#
#     # 添加胡椒噪声（黑色像素）
#     coords = [np.random.randint(0, i - 1, num_noise) for i in [h, w]]
#     image[coords[0], coords[1]] = 0
#
#     # 添加盐噪声（白色像素）
#     coords = [np.random.randint(0, i - 1, num_noise) for i in [h, w]]
#     image[coords[0], coords[1]] = 255
#
#     return image
#
# def generate_random_noise_levels(num_samples, min_level=0.0, max_level=30.0, low_ratio=0.5):
#     """
#     生成随机噪声强度
#     :param num_samples: 样本数量
#     :param min_level: 最小噪声强度
#     :param max_level: 最大噪声强度
#     :param low_ratio: 低强度噪声(0-10%)的占比
#     :return: 随机噪声强度列表（保留两位小数）
#     """
#     # 计算低强度和高强度噪声的样本数量
#     low_count = int(num_samples * low_ratio)
#     high_count = num_samples - low_count
#
#     # 生成低强度噪声(0-10%)
#     low_levels = [round(random.uniform(min_level, 10.0), 2) for _ in range(low_count)]
#     # 生成高强度噪声(10-30%)
#     high_levels = [round(random.uniform(10.0, max_level), 2) for _ in range(high_count)]
#
#     # 合并并打乱顺序
#     all_levels = low_levels + high_levels
#     random.shuffle(all_levels)
#
#     return all_levels
#
# def process_images(input_dir, output_dir, num_samples=20):
#     """
#     批量为图像添加随机强度的椒盐噪声
#     :param input_dir: 输入图像目录
#     :param output_dir: 输出图像目录
#     :param num_samples: 每张图像生成的随机噪声样本数量
#     """
#     # 创建输出目录（如果不存在）
#     os.makedirs(output_dir, exist_ok=True)
#
#     # 有效图像扩展名
#     valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')
#
#     # 获取所有图像文件
#     image_files = [f for f in os.listdir(input_dir)
#                    if f.lower().endswith(valid_exts)]
#
#     if not image_files:
#         print(f"在 {input_dir} 中未找到任何图像文件")
#         return
#
#     print(f"找到 {len(image_files)} 张图像...")
#     print(f"将为每张图像生成 {num_samples} 个随机噪声样本（0.0%-30.0%）")
#
#     # 处理每张图像
#     for img_file in tqdm(image_files, desc="处理进度"):
#         img_path = os.path.join(input_dir, img_file)
#         img = cv2.imread(img_path)
#
#         if img is None:
#             print(f"无法读取图像: {img_file}")
#             continue
#
#         # 生成随机噪声强度（0-30%，保留两位小数）
#         noise_levels = generate_random_noise_levels(num_samples)
#
#         # 为每张图像添加不同强度的噪声
#         for level in noise_levels:
#             noisy_img = img.copy()
#             noisy_img = add_salt_pepper_noise(noisy_img, level)
#
#             # 构建输出文件名
#             base_name, ext = os.path.splitext(img_file)
#             output_name = f"{base_name}_noise_{level}%{ext}"
#             output_path = os.path.join(output_dir, output_name)
#
#             cv2.imwrite(output_path, noisy_img)
#
# if __name__ == "__main__":
#     # 输入输出目录设置
#     input_directory = "../813"  # 输入图像所在目录
#     output_directory = "jnoise_test"     # 输出图像保存目录
#     num_samples_per_image = 130  # 每张图像生成的随机样本数量
#
#     # 检查输入目录是否存在
#     if not os.path.exists(input_directory):
#         print(f"输入目录不存在: {input_directory}")
#     else:
#         process_images(input_directory, output_directory, num_samples_per_image)
#         print(f"\n所有图像已处理完成，保存至 {output_directory}")
