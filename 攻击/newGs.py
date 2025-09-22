import os
import cv2
import numpy as np
from tqdm import tqdm


def add_gaussian_noise(image, sigma):
    """
    向图像添加高斯噪声
    :param image: 输入图像
    :param sigma: 噪声标准差
    :return: 带噪声的图像
    """
    row, col, ch = image.shape
    # 生成高斯噪声（均值0，标准差sigma）
    gauss = np.random.normal(0, sigma, (row, col, ch))
    noisy = image + gauss
    # 裁剪到0-255范围并转换为uint8类型
    return np.clip(noisy, 0, 255).astype('uint8')


def process_image_with_noise_levels(image_path, output_folder, noise_levels):
    """
    为单张图片生成不同噪声强度的版本（无可视化）
    :param image_path: 图片路径
    :param output_folder: 输出文件夹
    :param noise_levels: 噪声强度列表
    """
    try:
        # 读取图片（BGR格式）
        img = cv2.imread(image_path)
        if img is None:
            print(f"无法读取图片: {image_path}")
            return

        # 为每个噪声强度生成并保存噪声图片
        for sigma in noise_levels:
            # 添加噪声（直接处理BGR格式，避免颜色空间转换损耗）
            noisy_img = add_gaussian_noise(img, sigma=sigma)

            # 构建输出文件名
            img_name = os.path.basename(image_path)
            name_part, ext = os.path.splitext(img_name)
            output_path = os.path.join(output_folder, f"{name_part}_gnoise_{sigma}{ext}")

            # 保存带噪声的图片
            cv2.imwrite(output_path, noisy_img)

    except Exception as e:
        print(f"处理图片 {image_path} 时出错: {str(e)}")


def process_folder(input_folder, output_folder):
    """
    处理文件夹中的所有图片（无可视化）
    :param input_folder: 输入文件夹路径
    :param output_folder: 输出文件夹路径
    """
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 获取文件夹中的所有图片文件
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    image_files = [f for f in os.listdir(input_folder)
                   if f.lower().endswith(valid_extensions)]

    if not image_files:
        print(f"在文件夹 {input_folder} 中没有找到图片文件")
        return

    # 生成0-80范围内的6个噪声级别（间隔16）
    # 计算方式：(80-0)/(6-1) = 16，因此级别为0,
    # 16,32,48,64,80
    noise_levels = [i  for i in range(80)]  # [0, 16, 32, 48, 64, 80]

    print(f"开始处理 {len(image_files)} 张图片...")
    print(f"噪声级别: σ={noise_levels[0]} 到 σ={noise_levels[-1]}，共 {len(noise_levels)} 个级别")

    # 处理每张图片
    for img_file in tqdm(image_files, desc="处理进度"):
        img_path = os.path.join(input_folder, img_file)
        process_image_with_noise_levels(img_path, output_folder, noise_levels)


if __name__ == "__main__":
    # 输入参数设置
    input_folder = "915"  # 输入文件夹路径
    output_folder = "../网络/gaosi1%"  # 输出文件夹路径

    print("开始处理图片...")
    process_folder(input_folder, output_folder)
    print("\n处理完成! 所有结果已保存到:", os.path.abspath(output_folder))
# import os
# import cv2
# import numpy as np
# import random
# from tqdm import tqdm
#
#
# def add_gaussian_noise(image, sigma):
#     """向图像添加高斯噪声"""
#     row, col, ch = image.shape
#     gauss = np.random.normal(0, sigma, (row, col, ch))
#     noisy = image + gauss
#     return np.clip(noisy, 0, 255).astype('uint8')
#
#
# def process_image_with_noise(image_path, output_folder):
#     try:
#         img = cv2.imread(image_path)
#         if img is None:
#             print(f"无法读取图片: {image_path}")
#             return
#
#         # 提取原始文件名（含扩展名）和名称主体
#         full_img_name = os.path.basename(image_path)  # 例如："test_image.jpg"
#         name_body, ext = os.path.splitext(full_img_name)  # 例如：name_body="test_image", ext=".jpg"
#
#         # 生成80个区间（0-80%，每个1%区间随机取一个值，保留一位小数）
#         sigmas = []
#         for i in range(80):
#             start = i * 1.0
#             end = start + 1.0 if i < 79 else 80.0
#             sigma = round(random.uniform(start, end), 1)
#             sigmas.append(sigma)
#
#         # 生成文件名：明确保留原始完整名称，噪声参数作为后缀
#         for sigma in sigmas:
#             # 格式：[原始文件名]_噪声强度[sigma].ext
#             # 例如："test_image_噪声强度5.2.jpg"
#             output_filename = f"{name_body}_gno{sigma:.1f}{ext}"
#             output_path = os.path.join(output_folder, output_filename)
#             cv2.imwrite(output_path, add_gaussian_noise(img, sigma))
#
#     except Exception as e:
#         print(f"处理图片 {image_path} 时出错: {str(e)}")
#
#
# def process_folder(input_folder, output_folder):
#     os.makedirs(output_folder, exist_ok=True)
#     valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
#     image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(valid_extensions)]
#
#     if not image_files:
#         print(f"在 {input_folder} 中未找到图片文件")
#         return
#
#     print(f"处理 {len(image_files)} 张图片，每张生成80个噪声样本...")
#     for img_file in tqdm(image_files, desc="处理进度"):
#         process_image_with_noise(os.path.join(input_folder, img_file), output_folder)


if __name__ == "__main__":
    input_folder = "915"
    output_folder = "../网络/gnoise"
    process_folder(input_folder, output_folder)
    print(f"处理完成，结果保存在：{os.path.abspath(output_folder)}")