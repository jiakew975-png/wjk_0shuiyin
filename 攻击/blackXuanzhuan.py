# import os
# import cv2
# import numpy as np
# from tqdm import tqdm

# def apply_rotation(image, angle, fill_value=0):
#     """
#     应用旋转变换
#     :param image: 输入图像
#     :param angle: 旋转角度（正数逆时针，负数顺时针）
#     :param fill_value: 填充背景的值（默认黑色）
#     :return: 旋转后的图像
#     """
#     h, w = image.shape[:2]
#     center = (w // 2, h // 2)

#     # 获取旋转矩阵
#     M = cv2.getRotationMatrix2D(center, angle, 1.0)

#     # 计算新图像尺寸（避免裁剪）
#     cos = np.abs(M[0, 0])
#     sin = np.abs(M[0, 1])
#     new_w = int((h * sin) + (w * cos))
#     new_h = int((h * cos) + (w * sin))

#     # 调整旋转矩阵中心点
#     M[0, 2] += (new_w / 2) - center[0]
#     M[1, 2] += (new_h / 2) - center[1]

#     # 应用旋转
#     rotated = cv2.warpAffine(
#         image, M, (new_w, new_h),
#         borderValue=(fill_value, fill_value, fill_value))

#     return rotated

# def process_images(input_dir, output_dir):
#     """
#     处理文件夹中的所有图像
#     :param input_dir: 输入文件夹路径
#     :param output_dir: 输出文件夹路径
#     """
#     os.makedirs(output_dir, exist_ok=True)

#     # 支持的图像格式
#     valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
#     image_files = [f for f in os.listdir(input_dir)
#                    if f.lower().endswith(valid_exts)]

#     if not image_files:
#         print(f"在文件夹 {input_dir} 中没有找到图像文件")
#         return

#     # 旋转角度列表（0.5到360度，步长1度）
#     # rotation_angles = [i * 1.0 + 0.5 for i in range(0, 360)]
#     rotation_angles = list(range(0, 361, 1))  # [1, 6, 11, ..., 356]

#     print(f"开始处理 {len(image_files)} 张图像...")
#     print(f"旋转角度: ±1°到±360°，共 {len(rotation_angles)*2} 个级别（步长5°）")

#     # 处理每张图像
#     for img_file in tqdm(image_files, desc="处理进度"):
#         img_path = os.path.join(input_dir, img_file)
#         img = cv2.imread(img_path)

#         if img is None:
#             print(f"无法读取图像: {img_path}")
#             continue

#         base_name, ext = os.path.splitext(img_file)

#         # 对每个角度生成旋转图像
#         for angle in rotation_angles:
#             # 逆时针旋转（正角度）
#             rotated_ccw = apply_rotation(img, angle)
#             output_name = f" {base_name}_ccw_{angle}deg{ext}"#  {base_name}_ccw_{angle}deg{ext}
#             output_path = os.path.join(output_dir, output_name)
#             cv2.imwrite(output_path, rotated_ccw)

#             # # 顺时针旋转（负角度）
#             # rotated_cw = apply_rotation(img, -angle)
#             # output_name = f"deg{ext}"
#             # output_path = os.path.join(output_dir, output_name)
#             # cv2.imwrite(output_path, rotated_cw)

# if __name__ == "__main__":
#     input_directory = "../813"  # 替换为你的输入文件夹路径
#     output_directory = "blackXuanzhuan"  # 输出文件夹路径

#     # 确保输入目录存在
#     if not os.path.exists(input_directory):
#         print(f"输入文件夹不存在: {input_directory}")
#     else:
#         process_images(input_directory, output_directory)
#         print(f"\n旋转攻击完成，结果保存在 {output_directory}")
#         print(f"每张图像生成 {len(range(1, 361, 5))*2} 个旋转版本")  

import os
import cv2
import numpy as np
import random  # 新增：用于生成随机角度
from tqdm import tqdm

def apply_rotation(image, angle, fill_value=0):
    """
    应用旋转变换
    :param image: 输入图像
    :param angle: 旋转角度（正数逆时针，负数顺时针）
    :param fill_value: 填充背景的值（默认黑色）
    :return: 旋转后的图像
    """
    h, w = image.shape[:2]
    center = (w // 2, h // 2)

    # 获取旋转矩阵
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # 计算新图像尺寸（避免裁剪）
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    # 调整旋转矩阵中心点
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]

    # 应用旋转
    rotated = cv2.warpAffine(
        image, M, (new_w, new_h),
        borderValue=(fill_value, fill_value, fill_value))

    return rotated

def process_images(input_dir, output_dir):
    """
    处理文件夹中的所有图像
    :param input_dir: 输入文件夹路径
    :param output_dir: 输出文件夹路径
    """
    os.makedirs(output_dir, exist_ok=True)

    # 支持的图像格式
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    image_files = [f for f in os.listdir(input_dir)
                   if f.lower().endswith(valid_exts)]

    if not image_files:
        print(f"在文件夹 {input_dir} 中没有找到图像文件")
        return

    # 生成每个1度区间内的随机角度 [0,1), [1,2), ..., [359,360)
    rotation_angles = [i + random.random() for i in range(0, 360)]
    
    # 可以按需求添加顺时针旋转（负角度）
    # rotation_angles += [-i - random.random() for i in range(1, 360)]

    print(f"开始处理 {len(image_files)} 张图像...")
    print(f"旋转角度: 在0-1°, 1-2°...359-360°区间内各随机生成1个角度，共 {len(rotation_angles)} 个角度")

    # 处理每张图像
    for img_file in tqdm(image_files, desc="处理进度"):
        img_path = os.path.join(input_dir, img_file)
        img = cv2.imread(img_path)

        if img is None:
            print(f"无法读取图像: {img_path}")
            continue

        base_name, ext = os.path.splitext(img_file)

        # 对每个随机角度生成旋转图像
        for angle in rotation_angles:
            # 应用旋转（根据角度正负自动确定顺逆时针）
            rotated = apply_rotation(img, angle)
            
            # 生成更精确的文件名，保留两位小数显示角度
            direction = "ccw" if angle >= 0 else "cw"
            abs_angle = abs(angle)
            output_name = f"{base_name}_{direction}_{abs_angle:.2f}deg{ext}"
            output_path = os.path.join(output_dir, output_name)
            cv2.imwrite(output_path, rotated)

if __name__ == "__main__":
    input_directory = "915"  # 替换为你的输入文件夹路径
    output_directory = "../网络/test"  # 输出文件夹路径

    # 确保输入目录存在
    if not os.path.exists(input_directory):
        print(f"输入文件夹不存在: {input_directory}")
    else:
        process_images(input_directory, output_directory)
        print(f"\n旋转攻击完成，结果保存在 {output_directory}")
        print(f"每张图像生成 {360} 个旋转版本（每个1度区间1个随机角度）")
