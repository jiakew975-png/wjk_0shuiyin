import cv2
import os
import numpy as np
from natsort import natsorted  # 用于自然排序（支持数字顺序）

def xor_selected_with_folder(selected_images, folder1_path, folder2_path, output_folder_path):
    """
    第一个文件夹中指定的图片与第二个文件夹中的所有图片分别执行异或操作

    :param selected_images: 第一个文件夹中需要处理的指定图片名称列表
    :param folder1_path: 第一个文件夹路径（包含指定图片）
    :param folder2_path: 第二个文件夹路径
    :param output_folder_path: 结果保存文件夹路径
    """
    # 获取第二个文件夹中的所有图片（按自然排序）
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    folder2_images = [
        f for f in os.listdir(folder2_path)
        if f.lower().endswith(image_extensions)
    ]
    folder2_images = natsorted(folder2_images)

    # 验证第一个文件夹中指定的图片是否存在
    valid_selected = []
    for img_name in selected_images:
        img_path = os.path.join(folder1_path, img_name)
        if os.path.exists(img_path) and img_name.lower().endswith(image_extensions):
            valid_selected.append(img_name)
        else:
            print(f"警告：指定图片 '{img_name}' 在 {folder1_path} 中不存在或不是图片文件，已跳过")

    if not valid_selected:
        print("错误：没有有效的指定图片，无法进行异或操作")
        return

    # 创建输出文件夹
    os.makedirs(output_folder_path, exist_ok=True)
    print(f"已创建输出文件夹：{output_folder_path}")
    print(f"将处理 {len(valid_selected)} 张指定图片与 {len(folder2_images)} 张目标图片的异或操作...")

    # 遍历第一个文件夹中的指定图片
    for selected_idx, selected_img_name in enumerate(valid_selected, 1):
        selected_img_path = os.path.join(folder1_path, selected_img_name)

        # 读取指定图片
        selected_img = cv2.imread(selected_img_path, cv2.IMREAD_UNCHANGED)
        if selected_img is None:
            print(f"警告：无法读取指定图片 {selected_img_name}，已跳过")
            continue

        # 与第二个文件夹中的所有图片进行异或
        for target_idx, target_img_name in enumerate(folder2_images, 1):
            target_img_path = os.path.join(folder2_path, target_img_name)

            # 读取目标图片
            target_img = cv2.imread(target_img_path, cv2.IMREAD_UNCHANGED)
            if target_img is None:
                print(f"警告：无法读取目标图片 {target_img_name}，已跳过")
                continue

            # 调整尺寸（确保两张图尺寸一致）
            if selected_img.shape[:2] != target_img.shape[:2]:  # 只比较宽高，忽略通道数
                target_img = cv2.resize(target_img, (selected_img.shape[1], selected_img.shape[0]),
                                      interpolation=cv2.INTER_AREA)

            # 统一通道数（若一张是灰度图，一张是彩色图）
            if len(selected_img.shape) != len(target_img.shape):
                if len(selected_img.shape) == 3:  # 指定图是彩色（3通道），目标图是灰度（单通道）
                    target_img = cv2.cvtColor(target_img, cv2.COLOR_GRAY2BGR)
                else:  # 指定图是灰度，目标图是彩色
                    selected_img = cv2.cvtColor(selected_img, cv2.COLOR_GRAY2BGR)

            # 执行异或操作
            xor_result = cv2.bitwise_xor(selected_img, target_img)

            # 保存结果（文件名包含指定图片和目标图片名称）
            ext = os.path.splitext(selected_img_name)[1] if os.path.splitext(selected_img_name)[1] else '.png'
            result_name = f"{os.path.splitext(target_img_name)[0]}{ext}"
            result_path = os.path.join(output_folder_path, result_name)
            cv2.imwrite(result_path, xor_result)

            # 打印进度
            if (target_idx % 10 == 0) or (target_idx == len(folder2_images)):
                print(f"已处理 {selected_idx}/{len(valid_selected)} 张指定图片与 {target_idx}/{len(folder2_images)} 张目标图片")

    print(f"所有异或操作完成，结果保存至：{output_folder_path}")

if __name__ == "__main__":
    # 配置参数
    folder1 = "xidian_0shuiyin"          # 第一个文件夹路径
    folder2 = "9_zhiluan_trans"  # 第二个文件夹路径
    output_folder = "xidian_0shuiyin"    # 结果保存文件夹路径

    # 在这里指定第一个文件夹中需要处理的图片（包含文件扩展名）
    # 例如：selected_images = ["image1.png", "photo2.jpg", "pic3.bmp"]
    selected_images = ["e.png"]  # 修改为你需要的图片名称

    # 执行异或操作
    xor_selected_with_folder(selected_images, folder1, folder2, output_folder)

# import cv2
# import os
# import numpy as np
# from natsort import natsorted  # 用于自然排序（支持数字顺序）

# def xor_two_folders(folder1_path, folder2_path, output_folder_path):
#     """
#     对两个文件夹中的图片按顺序依次执行异或操作（不进行二值化）

#     :param folder1_path: 第一个文件夹路径
#     :param folder2_path: 第二个文件夹路径
#     :param output_folder_path: 结果保存文件夹路径
#     """
#     # 获取两个文件夹中的图片文件（按自然排序）
#     image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
#     folder1_images = [
#         f for f in os.listdir(folder1_path)
#         if f.lower().endswith(image_extensions)
#     ]
#     folder2_images = [
#         f for f in os.listdir(folder2_path)
#         if f.lower().endswith(image_extensions)
#     ]

#     # 自然排序（确保"img1.png"在"img10.png"之前）
#     folder1_images = natsorted(folder1_images)
#     folder2_images = natsorted(folder2_images)

#     # 检查图片数量是否一致
#     if len(folder1_images) != len(folder2_images):
#         print(f"警告：两个文件夹图片数量不一致（{len(folder1_images)} vs {len(folder2_images)}）")
#         print("将按较少数量的图片进行配对")
#         min_count = min(len(folder1_images), len(folder2_images))
#         folder1_images = folder1_images[:min_count]
#         folder2_images = folder2_images[:min_count]

#     # 创建输出文件夹
#     os.makedirs(output_folder_path, exist_ok=True)
#     print(f"已创建输出文件夹：{output_folder_path}")
#     print(f"开始处理 {len(folder1_images)} 对图片...")

#     # 依次对两个文件夹中的图片执行异或操作
#     for i, (img1_name, img2_name) in enumerate(zip(folder1_images, folder2_images)):
#         img1_path = os.path.join(folder1_path, img1_name)
#         img2_path = os.path.join(folder2_path, img2_name)

#         # 读取图片（保留原始通道信息）
#         img1 = cv2.imread(img1_path, cv2.IMREAD_UNCHANGED)
#         img2 = cv2.imread(img2_path, cv2.IMREAD_UNCHANGED)

#         # 检查图片是否读取成功
#         if img1 is None:
#             print(f"警告：无法读取 {img1_name}，已跳过该对图片")
#             continue
#         if img2 is None:
#             print(f"警告：无法读取 {img2_name}，已跳过该对图片")
#             continue

#         # 调整尺寸（确保两张图尺寸一致）
#         if img1.shape[:2] != img2.shape[:2]:  # 只比较宽高，忽略通道数
#             img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]), interpolation=cv2.INTER_AREA)
#             print(f"警告：{img1_name} 与 {img2_name} 尺寸不一致，已自动调整")

#         # 统一通道数（若一张是灰度图，一张是彩色图）
#         if len(img1.shape) != len(img2.shape):
#             if len(img1.shape) == 3:  # 图1是彩色（3通道），图2是灰度（单通道）
#                 img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
#             else:  # 图1是灰度，图2是彩色
#                 img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
#             print(f"警告：{img1_name} 与 {img2_name} 通道数不一致，已自动统一为3通道")

#         # 直接执行异或操作（不进行二值化）
#         xor_result = cv2.bitwise_xor(img1, img2)

#         # 保存结果（文件名包含两对原图名称）
#         ext = os.path.splitext(img1_name)[1] if os.path.splitext(img1_name)[1] else '.png'
#         result_name = f"xor_{i+1}_{os.path.splitext(img1_name)[0]}_{os.path.splitext(img2_name)[0]}{ext}"
#         result_path = os.path.join(output_folder_path, result_name)
#         cv2.imwrite(result_path, xor_result)

#         # 打印进度
#         if (i + 1) % 10 == 0:
#             print(f"已处理 {i+1}/{len(folder1_images)} 对图片")

#     print(f"所有图片异或操作完成，结果保存至：{output_folder_path}")

# if __name__ == "__main__":
#     # 配置两个文件夹路径和输出路径
#     folder1 = "output/all"          # 第一个文件夹路径
#     folder2 = "xau_0shuiyin"  # 第二个文件夹路径
#     output_folder = "output/alln"    # 结果保存文件夹路径

#     # 执行异或操作
#     xor_two_folders(folder1, folder2, output_folder)