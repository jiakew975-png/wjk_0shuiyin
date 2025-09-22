import cv2
import numpy as np
import os
import pandas as pd


def calculate_nc(image_a, image_b):
    # 将图像转换为灰度图像
    gray_a = cv2.cvtColor(image_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(image_b, cv2.COLOR_BGR2GRAY)

    # 计算归一化的相关性 (Normalized Cross-Correlation)
    norm_a = (gray_a - np.mean(gray_a)) / (np.std(gray_a) + 1e-5)
    norm_b = (gray_b - np.mean(gray_b)) / (np.std(gray_b) + 1e-5)

    nc_value = np.sum(norm_a * norm_b) / np.sqrt(np.sum(norm_a ** 2) * np.sum(norm_b ** 2))
    return nc_value


def main(folder_path, reference_image_path):
    # 读取参考图像
    reference_image = cv2.imread(reference_image_path)

    # 获取文件夹中的所有图片文件
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

    nc_results = []

    # 对每一张图片计算NC值
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)

        # 读取当前图片
        current_image = cv2.imread(image_path)

        # 计算NC值
        nc_value = calculate_nc(reference_image, current_image)

        # 保存结果（包含图片名和NC值）
        nc_results.append({"Image": image_file, "NC Value": nc_value})

    # 创建DataFrame
    df = pd.DataFrame(nc_results)

    # 保存到Excel文件
    # excel_path = "nc_resultsPepper11.xlsx"
    # df.to_excel(excel_path, index=False)
    # print(f"NC值已保存到: {excel_path}")

    # 输出结果到控制台
    print("\nNC值结果:")
    print(df.to_string(index=False))


if __name__ == "__main__":
    folder_path = "output_recovered"  # 替换为你的文件夹路径
    reference_image_path = "binary_output/binary_127_理工大.jpg"  # 替换为你的参考图片路径
    main(folder_path, reference_image_path)