import cv2
import numpy as np
import os


def auto_canny(image, sigma=0.33):
    """
    自动确定 Canny 边缘检测的阈值，基于图像中值。
    Args:
        image: 输入灰度图像。
        sigma: 中值偏差因子。
    Returns:
        包含边缘图像和计算出的低阈值和高阈值的元组。
    """
    v = np.median(image)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    return edged, lower, upper


def process_folder(input_folder, output_folder, use_auto=True, low_thresh=50, high_thresh=150):
    """
    处理文件夹中的所有图片，生成Canny边缘图并保存

    Args:
        input_folder: 输入图片文件夹路径
        output_folder: 输出结果文件夹路径
        use_auto: 是否使用自动阈值，False则使用固定阈值
        low_thresh: 固定低阈值（当use_auto为False时使用）
        high_thresh: 固定高阈值（当use_auto为False时使用）
    """
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 获取输入文件夹中的所有图片文件
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
    image_files = [f for f in os.listdir(input_folder)
                   if f.lower().endswith(valid_extensions)]

    if not image_files:
        print(f"警告: 在文件夹 {input_folder} 中没有找到图片文件")
        return

    print(f"开始处理 {len(image_files)} 张图片...")

    for i, filename in enumerate(image_files, 1):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, f"canny_{filename}")

        # 读取图片
        img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"无法读取图片: {input_path}")
            continue

        # 应用Canny边缘检测
        if use_auto:
            edges, low, high = auto_canny(img)
            print(f"处理 {filename}: 自动阈值 - 低={low}, 高={high}")
        else:
            edges = cv2.Canny(img, low_thresh, high_thresh)
            print(f"处理 {filename}: 固定阈值 - 低={low_thresh}, 高={high_thresh}")

        # 保存结果
        cv2.imwrite(output_path, edges)
        print(f"已保存: {output_path} ({i}/{len(image_files)})")

    print("所有图片处理完成!")


# 使用示例
if __name__ == "__main__":
    input_folder = "/home/hp/桌面/moments/4蝴蝶"  # 输入图片文件夹
    output_folder = "轮廓_canny"  # 输出结果文件夹

    # 方式1: 使用自动阈值
    process_folder(input_folder, output_folder, use_auto=True)

    # 方式2: 使用固定阈值
    # process_folder(input_folder, output_folder, use_auto=False, low_thresh=50, high_thresh=150)

# import cv2
# import numpy as np
#
#
# def extract_white_contour(image_path, output_path='9butterfly.png'):
#     # 读取图片并转为灰度
#     img = cv2.imread(image_path)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
#     # 针对高光区域增强（突出翅膀亮部）
#     _, bright_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)  # 捕捉白色高光
#
#     # 多阶段边缘检测
#     blurred = cv2.GaussianBlur(gray, (3, 3), 0)  # 较小核保留细节
#     edges = cv2.Canny(blurred, 30, 120)  # 较低阈值捕捉纤细纹路
#
#     # 合并高光区域和边缘检测结果
#     combined = cv2.bitwise_or(edges, bright_mask)
#
#     # 生成最终结果（白轮廓黑背景）
#     result = np.zeros_like(gray)  # 全黑背景
#     result[combined > 0] = 255  # 白色轮廓
#
#     cv2.imwrite(output_path, result)
#     print(f"白轮廓图已保存至: {output_path}")
#
#
# # 使用示例
# extract_white_contour('coloful/9.png')