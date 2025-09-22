# import os
# import numpy as np
# from PIL import Image

# def arnold_transform(image, iterations):
#     """对单张图像进行Arnold置乱"""
#     # 转为灰度图并获取像素矩阵
#     img_array = np.array(image.convert('L'))
#     n = img_array.shape[0]  # 图像尺寸（N×N）
#     result = img_array.copy()
    
#     for _ in range(iterations):
#         new_array = np.zeros_like(result)
#         for x in range(n):
#             for y in range(n):
#                 # Arnold变换公式
#                 x_new = (x + y) % n
#                 y_new = (x + 2 * y) % n
#                 new_array[x_new, y_new] = result[x, y]
#         result = new_array
    
#     return Image.fromarray(result)

# def process_folder(input_folder, output_folder, iterations=5):
#     """递归处理文件夹及其所有子文件夹中的图像"""
#     # 支持的图像文件格式
#     image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')
    
#     # 递归遍历所有文件夹和子文件夹
#     for root, dirs, files in os.walk(input_folder):
#         # 计算当前目录相对于输入文件夹的路径
#         relative_path = os.path.relpath(root, input_folder)
#         # 构建对应的输出目录路径
#         output_dir = os.path.join(output_folder, relative_path)
        
#         # 创建输出目录（如果不存在）
#         if not os.path.exists(output_dir):
#             os.makedirs(output_dir)
#             print(f"创建输出文件夹: {output_dir}")
        
#         # 处理当前目录中的所有文件
#         for filename in files:
#             # 检查文件是否为图像
#             if filename.lower().endswith(image_extensions):
#                 input_path = os.path.join(root, filename)
                
#                 try:
#                     # 打开图像
#                     with Image.open(input_path) as img:
#                         # 确保图像是正方形，如果不是则裁剪为正方形
#                         width, height = img.size
#                         n = min(width, height)
#                         img = img.crop((0, 0, n, n))  # 从左上角裁剪
                        
#                         # 进行Arnold置乱
#                         scrambled_img = arnold_transform(img, iterations)
                        
#                         # 保存处理后的图像
#                         output_filename = filename
#                         output_path = os.path.join(output_dir, output_filename)
#                         scrambled_img.save(output_path)
                        
#                         print(f"已处理: {input_path} -> {output_path}")
                        
#                 except Exception as e:
#                     print(f"处理 {input_path} 时出错: {str(e)}")
    
#     print("批量处理完成!")

# if __name__ == "__main__":
#     # 配置参数
#     input_folder = "xidain"   # 输入图像文件夹
#     output_folder = "xidain_置乱"  # 输出置乱图像文件夹
#     transform_iterations = 8  # 置乱迭代次数，次数越多置乱效果越明显
    
#     # 执行批量处理
#     process_folder(input_folder, output_folder, transform_iterations)

import os
import numpy as np
from PIL import Image

def arnold_transform(image, iterations):
    """对单张图像进行Arnold置乱（输出单通道灰度图，记录尺寸）"""
    # 1. 强制转为单通道灰度图（避免通道干扰）
    gray_img = image.convert('L')
    width, height = gray_img.size
    # 2. 裁剪为正方形（从左上角裁剪，记录n值）
    n = min(width, height)
    square_img = gray_img.crop((0, 0, n, n))
    img_array = np.array(square_img, dtype=np.uint8)
    result = img_array.copy()
    
    # 3. Arnold正变换（公式不变）
    for _ in range(iterations):
        new_array = np.zeros_like(result)
        for x in range(n):
            for y in range(n):
                x_new = (x + y) % n
                y_new = (x + 2 * y) % n
                new_array[x_new, y_new] = result[x, y]
        result = new_array
    
    # 返回：置乱图 + 置乱时的n值（用于逆置乱对齐）
    return Image.fromarray(result), n

def process_folder(input_folder, output_folder, iterations=5):
    """递归处理，输出无损PNG，记录置乱参数"""
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')
    os.makedirs(output_folder, exist_ok=True)
    # 记录置乱参数（n值、迭代次数），供逆置乱使用
    param_log = []

    for root, dirs, files in os.walk(input_folder):
        relative_path = os.path.relpath(root, input_folder)
        output_dir = os.path.join(output_folder, relative_path)
        os.makedirs(output_dir, exist_ok=True)

        for filename in files:
            if filename.lower().endswith(image_extensions):
                input_path = os.path.join(root, filename)
                try:
                    with Image.open(input_path) as img:
                        # 执行置乱，获取置乱图和n值
                        scrambled_img, n = arnold_transform(img, iterations)
                        # 4. 强制保存为PNG（无损格式，避免尺寸/像素损失）
                        base_name = os.path.splitext(filename)[0]
                        output_filename = f"{base_name}.png"
                        output_path = os.path.join(output_dir, output_filename)
                        scrambled_img.save(output_path)
                        # 记录参数（用于逆置乱核对）
                        param_log.append(f"{output_filename}: n={n}, iterations={iterations}")
                        print(f"已置乱: {input_path} -> {output_path}")
                except Exception as e:
                    print(f"置乱 {input_path} 出错: {str(e)}")
    
    # 保存参数日志（方便逆置乱时核对）
    log_path = os.path.join(output_folder, "scramble_params.log")
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(param_log))
    print(f"\n置乱完成！参数日志保存至: {log_path}")

if __name__ == "__main__":
    input_folder = "result822/zhiluan/9蝴蝶_trans"  # 置乱图文件夹（含scramble_params.log）
    output_folder = "9_zhiluan_trans"   # 恢复图文件夹
    transform_iterations= 8       # 与置乱完全相同的迭代次数
    process_folder(input_folder, output_folder, transform_iterations)