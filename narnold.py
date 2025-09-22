# import os
# import numpy as np
# from PIL import Image

# def inverse_arnold_transform(image, iterations):
#     """对单张置乱图像进行Arnold逆变换（恢复）"""
#     # 转为数组处理（假设输入是二值图或灰度图）
#     img_array = np.array(image)
#     n = img_array.shape[0]  # 图像尺寸（N×N）
#     result = img_array.copy()
    
#     # 逆变换公式：基于正变换推导而来
#     for _ in range(iterations):
#         new_array = np.zeros_like(result)
#         for x in range(n):
#             for y in range(n):
#                 # Arnold逆变换公式
#                 x_new = (2 * x - y) % n
#                 y_new = (-x + y) % n
#                 new_array[x_new, y_new] = result[x, y]
#         result = new_array
    
#     return Image.fromarray(result)

# def process_inverse_folder(input_folder, output_folder, iterations=5):
#     """批量处理文件夹中的置乱图像，进行逆置乱恢复"""
#     # 创建输出文件夹（如果不存在）
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)
#         print(f"创建输出文件夹: {output_folder}")
    
#     # 支持的图像文件格式
#     image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')
    
#     # 遍历输入文件夹中的所有文件
#     for filename in os.listdir(input_folder):
#         # 检查文件是否为图像
#         if filename.lower().endswith(image_extensions):
#             input_path = os.path.join(input_folder, filename)
            
#             try:
#                 # 打开图像
#                 with Image.open(input_path) as img:
#                     # 确保图像是正方形（置乱时已处理为正方形）
#                     width, height = img.size
#                     if width != height:
#                         print(f"警告：{filename} 不是正方形，可能无法正确恢复")
#                         n = min(width, height)
#                         img = img.crop((0, 0, n, n))
                    
#                     # 进行Arnold逆置乱
#                     recovered_img = inverse_arnold_transform(img, iterations)
                    
#                     # 保存恢复后的图像
#                     output_filename = f"recovered_{iterations}_" + filename
#                     output_path = os.path.join(output_folder, output_filename)
#                     recovered_img.save(output_path)
                    
#                     print(f"已恢复: {filename} -> {output_filename}")
                    
#             except Exception as e:
#                 print(f"恢复 {filename} 时出错: {str(e)}")
    
#     print("批量恢复完成!")

# if __name__ == "__main__":
#     # 配置参数（需与置乱时保持一致）
#     input_folder = "output/xor/92"  # 置乱图像所在文件夹
#     output_folder = "9_徽标"        # 恢复后图像保存文件夹
#     transform_iterations = 8                  # 必须与置乱时的迭代次数相同

#     # 执行批量恢复
#     process_inverse_folder(input_folder, output_folder, transform_iterations)
import os
import numpy as np
from PIL import Image

def inverse_arnold_transform(gray_img, iterations, n):
    """
    对单张置乱图像执行Arnold逆变换（恢复）
    :param gray_img: 输入的灰度图像（PIL.Image对象）
    :param iterations: 逆变换迭代次数（需与置乱时一致）
    :param n: 图像尺寸（置乱时的正方形尺寸，此处固定为128）
    :return: 恢复后的图像（PIL.Image对象）
    """
    # 转为NumPy数组处理
    img_array = np.array(gray_img, dtype=np.uint8)
    # 强制调整尺寸为 n×n（确保与置乱时尺寸一致）
    if img_array.shape[0] != n or img_array.shape[1] != n:
        gray_img = gray_img.resize((n, n), resample=Image.LANCZOS)
        img_array = np.array(gray_img)
    result = img_array.copy()
    
    # 执行Arnold逆变换
    for _ in range(iterations):
        new_array = np.zeros_like(result)
        for x in range(n):
            for y in range(n):
                # Arnold逆变换公式
                x_new = (2 * x - y) % n
                y_new = (-x + y) % n
                new_array[x_new, y_new] = result[x, y]
        result = new_array
    
    return Image.fromarray(result)

def process_inverse_folder(input_folder, output_folder, target_iterations, fixed_n=128):
    """
    批量逆置乱：使用固定的尺寸n和迭代次数，无需参数日志
    :param fixed_n: 固定图像尺寸（需与置乱时一致，此处设为128）
    """
    # 创建输出文件夹（不存在则自动创建）
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"创建输出文件夹: {output_folder}")
    
    # 支持的图像格式
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    
    # 遍历输入文件夹中的所有图像
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(image_extensions):
            input_path = os.path.join(input_folder, filename)
            try:
                # 打开图像并转为灰度图
                with Image.open(input_path) as img:
                    gray_img = img.convert('L')
                    # 执行逆置乱（使用固定n和迭代次数）
                    recovered_img = inverse_arnold_transform(gray_img, target_iterations, fixed_n)
                    # 保存恢复后的图像（文件名标注迭代次数）
                    output_filename = f"recovered_{target_iterations}_{filename}"
                    output_path = os.path.join(output_folder, output_filename)
                    recovered_img.save(output_path)
                    print(f"已恢复: {filename} -> {output_filename}")
            except Exception as e:
                print(f"恢复 {filename} 时出错: {str(e)}")
    
    print("逆置乱批量处理完成!")

if __name__ == "__main__":
    # ===================== 配置参数 =====================
    input_folder = "xidian_0shuiyin"  # 置乱图文件夹（含scramble_params.log）
    output_folder = "9_徽标"   # 恢复图文件夹
    transform_iterations = 8        # 逆变换迭代次数（必须与置乱时一致）
    fixed_image_size = 128          # 固定图像尺寸（置乱时的n值，此处设为128）
    # ===================================================
    
    # 执行批量逆置乱
    process_inverse_folder(input_folder, output_folder, transform_iterations, fixed_image_size)