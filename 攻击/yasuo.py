import os
import cv2
from tqdm import tqdm

def apply_jpeg_compression(image, quality):
    """
    应用JPEG压缩
    :param image: 输入图像
    :param quality: 压缩质量 (1-100)
    :return: 压缩后的图像
    """
    # 编码参数设置
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]

    # 临时压缩到内存缓冲区
    ret, buffer = cv2.imencode('.jpg', image, encode_params)

    # 解码回图像
    compressed_img = cv2.imdecode(buffer, cv2.IMREAD_UNCHANGED)

    return compressed_img

def process_images(input_dir, output_dir):
    """
    处理文件夹中的所有图像
    :param input_dir: 输入文件夹路径
    :param output_dir: 输出文件夹路径
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 支持的图像格式
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')

    # 获取所有图像文件
    image_files = [f for f in os.listdir(input_dir)
                   if f.lower().endswith(valid_exts)]

    if not image_files:
        print(f"在文件夹 {input_dir} 中没有找到图像文件")
        return

    # 压缩质量列表 (1-100，步长1)
    quality_levels = list(range(100,0, -1))  # 从100到1，步长-1

    print(f"开始处理 {len(image_files)} 张图像...")
    print(f"压缩质量级别: {quality_levels[0]} 到 {quality_levels[-1]}，共 {len(quality_levels)} 个级别")

    # 处理每张图像
    for img_file in tqdm(image_files, desc="处理进度"):
        img_path = os.path.join(input_dir, img_file)
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)  # 保留原始通道数

        if img is None:
            print(f"无法读取图像: {img_path}")
            continue

        # 获取原始图像信息
        base_name, ext = os.path.splitext(img_file)

        # 为每个压缩级别生成并保存图像
        for quality in quality_levels:
            compressed_img = apply_jpeg_compression(img, quality)

            # 构造输出文件名 (统一使用.jpg扩展名)
            output_name = f"{base_name}_jpeg_q{quality:03d}.jpg"  # 格式化为3位数
            output_path = os.path.join(output_dir, output_name)

            # 保存压缩后的图像
            cv2.imwrite(output_path, compressed_img, [cv2.IMWRITE_JPEG_QUALITY, quality])

if __name__ == "__main__":
    # 设置输入输出目录
    input_directory = "915"  # 替换为你的输入文件夹路径
    output_directory = "jpeg_test"  # 输出文件夹路径

    # 确保输入目录存在
    if not os.path.exists(input_directory):
        print(f"输入文件夹不存在: {input_directory}")
    else:
        process_images(input_directory, output_directory)
        print(f"\n所有图像处理完成，结果已保存到 {output_directory}")
        print(f"每张图像生成 {len(range(100, 0, -2))} 个压缩版本")