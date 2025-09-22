# import cv2
# import numpy as np
# import os
# from typing import Callable, Tuple
# from pathlib import Path
#
# # 裁剪方式函数（仅保留中心裁剪）
# def center_crop(image: np.ndarray, keep_ratio: float) -> np.ndarray:
#     """中心区域裁剪，按保留比例计算尺寸"""
#     h, w = image.shape[:2]
#     crop_h = int(h * np.sqrt(keep_ratio))
#     crop_w = int(w * np.sqrt(keep_ratio))
#     crop_h = max(32, crop_h)  # 最小尺寸限制，确保有足够区域进行裁剪
#     crop_w = max(32, crop_w)
#
#     start_y = (h - crop_h) // 2
#     start_x = (w - crop_w) // 2
#     return image[start_y:start_y + crop_h, start_x:start_x + crop_w]
#
# def generate_stepwise_ratios(start: float, end: float, step: float) -> list:
#     """生成从start到end（包含end）的等步长比例列表"""
#     ratios = []
#     current = start
#     while current >= end - 1e-9:  # 考虑浮点数精度问题
#         ratios.append(round(current, 2))
#         current -= step
#     return ratios
#
# def process_single_image(image: np.ndarray, image_name: str, output_dir: str,
#                          crop_method: Callable, ratios: list):
#     """处理单张图片，生成所有比例的裁剪样本（不调整尺寸）"""
#     # 为每张图片创建单独的子目录，避免文件名冲突
#     img_output_dir = os.path.join(output_dir, Path(image_name).stem)
#     os.makedirs(img_output_dir, exist_ok=True)
#
#     generated = 0
#     total_samples = len(ratios)
#
#     for ratio in ratios:
#         # 生成裁剪图像（不进行尺寸调整）
#         try:
#             cropped_img = crop_method(image, ratio)
#         except Exception as e:
#             print(f"图片 {image_name} 生成比例为{ratio}的样本失败：{e}")
#             continue
#
#         # 保存图像
#         save_path = os.path.join(img_output_dir,
#             f"{Path(image_name).stem}_crop_{ratio:.2f}.jpg")
#         cv2.imwrite(save_path, cropped_img)
#
#         generated += 1
#
#     return generated, total_samples
#
# def main():
#     # 配置参数
#     input_dir = "../813"  # 输入图片文件夹路径
#     output_dir = "cropped_output_train"  # 输出目录
#     crop_method: Callable = center_crop  # 仅使用中心裁剪
#
#     # 生成从1.0到0.1，步长为0.01的裁剪比例
#     start_ratio = 1.0
#     end_ratio = 0.1
#     step = 0.01
#     ratios = generate_stepwise_ratios(start_ratio, end_ratio, step)
#     total_ratios = len(ratios)
#
#     # 创建输出目录
#     os.makedirs(output_dir, exist_ok=True)
#
#     # 获取输入目录中所有图片文件
#     image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')
#     image_files = [f for f in os.listdir(input_dir)
#                   if f.lower().endswith(image_extensions)
#                   and os.path.isfile(os.path.join(input_dir, f))]
#
#     if not image_files:
#         print(f"错误：在 {input_dir} 中未找到任何图片文件")
#         return
#
#     print(f"发现 {len(image_files)} 张图片需要处理")
#     print(f"将为每张图片生成从{start_ratio}到{end_ratio}，步长为{step}的裁剪样本")
#     print(f"每张图片将生成 {total_ratios} 个样本")
#     print(f"裁剪后的图像将保持裁剪后的原始尺寸，不进行缩放")
#
#     # 处理所有图片
#     total_generated = 0
#     total_expected = len(image_files) * total_ratios
#
#     for i, image_file in enumerate(image_files, 1):
#         image_path = os.path.join(input_dir, image_file)
#         image = cv2.imread(image_path)
#
#         if image is None:
#             print(f"警告：无法读取图像 {image_file}，已跳过")
#             continue
#
#         # 处理单张图片
#         generated, _ = process_single_image(
#             image, image_file, output_dir, crop_method, ratios
#         )
#
#         total_generated += generated
#         print(f"已处理 {i}/{len(image_files)} 张图片 - {image_file} - 生成 {generated} 个样本")
#
#     print(f"\n所有图片处理完成")
#     print(f"总生成样本数：{total_generated}/{total_expected}")
#     print(f"样本保存至：{os.path.abspath(output_dir)}")
#     print(f"裁剪后的图像保持裁剪后的原始尺寸")

# if __name__ == "__main__":
#     main()


#
# import cv2
# import numpy as np
# import os
# from typing import Callable, Tuple, Dict, List
# from pathlib import Path
#
# # 裁剪方式函数集合
# def random_crop(image: np.ndarray, keep_ratio: float) -> np.ndarray:
#     """随机区域裁剪（取中间位置避免极端边缘）"""
#     h, w = image.shape[:2]
#     crop_h = int(h * np.sqrt(keep_ratio))
#     crop_w = int(w * np.sqrt(keep_ratio))
#     crop_h = max(32, crop_h)  # 最小尺寸限制，避免裁剪区域过小
#     crop_w = max(32, crop_w)
#
#     # 边界情况退化为左上角裁剪
#     max_y = h - crop_h
#     max_x = w - crop_w
#     if max_y <= 0 or max_x <= 0:
#         return top_left_crop(image, keep_ratio)
#
#     start_y = (h - crop_h) // 2
#     start_x = (w - crop_w) // 2
#     return image[start_y:start_y + crop_h, start_x:start_x + crop_w]
#
# def top_left_crop(image: np.ndarray, keep_ratio: float) -> np.ndarray:
#     """左上角区域裁剪"""
#     h, w = image.shape[:2]
#     crop_h = int(h * np.sqrt(keep_ratio))
#     crop_w = int(w * np.sqrt(keep_ratio))
#     crop_h = max(32, crop_h)
#     crop_w = max(32, crop_w)
#     return image[:crop_h, :crop_w]
#
# def top_right_crop(image: np.ndarray, keep_ratio: float) -> np.ndarray:
#     """右上角区域裁剪"""
#     h, w = image.shape[:2]
#     crop_h = int(h * np.sqrt(keep_ratio))
#     crop_w = int(w * np.sqrt(keep_ratio))
#     crop_h = max(32, crop_h)
#     crop_w = max(32, crop_w)
#     return image[:crop_h, w - crop_w:]
#
# def bottom_left_crop(image: np.ndarray, keep_ratio: float) -> np.ndarray:
#     """左下角区域裁剪"""
#     h, w = image.shape[:2]
#     crop_h = int(h * np.sqrt(keep_ratio))
#     crop_w = int(w * np.sqrt(keep_ratio))
#     crop_h = max(32, crop_h)
#     crop_w = max(32, crop_w)
#     return image[h - crop_h:, :crop_w]
#
# def bottom_right_crop(image: np.ndarray, keep_ratio: float) -> np.ndarray:
#     """右下角区域裁剪"""
#     h, w = image.shape[:2]
#     crop_h = int(h * np.sqrt(keep_ratio))
#     crop_w = int(w * np.sqrt(keep_ratio))
#     crop_h = max(32, crop_h)
#     crop_w = max(32, crop_w)
#     return image[h - crop_h:, w - crop_w:]
#
# def resize_image(image: np.ndarray, target_size: Tuple[int, int] = (128, 128)) -> np.ndarray:
#     """将图像调整为目标尺寸（128x128）"""
#     return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
#
# # 固定步长比例配置
# MIN_CROP_RATIO = 0.10    # 最小保留比例
# MAX_CROP_RATIO = 1.00   # 最大保留比例
# CROP_STEP = 0.01         # 步长（每0.01一个间隔）
#
# def generate_fixed_ratios() -> List[float]:
#     """生成固定步长的裁剪比例列表（从MAX到MIN）"""
#     ratios = []
#     current_ratio = MAX_CROP_RATIO
#     # 保留3位小数解决浮点数精度误差
#     while current_ratio >= MIN_CROP_RATIO - 1e-9:
#         ratios.append(round(current_ratio, 3))
#         current_ratio -= CROP_STEP
#     return ratios
#
# # 裁剪方式及其权重（随机裁剪50%，四角各12.5%）
# CROP_METHODS: Dict[str, Tuple[Callable, float]] = {
#     "random": (random_crop, 0.5),
#     "top_left": (top_left_crop, 0.125),
#     "top_right": (top_right_crop, 0.125),
#     "bottom_left": (bottom_left_crop, 0.125),
#     "bottom_right": (bottom_right_crop, 0.125)
# }
#
# def calculate_samples_per_method(total_samples: int) -> Dict[str, int]:
#     """按权重计算每种裁剪方式的样本数量"""
#     method_counts = {}
#     total_weight = sum(weight for _, weight in CROP_METHODS.values())
#
#     # 基础数量分配
#     for method, (_, weight) in CROP_METHODS.items():
#         count = int(total_samples * (weight / total_weight))
#         method_counts[method] = count
#
#     # 处理四舍五入误差（优先补充到随机裁剪）
#     total_calculated = sum(method_counts.values())
#     if total_calculated < total_samples:
#         method_counts["random"] += (total_samples - total_calculated)
#
#     return method_counts
#
# def get_crop_method(method_name: str) -> Callable:
#     """根据名称获取裁剪函数"""
#     return CROP_METHODS[method_name][0]
#
# def process_single_image(image_path: str, total_samples: int, root_output_dir: str,
#                          target_size: Tuple[int, int]) -> int:
#     """处理单张图片：为该图片创建独立子文件夹，生成所有裁剪样本"""
#     # 1. 读取图像
#     image = cv2.imread(image_path)
#     if image is None:
#         print(f"错误：无法读取图像 {Path(image_path).name}")
#         return 0
#
#     # 2. 生成当前图片的独立输出子文件夹（以图片名命名）
#     img_stem = Path(image_path).stem  # 获取图片名（不含扩展名）
#     img_output_dir = os.path.join(root_output_dir, img_stem)  # 子文件夹路径
#     os.makedirs(img_output_dir, exist_ok=True)  # 确保文件夹存在
#
#     # 3. 生成固定比例列表
#     fixed_ratios = generate_fixed_ratios()
#     total_fixed_ratios = len(fixed_ratios)
#     if total_fixed_ratios == 0:
#         print(f"错误：裁剪比例配置无效（MIN={MIN_CROP_RATIO}，MAX={MAX_CROP_RATIO}）")
#         return 0
#
#     # 4. 计算每种裁剪方式的样本数量
#     method_counts = calculate_samples_per_method(total_samples)
#     # 生成裁剪方式列表（按数量重复）
#     method_list = []
#     for method, count in method_counts.items():
#         method_list.extend([method] * count)
#
#     # 5. 循环生成并保存样本
#     generated = 0
#     for idx in range(total_samples):
#         # 按顺序选择裁剪方式和比例
#         method_name = method_list[idx]
#         ratio = fixed_ratios[idx % total_fixed_ratios]  # 比例循环使用
#         crop_method = get_crop_method(method_name)
#
#         try:
#             # 裁剪+尺寸调整
#             cropped_img = crop_method(image, ratio)
#             resized_img = resize_image(cropped_img, target_size)
#         except Exception as e:
#             print(f"图片{img_stem}：生成{method_name}样本失败（比例{ratio:.3f}）：{e}")
#             continue
#
#         # 构造保存路径（子文件夹内的文件名）
#         save_filename = f"{img_stem}_{method_name}_crop_{ratio:.3f}_resized.jpg"
#         save_path = os.path.join(img_output_dir, save_filename)
#         cv2.imwrite(save_path, resized_img)
#         generated += 1
#
#     # 输出当前图片的子文件夹路径
#     print(f"样本保存路径：{os.path.abspath(img_output_dir)}")
#     return generated
#
# def main():
#     # 核心配置参数（可按需调整）
#     input_dir = "../813"                # 输入图片文件夹路径
#     total_samples_per_image = 800       # 每张图片生成的总样本数
#     root_output_dir = "cropped_800"    # 根输出目录
#     target_size = (128, 128)            # 输出图像固定尺寸
#
#     # 创建根输出目录
#     os.makedirs(root_output_dir, exist_ok=True)
#     print(f"根输出目录：{os.path.abspath(root_output_dir)}")
#
#     # 获取输入目录中的所有图片文件
#     image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')
#     image_files = [
#         os.path.join(input_dir, f)
#         for f in os.listdir(input_dir)
#         if f.lower().endswith(image_extensions) and os.path.isfile(os.path.join(input_dir, f))
#     ]
#
#     # 检查是否有图片文件
#     if not image_files:
#         print(f"错误：在输入目录 {input_dir} 中未找到任何图片文件")
#         return
#
#     # 批量处理所有图片
#     total_processed_imgs = len(image_files)
#     total_generated_samples = 0
#     print(f"\n共发现 {total_processed_imgs} 张图片，开始处理...\n")
#
#     for img_idx, img_path in enumerate(image_files, 1):
#         img_name = Path(img_path).name
#         print(f"=== 处理进度：{img_idx}/{total_processed_imgs} - 图片：{img_name} ===")
#         generated = process_single_image(img_path, total_samples_per_image, root_output_dir, target_size)
#         total_generated_samples += generated
#         print(f"生成完成：{generated}/{total_samples_per_image} 个样本\n")
#
#     # 输出最终统计信息
#     fixed_ratios = generate_fixed_ratios()
#     print("="*50)
#     print("          所有图片处理完成 - 统计汇总          ")
#     print("="*50)
#     print(f"总处理图片数：{total_processed_imgs} 张")
#     print(f"单张图片目标样本数：{total_samples_per_image} 个")
#     print(f"总目标样本数：{total_processed_imgs * total_samples_per_image} 个")
#     print(f"总成功生成样本数：{total_generated_samples} 个")
#     print(f"根输出目录：{os.path.abspath(root_output_dir)}")
#     print(f"输出图像尺寸：{target_size[0]}x{target_size[1]}")
#     print(f"裁剪比例配置：{MIN_CROP_RATIO} ~ {MAX_CROP_RATIO}（步长{CROP_STEP}，共{len(fixed_ratios)}个固定比例）")
#     print(f"裁剪方式分配：{', '.join([f'{method}({weight*100:.1f}%)' for method, (_, weight) in CROP_METHODS.items()])}")
#     print("="*50)
#
# if __name__ == "__main__":
#     main()
#
import cv2
import numpy as np
import os
from typing import Callable
from pathlib import Path


# 裁剪方式函数（仅保留中心裁剪）
def center_crop(image: np.ndarray, keep_ratio: float) -> np.ndarray:
    """中心区域裁剪，按保留比例计算尺寸"""
    h, w = image.shape[:2]
    crop_h = int(h * np.sqrt(keep_ratio))
    crop_w = int(w * np.sqrt(keep_ratio))
    crop_h = max(32, crop_h)  # 最小尺寸限制，确保有足够区域进行裁剪
    crop_w = max(32, crop_w)

    start_y = (h - crop_h) // 2
    start_x = (w - crop_w) // 2
    return image[start_y:start_y + crop_h, start_x:start_x + crop_w]


def generate_fixed_step_ratios(start: float, end: float, step: float) -> list:
    """
    在[start, end]范围内按固定步长生成裁剪比例
    :param start: 起始比例（较大值）
    :param end: 结束比例（较小值）
    :param step: 步长（每次减少的量）
    :return: 按从大到小排序的比例列表（保留三位小数）
    """
    # 确保start >= end，步长为正数
    if start < end:
        start, end = end, start
    step = abs(step)  # 确保步长为正数

    # 按固定步长生成比例（处理浮点数精度问题）
    ratios = []
    current = start
    while current >= end - 1e-9:  # 允许微小误差，避免因精度问题漏掉end
        ratios.append(round(current, 3))  # 保留三位小数
        current -= step  # 按步长递减

    # 去重（避免因四舍五入产生重复值）
    ratios = list(sorted(set(ratios), reverse=True))
    return ratios


def process_single_image(image: np.ndarray, image_name: str, output_dir: str,
                         crop_method: Callable, ratios: list):
    """处理单张图片，生成所有比例的裁剪样本（不调整尺寸）"""
    # 为每张图片创建单独的子目录，避免文件名冲突
    img_output_dir = os.path.join(output_dir, Path(image_name).stem)
    os.makedirs(img_output_dir, exist_ok=True)

    generated = 0
    total_samples = len(ratios)

    for ratio in ratios:
        # 生成裁剪图像（不进行尺寸调整）
        try:
            cropped_img = crop_method(image, ratio)
        except Exception as e:
            print(f"图片 {image_name} 生成比例为{ratio}的样本失败：{e}")
            continue

        # 保存图像，保留三位小数显示
        save_path = os.path.join(img_output_dir,
                                 f"{Path(image_name).stem}_crop_{ratio:.3f}.jpg")
        cv2.imwrite(save_path, cropped_img)

        generated += 1

    return generated, total_samples


def main():
    # 配置参数（核心修改：使用固定步长替代固定数量）
    input_dir = "915"  # 输入图片文件夹路径
    output_dir = "../网络/test"  # 输出目录
    crop_method: Callable = center_crop  # 仅使用中心裁剪

    # 固定步长参数
    start_ratio = 1.0    # 起始比例（最大值）
    end_ratio = 0.1      # 结束比例（最小值）
    step = 0.005          # 步长（每次减少0.05，可自定义）

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 获取输入目录中所有图片文件
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')
    image_files = [f for f in os.listdir(input_dir)
                   if f.lower().endswith(image_extensions)
                   and os.path.isfile(os.path.join(input_dir, f))]

    if not image_files:
        print(f"错误：在 {input_dir} 中未找到任何图片文件")
        return

    # 按固定步长生成比例列表
    fixed_ratios = generate_fixed_step_ratios(start_ratio, end_ratio, step)
    sample_count = len(fixed_ratios)  # 实际生成的样本数量

    print(f"发现 {len(image_files)} 张图片需要处理")
    print(f"将为每张图片按固定步长生成裁剪比例：从{start_ratio}到{end_ratio}，步长{step}")
    print(f"共生成 {sample_count} 个比例（保留三位小数）")
    print(f"裁剪后的图像将保持裁剪后的原始尺寸，不进行缩放")

    # 处理所有图片
    total_generated = 0
    total_expected = len(image_files) * sample_count

    for i, image_file in enumerate(image_files, 1):
        image_path = os.path.join(input_dir, image_file)
        image = cv2.imread(image_path)

        if image is None:
            print(f"警告：无法读取图像 {image_file}，已跳过")
            continue

        # 处理单张图片（使用固定步长生成的比例）
        generated, _ = process_single_image(
            image, image_file, output_dir, crop_method, fixed_ratios
        )

        total_generated += generated
        print(f"已处理 {i}/{len(image_files)} 张图片 - {image_file} - 生成 {generated} 个样本")

    print(f"\n所有图片处理完成")
    print(f"总生成样本数：{total_generated}/{total_expected}")
    print(f"样本保存至：{os.path.abspath(output_dir)}")
    print(f"裁剪比例范围：{start_ratio} ~ {end_ratio}（步长{step}）")


if __name__ == "__main__":
    main()
    