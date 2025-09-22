import os
import cv2
import numpy as np
from tqdm import tqdm


def apply_translation_crop_blank(image, dx, dy):
    """
    平移后直接裁剪空白区域（不填充、不恢复原尺寸）
    :param image: 输入图像（BGR格式）
    :param dx: x方向平移量（正值右移，负值左移）
    :param dy: y方向平移量（正值下移，负值上移）
    :return: 裁剪空白后的有效图像（尺寸=原始尺寸-空白尺寸）
    """
    h, w = image.shape[:2]
    
    # 1. 先执行平移（临时保留全尺寸，用于定位有效区域）
    # 平移矩阵：dx右移/负左移，dy下移/负上移
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    translated_full = cv2.warpAffine(image, M, (w, h))  # 暂用原图尺寸，空白区域为黑边（后续裁剪）
    
    # 2. 计算有效像素区域（排除所有空白，仅保留原始图像平移后仍在画面内的部分）
    # -------------- x方向有效范围 --------------
    if dx > 0:
        # 右移：原始图像左侧空出dx像素（空白），有效区域为[dx, w)
        x_valid_start = dx
        x_valid_end = w
    elif dx < 0:
        # 左移：原始图像右侧空出|dx|像素（空白），有效区域为[0, w+dx)（dx为负，w+dx=w-|dx|）
        x_valid_start = 0
        x_valid_end = w + dx
    else:
        # 无x平移：全宽有效
        x_valid_start = 0
        x_valid_end = w
    
    # -------------- y方向有效范围 --------------
    if dy > 0:
        # 下移：原始图像上侧空出dy像素（空白），有效区域为[dy, h)
        y_valid_start = dy
        y_valid_end = h
    elif dy < 0:
        # 上移：原始图像下侧空出|dy|像素（空白），有效区域为[0, h+dy)（dy为负，h+dy=h-|dy|）
        y_valid_start = 0
        y_valid_end = h + dy
    else:
        # 无y平移：全高有效
        y_valid_start = 0
        y_valid_end = h
    
    # 3. 修正有效范围（避免坐标越界，确保有效区域非空）
    # 关键修复：将浮点数转换为整数，因为切片索引必须是整数
    x_valid_start = max(0, int(round(x_valid_start)))
    x_valid_end = min(w, int(round(x_valid_end)))
    y_valid_start = max(0, int(round(y_valid_start)))
    y_valid_end = min(h, int(round(y_valid_end)))
    
    # 4. 裁剪空白区域（仅保留有效像素，尺寸随有效范围变化）
    # 若有效区域为空（极端平移，如平移量≥图像尺寸），返回原图（避免报错）
    if x_valid_start >= x_valid_end or y_valid_start >= y_valid_end:
        return image  # 极端情况：所有像素都被平移出画面，返回原图
    else:
        cropped_valid = translated_full[y_valid_start:y_valid_end, x_valid_start:x_valid_end]
        return cropped_valid


def process_images(input_dir, output_dir, max_trans=50, step=0.5):
    """
    批量生成平移样本：平移后裁剪空白，不填充、不恢复原尺寸
    参数:
    :param input_dir: 输入图像目录
    :param output_dir: 输出图像目录
    :param max_trans: 最大平移像素
    :param step: 平移间隔，默认为0.5像素
    """
    os.makedirs(output_dir, exist_ok=True)

    # 支持的图像格式
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    image_files = [f for f in os.listdir(input_dir)
                   if f.lower().endswith(valid_exts)]

    if not image_files:
        print(f"在文件夹 {input_dir} 中没有找到图像文件")
        return

    # 平移参数：使用0.5像素间隔
    min_trans = 0
    # 生成带0.5间隔的平移强度列表，使用np.around避免浮点数精度问题
    trans_strengths = np.around(np.arange(min_trans, max_trans + step, step), decimals=1)

    # 8个平移方向：方向名 → (dx计算函数, dy计算函数)
    directions = [
        ('U', lambda s: (0, -s)),        # 上移：y负方向（下侧空白）
        ('D', lambda s: (0, s)),         # 下移：y正方向（上侧空白）
        ('L', lambda s: (-s, 0)),        # 左移：x负方向（右侧空白）
        ('R', lambda s: (s, 0)),         # 右移：x正方向（左侧空白）
        ('UL', lambda s: (-s, -s)),      # 左上移：x负+y负（右下空白）
        ('UR', lambda s: (s, -s)),       # 右上移：x正+y负（左下空白）
        ('DL', lambda s: (-s, s)),       # 左下移：x负+y正（右上空白）
        ('DR', lambda s: (s, s))         # 右下移：x正+y正（左上空白）
    ]

    # 打印配置信息
    print(f"===== 平移裁剪配置（无填充+动态尺寸） =====")
    print(f"输入目录：{input_dir}")
    print(f"输出目录：{output_dir}")
    print(f"处理图像数：{len(image_files)} 张")
    print(f"平移强度：{min_trans} ~ {max_trans} 像素（间隔{step}）")
    print(f"平移方向：8个（上/下/左/右/左上/右上/左下/右下）")
    print(f"预计生成样本数：{len(image_files)} × {len(trans_strengths)-1} × {len(directions)}")  # 减1是因为跳过了0
    print(f"核心特性：")
    print(f"  1. 平移后不填充任何空白")
    print(f"  2. 直接裁剪所有空白区域")
    print(f"  3. 输出尺寸=原始尺寸-空白尺寸（不恢复原尺寸）")
    print("======================================")

    # 批量处理每张图像
    for img_file in tqdm(image_files, desc="整体处理进度"):
        img_path = os.path.join(input_dir, img_file)
        img = cv2.imread(img_path)  # 读取原始图像（保留原始尺寸）

        if img is None:
            print(f"\n警告：无法读取图像 {img_file}，已跳过")
            continue

        orig_h, orig_w = img.shape[:2]  # 原始图像尺寸
        base_name, ext = os.path.splitext(img_file)

        # 生成该图像的所有平移-裁剪样本
        for strength in trans_strengths:
            # 跳过0平移（避免生成与原图相同的样本）
            if strength == 0:
                continue
                
            for dir_name, get_dxdy in directions:
                dx, dy = get_dxdy(strength)
                # 执行平移+裁剪空白
                result_img = apply_translation_crop_blank(img, dx, dy)
                # 获取结果图像尺寸（动态变化）
                res_h, res_w = result_img.shape[:2]
                
                # 文件名规则：原始名_方向_平移强度_结果尺寸.ext
                # 强度值保留一位小数，避免文件名中出现过多小数位
                output_name = f"{base_name}_trans{dir_name}{strength}{ext}"
                output_path = os.path.join(output_dir, output_name)
                
                # 保存图像（保持原始格式）
                cv2.imwrite(output_path, result_img)


if __name__ == "__main__":
    # 配置路径（根据实际情况修改）
    INPUT_DIR = "915"                # 输入图像文件夹
    OUTPUT_DIR = "../网络/pingyi"  # 输出文件夹（0.5像素间隔版本）
    MAX_TRANS = 30                      # 最大平移像素
    TRANS_STEP = 0.5                   # 平移间隔，0.5像素

    # 合法性检查
    if not os.path.exists(INPUT_DIR):
        print(f"错误：输入文件夹不存在 → {INPUT_DIR}")
    else:
        process_images(INPUT_DIR, OUTPUT_DIR, MAX_TRANS, TRANS_STEP)
        print(f"\n处理完成！所有样本已保存至：{os.path.abspath(OUTPUT_DIR)}")

