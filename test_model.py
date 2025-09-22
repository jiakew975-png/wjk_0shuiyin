# test_excnn_best_model.py
import torch
from PIL import Image
import torchvision.transforms as transforms
import os
import torch.nn as nn
import random
import numpy as np

# ================= 固定随机种子 =================
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ================= 配置参数 =================
class Opts:
    image_channel = 3
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    imsize = 128
opts = Opts()

# ================= 图像加载与处理 =================
loader = transforms.Compose([
    transforms.Resize([opts.imsize, opts.imsize]),
    transforms.ToTensor()
])
unloader = transforms.ToPILImage()

def image_loader(image_name):
    image = Image.open(image_name).convert('RGB')
    image = loader(image).unsqueeze(0)
    return image

def image_unloader(tensor):
    image = tensor.clone().cpu().squeeze(0)
    image = unloader(image)
    return image

# ================= 定义网络 =================
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class ExCNN(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.e1 = nn.Sequential(
            nn.Conv2d(opts.image_channel, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.e2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.l1 = self.make_block(block, 128, 128)
        self.l2 = nn.Sequential(nn.Conv2d(128, 64, 3, 1, 1),
                                nn.BatchNorm2d(64),
                                nn.ReLU())
        self.l3 = nn.Sequential(nn.Conv2d(128+64, 64, 3, 1, 1),
                                nn.BatchNorm2d(64),
                                nn.ReLU())
        self.l4 = nn.Sequential(nn.Conv2d(128+64+64, 64, 3, 1, 1),
                                nn.BatchNorm2d(64),
                                nn.ReLU())
        self.l5 = nn.Sequential(nn.Conv2d(128+64+64+64, 64, 1, 1, 0),
                                nn.BatchNorm2d(64),
                                nn.ReLU())
        self.l6 = self.make_block(block, 128+64, 128+64)
        self.l7 = self.make_block(block, 128+64, 128+64)
        self.l8 = nn.Sequential(nn.Conv2d(128+64, 128, 3, 1, 1),
                                nn.BatchNorm2d(128),
                                nn.ReLU())
        self.l9 = nn.Sequential(nn.ConvTranspose2d(128+128, 64, 3, 2, 1, 1),
                                nn.BatchNorm2d(64),
                                nn.ReLU())
        self.l10 = nn.Sequential(nn.Conv2d(64, 1, 1, 1, 0),
                                 nn.Sigmoid())

    def make_block(self, block, in_c, out_c):
        return nn.Sequential(block(in_c, out_c, 1, False))

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(e1)
        l1 = self.l1(e2)
        l2 = self.l2(l1)
        l3 = self.l3(torch.cat((l1,l2),1))
        l4 = self.l4(torch.cat((l1,l2,l3),1))
        l5 = self.l5(torch.cat((l1,l2,l3,l4),1))
        l6 = self.l6(torch.cat((l1,l5),1))
        l7 = self.l7(l6)
        l8 = self.l8(l7)
        l9 = self.l9(torch.cat((e2,l8),1))
        l10 = self.l10(l9)
        return l10

# ================= 加载训练好的最优模型 =================
model_path = 'saved_822_Pepper1/excnn_best.pth'
net = ExCNN(ResidualBlock)

checkpoint = torch.load(model_path, map_location=opts.device)
net.load_state_dict(checkpoint['model_state_dict'])   # ✅ 只取模型参数

net.to(opts.device)
# net.eval()  # 测试模式

# ================= 测试 =================
IMAGE1_PATH = 'grouped_test/pingyi0.3/Peppers'
SAVE_PATH = 'result822/Peppers_trans1'
os.makedirs(SAVE_PATH, exist_ok=True)

image_pathes = os.listdir(IMAGE1_PATH)
image_pathes.sort()  # 保证每次顺序一致
for img_path in image_pathes:
    if os.path.isdir(os.path.join(IMAGE1_PATH, img_path)):
        continue
    x1 = image_loader(os.path.join(IMAGE1_PATH, img_path)).to(opts.device)
    with torch.no_grad():
        pred = net(x1)
    res = image_unloader(pred)
    res.save(os.path.join(SAVE_PATH, img_path))

print(f"测试完成，结果已保存到 {SAVE_PATH}")


# # 灰度图测试
# # 测试
# import torch
# from torch.autograd import Variable
# from torch.utils.data import TensorDataset, DataLoader
# from PIL import Image
# import torchvision.transforms as transforms
# import os
# import numpy as np
# import argparse
#
# # ----------------- 配置 -----------------
# opts = argparse.Namespace()
# opts.image_channel = 3  # 输入/输出通道
# opts.device = 'cuda' if torch.cuda.is_available() else 'cpu'
# opts.imsize = 128
# opts.batch_size = 16
#
# # 模型保存路径
# model_path = 'saved_822_Camer/excnn_best.pth'
#
# # 测试集路径
# IMAGE1_PATH = 'grouped_test/pingyi0.3/Cameraman256'
# SAVE_PATH = 'result822/Cameraman256_trans'
#
# if not os.path.exists(SAVE_PATH):
#     os.makedirs(SAVE_PATH)
#
# # ----------------- 图像处理 -----------------
# loader = transforms.Compose([
#     transforms.Resize([opts.imsize, opts.imsize]),
#     transforms.ToTensor()
# ])
#
# unloader = transforms.ToPILImage()
#
# def image_loader(image_name):
#     image = Image.open(image_name).convert('RGB')  # 转成3通道
#     image = loader(image).unsqueeze(0)  # 增加 batch 维度
#     return image
#
# def image_unloader(tensor):
#     """
#     tensor: [B, C, H, W] 或 [C, H, W]
#     输出: PIL Image
#     """
#     image = tensor.clone().cpu()
#     if image.ndim == 4:  # [B,C,H,W]
#         image = image[0]   # 取 batch 第一个元素
#     # 如果是多通道，可直接保存彩色图
#     # 如果只想灰度图，可取第一个通道
#     # image = image[0].unsqueeze(0)  # 单通道灰度图
#     return unloader(image)
#
# # ----------------- 定义网络结构 -----------------
# import torch.nn as nn
#
# def conv3x3(in_channels, out_channels, stride=1):
#     return nn.Conv2d(in_channels, out_channels, kernel_size=3,
#                      stride=stride, padding=1, bias=False)
#
# class ResidualBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, stride=1, downsample=None):
#         super(ResidualBlock, self).__init__()
#         self.conv1 = conv3x3(in_channels, out_channels, stride)
#         self.bn1 = nn.BatchNorm2d(out_channels)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = conv3x3(out_channels, out_channels)
#         self.bn2 = nn.BatchNorm2d(out_channels)
#         self.downsample = downsample
#
#     def forward(self, x):
#         residual = x
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#         out = self.conv2(out)
#         out = self.bn2(out)
#         if self.downsample:
#             residual = self.downsample(x)
#         out += residual
#         out = self.relu(out)
#         return out
#
# class ExCNN(nn.Module):
#     def __init__(self, block):
#         super(ExCNN, self).__init__()
#         self.e1 = nn.Sequential(
#             nn.Conv2d(opts.image_channel, 64, 3, 1, 1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.MaxPool2d(2,2)
#         )
#         self.e2 = nn.Sequential(
#             nn.Conv2d(64, 128, 3, 1, 1),
#             nn.BatchNorm2d(128),
#             nn.ReLU()
#         )
#         self.l1 = self.make_block(block, 128, 128)
#         self.l2 = nn.Sequential(
#             nn.Conv2d(128, 64, 3, 1, 1),
#             nn.BatchNorm2d(64),
#             nn.ReLU()
#         )
#         self.l3 = nn.Sequential(
#             nn.Conv2d(128+64, 64, 3, 1, 1),
#             nn.BatchNorm2d(64),
#             nn.ReLU()
#         )
#         self.l4 = nn.Sequential(
#             nn.Conv2d(128+64+64, 64, 3, 1, 1),
#             nn.BatchNorm2d(64),
#             nn.ReLU()
#         )
#         self.l5 = nn.Sequential(
#             nn.Conv2d(128+64+64+64, 64, 1, 1, 0),
#             nn.BatchNorm2d(64),
#             nn.ReLU()
#         )
#         self.l6 = self.make_block(block, 128+64, 128+64)
#         self.l7 = self.make_block(block, 128+64, 128+64)
#         self.l8 = nn.Sequential(
#             nn.Conv2d(128+64, 128, 3, 1, 1),
#             nn.BatchNorm2d(128),
#             nn.ReLU()
#         )
#         self.l9 = nn.Sequential(
#             nn.ConvTranspose2d(128+128, 64, 3, 2, 1, 1),
#             nn.BatchNorm2d(64),
#             nn.ReLU()
#         )
#         self.l10 = nn.Sequential(
#             nn.Conv2d(64, 3, 1, 1, 0),  # 输出 3 通道
#             nn.Sigmoid()
#         )
#
#     def make_block(self, block, in_c, out_c):
#         layers = []
#         layers.append(block(in_c, out_c, 1, False))
#         return nn.Sequential(*layers)
#
#     def forward(self, x):
#         e1 = self.e1(x)
#         e2 = self.e2(e1)
#         l1 = self.l1(e2)
#         l2 = self.l2(l1)
#         in_l3 = torch.cat((l1,l2), dim=1)
#         l3 = self.l3(in_l3)
#         in_l4 = torch.cat((l1,l2,l3), dim=1)
#         l4 = self.l4(in_l4)
#         in_l5 = torch.cat((l1,l2,l3,l4), dim=1)
#         l5 = self.l5(in_l5)
#         in_l6 = torch.cat((l1,l5), dim=1)
#         l6 = self.l6(in_l6)
#         l7 = self.l7(l6)
#         l8 = self.l8(l7)
#         in_l9 = torch.cat((e2,l8), dim=1)
#         l9 = self.l9(in_l9)
#         l10 = self.l10(l9)
#         return l10
#
# # ----------------- 加载模型 -----------------
# net = ExCNN(ResidualBlock).to(opts.device)
# checkpoint = torch.load(model_path, map_location=opts.device)
# net.load_state_dict(checkpoint['model_state_dict'])
# # net.eval()
# print(f"模型已加载: {model_path}")
#
# # ----------------- 测试 -----------------
# image_pathes = os.listdir(IMAGE1_PATH)
#
# for img_path in image_pathes:
#     if os.path.isdir(os.path.join(IMAGE1_PATH, img_path)):
#         continue
#     x = image_loader(os.path.join(IMAGE1_PATH, img_path)).to(opts.device)
#     with torch.no_grad():
#         pred = net(x)
#     res = image_unloader(pred)
#     res.save(os.path.join(SAVE_PATH, img_path))
#
# print(f"测试完成，结果保存到 {SAVE_PATH}")
