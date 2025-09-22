import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import TensorDataset, DataLoader

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models

import argparse

import numpy as np

import copy
import cv2
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

opts = argparse.Namespace()
opts.image_channel = 3
opts.device = 'cuda'
opts.batch_size = 16 //8
print(opts.batch_size)
opts.imsize = 128
opts.lr = 5e-3
opts.dtype = torch.FloatTensor
opts.use_cuda = torch.cuda.is_available()

# use_cuda = torch.cuda.is_available()
# dtype = torch.FloatTensor
# # desired size of the output image
# imsize = 128 if use_cuda else 128
# use small size if no gpu
# device = 'cpu'
# if use_cuda:
#     device = 'cuda'

loader = transforms.Compose([
    transforms.Resize([opts.imsize, opts.imsize]),  # scale imported image
    transforms.ToTensor()])  # transform it into a torch tensor

# 打开一张图像，应用一些预处理操作，并增加一个维度以符合神经网络的输入要求
def image_loader(image_name):
    image = Image.open(image_name)
    image = Variable(loader(image))
    # fake batch dimension required to fit network's input dimensions    [batch_size, channels, height, width]
    # 增加批处理维度
    image = image.unsqueeze(0)
    return image


unloader = transforms.ToPILImage()  # reconvert into PIL image

# 函数接收一个图像的张量，处理并显示该图像，允许可选的标题传入
def imshow(tensor, title=None):
    image = tensor.clone().cpu()  # we clone the tensor to not do changes on it
    image = image.view(3, opts.imsize, opts.imsize)  # remove the fake batch dimension
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

# 处理一个张量，将其转换为适合显示或保存的图像格式
def image_unloader(tensor):
    image = tensor.clone().cpu()  # we clone the tensor to not do changes on it
    image = image.view(1, opts.imsize, opts.imsize)  # remove the fake batch dimension
    image = unloader(image)
    return image


# 3x3 convolution
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False)
#     padding=1  使用 1 垫边以保持输入输出的空间维度相同


# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        # inplace=True表示它会在原地修改输入，从而节省内存
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
        # 如果需要下采样，改变 residual 的形状
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


# ResNet
class ExCNN(nn.Module):
    def __init__(self, block):
        super(ExCNN, self).__init__()
        self.e1 = nn.Sequential(
            # param [input_c, output_c, kernel_size, stride, padding]
            nn.Conv2d(opts.image_channel, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )  # 64,32,32
        self.e2 = nn.Sequential(
            # param [input_c, output_c, kernel_size, stride, padding]
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )  # 128,32,32
        self.l1 = self.make_block(block, 128, 128)
        self.l2 = nn.Sequential(
            # param [input_c, output_c, kernel_size, stride, padding]
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )  # 64,32,32
        self.l3 = nn.Sequential(
            # param [input_c, output_c, kernel_size, stride, padding]
            nn.Conv2d(128 + 64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )  # 64,32,32
        self.l4 = nn.Sequential(
            # param [input_c, output_c, kernel_size, stride, padding]
            nn.Conv2d(128 + 64 + 64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )  # 64,32,32
        self.l5 = nn.Sequential(
            # param [input_c, output_c, kernel_size, stride, padding]
            nn.Conv2d(128 + 64 + 64 + 64, 64, 1, 1, 0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )  # 1, 128, 128
        self.l6 = self.make_block(block, 128 + 64, 128 + 64)
        self.l7 = self.make_block(block, 128 + 64, 128 + 64)
        self.l8 = nn.Sequential(
            # param [input_c, output_c, kernel_size, stride, padding]
            nn.Conv2d(128 + 64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )  # 64,32,32
        self.l9 = nn.Sequential(
            nn.ConvTranspose2d(128 + 128, 64, 3, 2, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )  # 64, 64, 64
        self.l10 = nn.Sequential(
            # param [input_c, output_c, kernel_size, stride, padding]
            nn.Conv2d(64, 1, 1, 1, 0),  # 改成 1 通道
            nn.Sigmoid()
        )  # 1, 64, 64

    def make_block(self, block, in_c, out_c):
        layers = []
        layers.append(block(in_c, out_c, 1, False))
        return nn.Sequential(*layers)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(e1)
        l1 = self.l1(e2)
        l2 = self.l2(l1)
        # 将 l1 和 l2 的输出沿通道维度（dim=1）连接起来，形成输入到下一层的张量 in_l3
        in_l3 = torch.cat((l1, l2), dim=1)
        l3 = self.l3(in_l3)
        in_l4 = torch.cat((l1, l2, l3), dim=1)
        l4 = self.l4(in_l4)
        in_l5 = torch.cat((l1, l2, l3, l4), dim=1)
        l5 = self.l5(in_l5)
        in_l6 = torch.cat((l1, l5), dim=1)
        l6 = self.l6(in_l6)
        l7 = self.l7(l6)
        l8 = self.l8(l7)
        in_l9 = torch.cat((e2, l8), dim=1)
        l9 = self.l9(in_l9)
        l10 = self.l10(l9)
        return l10


# 创建了一个特定的神经网络模型，并将其部署到指定的设备上以便进行后续的训练或推断操作
net = ExCNN(ResidualBlock).to(opts.device)
# 初始化一个Adam优化器，用于对神经网络模型中的所有参数进行优化
optimizer = optim.Adam(net.parameters(), lr=opts.lr)

# 创建一个用于计算均方误差（Mean Squared Error，MSE）的损失函数，计算了模型预测值与目标值之间的每个元素的平方差，并对这些差值取均值
criterion = nn.MSELoss()

IMAGE1_PATH = 'grouped_train2000_trans0.5/Peppers'
TARGET_PATH = 'a/f.png'
image_pathes = os.listdir(IMAGE1_PATH)
images1 = []
# 遍历指定目录下的图片文件，将每个图片加载并存储在 images1 列表中，以便进行后续的处理和分析
for img_path in image_pathes:
    if os.path.isdir(img_path):
        continue
    tmp = image_loader(os.path.join(IMAGE1_PATH, img_path)).cpu().detach().numpy()
    images1.append(tmp)


images1 = np.asarray(images1).squeeze(1)

target = image_loader(TARGET_PATH).cpu().detach().numpy()
targets = [target for i in range(len(images1))]
targets = np.asarray(targets).squeeze(1)
train_x1 = torch.from_numpy(images1).type(opts.dtype)

train_y = torch.from_numpy(targets).type(opts.dtype)

train_dataset = TensorDataset(train_x1, train_y)
train_loader = DataLoader(train_dataset, batch_size=opts.batch_size, shuffle=True)



IMAGE1_PATH = 'grouped_train2000_trans0.5/Peppers'
TARGET_PATH = 'a/f.png'
image_pathes = os.listdir(IMAGE1_PATH)
images1 = []
# 遍历存储在 image_pathes 列表中的图片文件路径，使用 image_loader 函数加载该图片，并且将其从GPU中移到CPU上，然后以Numpy数组的形式存储在 images1 列表中
for img_path in image_pathes:
    if os.path.isdir(img_path):
        continue
    tmp = image_loader(os.path.join(IMAGE1_PATH, img_path)).cpu().detach().numpy()
    images1.append(tmp)


# 将存储在 images1 列表中的图像数据转换为 NumPy 数组，并使用 squeeze(1) 方法去除所有维度为1的维度
images1 = np.asarray(images1).squeeze(1)
target = image_loader(TARGET_PATH).cpu().detach().numpy()
targets = [target for i in range(len(images1))]
targets = np.asarray(targets).squeeze(1)
test_x1 = torch.from_numpy(images1).type(opts.dtype)
test_y = torch.from_numpy(targets).type(opts.dtype)
test_dataset = TensorDataset(train_x1, train_y)
test_loader = DataLoader(train_dataset, batch_size=opts.batch_size, shuffle=True)

# 初始化 Adam 优化器
optimizer = optim.Adam(net.parameters(), lr=opts.lr)

# StepLR 调度器：每 50 个 epoch 学习率衰减为原来的 0.5
scheduler = StepLR(optimizer, step_size=50, gamma=0.5)

# 可选：warmup 初期线性增大学习率（可注释）
warmup_epochs = 5
initial_lr = 1e-5  # warmup起始LR
for param_group in optimizer.param_groups:
    param_group['lr'] = initial_lr

best_loss = float('inf')  # 初始化最优 loss
model_save_path = 'saved_822_Pepper1'
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)

for epoch in range(350):
    net.train()
    train_loss = 0

    # warmup策略
    if epoch < warmup_epochs:
        lr = initial_lr + (opts.lr - initial_lr) * (epoch + 1) / warmup_epochs
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    for i, data in enumerate(train_loader):
        x, y = data
        x, y = x.to(opts.device), y.to(opts.device)

        optimizer.zero_grad()
        pred = net(x)
        loss = criterion(pred, y)
        loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optimizer.step()
        train_loss += loss.item()

    avg_loss = train_loss / len(train_loader)
    print(f'[Epoch {epoch+1}] avg_train_loss: {avg_loss:.8f}, lr: {optimizer.param_groups[0]["lr"]:.1e}')

    # StepLR 更新学习率
    if epoch >= warmup_epochs:
        scheduler.step()

    # 保存最优模型
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save({
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': best_loss,
            'opts': opts
        }, os.path.join(model_save_path, 'excnn_best.pth'))
        print(f'--> Best model updated at epoch {epoch + 1}, loss: {best_loss:.8f}')

# 保存最终模型
torch.save({
    'epoch': epoch,
    'model_state_dict': net.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': avg_loss,
    'opts': opts
}, os.path.join(model_save_path, 'excnn_final.pth'))
print(f"训练结束，最终模型已保存到 {model_save_path}")

IMAGE1_PATH = 'grouped_test/Pepper'
SAVE_PATH = 'result822/Peppers'
if not os.path.exists(SAVE_PATH):
    os.mkdir(SAVE_PATH)
image_pathes = os.listdir(IMAGE1_PATH)
images = []
for img_path in image_pathes:
    if os.path.isdir(img_path):
        continue
    x1 = image_loader(os.path.join(IMAGE1_PATH, img_path)).detach()
    x = x1.to(opts.device)
    pred = net(x)
    res = image_unloader(pred)
    res.save(os.path.join(SAVE_PATH, img_path))