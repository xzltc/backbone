# -*- coding = utf-8 -*-
# @Time :2021/2/28 下午3:25
# @Author: XZL
# @File : unet_part.py
# @Software: PyCharm
import torch
import torch.nn as nn
import torch.nn.functional as F


# 连续两次卷积操作模块
class DubleConv(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(DubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=0),  # 572-3+1 = 570
            # 卷积层之后总会添加BatchNorm2d进行数据的归一化处理，这使得数据在Relu之前不会因为
            # 数据过大而导致网络性能的不稳定。
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),  # inplace直接改变原来的值，减少内存调度
            nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=0),  # padding为0图像会缩小
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


# 下采样输出模块
class Down(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(Down, self).__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(2),  # 下采样以2为步长
            DubleConv(input_channels, output_channels)
        )

    def forward(self, x):
        return self.down(x)


# 上采样
class Up(nn.Module):
    def __init__(self, input_channels, output_channels, bilinear=True):
        # bilinear表示使用双线性差值上采样，否则使用反卷积
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        else:
            # 反卷积变大
            self.up = nn.ConvTranspose2d(input_channels, input_channels // 2, kernel_size=2, stride=2)

        self.conv = DubleConv(input_channels, output_channels)

    def forward(self, x1, x2):
        # x1上采样 -> 特征融合 -> 双卷积
        x1 = self.up(x1)
        # x2 特征融合，比x1大需要对x1进行padding，计算差距
        # 这里的融合没有按照论文中的方法，通过对上采样的x进行padding方式再进行融合，论文是反一反
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # x2和x1在列维度上融合
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


# 最终输出层
class OutConv(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(OutConv, self).__init__()
        self.out_conv = nn.Conv2d(input_channels, output_channels, kernel_size=1)

    def forward(self, x):
        return self.out_conv(x)
