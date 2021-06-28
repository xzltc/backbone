# -*- coding = utf-8 -*-
# @Time :2021/6/23 2:55 下午
# @Author: XZL
# @File : nin_model.py
# @Software: PyCharm
import torch
from torch import nn
from utils import *
from torch.nn import functional as F


class NiN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(NiN, self).__init__()
        self.conv1_1 = nn.Conv2d(in_channels, 96, kernel_size=11, stride=4, padding=0),
        self.conv1_2 = nn.Conv2d(96, 96, kernel_size=1)
        self.conv1_3 = nn.Conv2d(96, 96, kernel_size=1)

        self.conv2_1 = nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
        self.conv2_2 = nn.Conv2d(256, 256, kernel_size=1)
        self.conv2_3 = nn.Conv2d(256, 256, kernel_size=1)

        self.conv3_1 = nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
        self.conv3_2 = nn.Conv2d(384, 384, kernel_size=1)
        self.conv3_3 = nn.Conv2d(384, 384, kernel_size=1)

        self.conv4_1 = nn.Conv2d(384, out_channels, kernel_size=3, stride=1, padding=1),
        self.conv4_2 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.conv4_3 = nn.Conv2d(out_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = F.relu(self.conv1_3(x))
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = F.relu(self.conv2_3(x))
        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = F.relu(self.conv3_3(x))
        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))
        x = F.relu(self.conv4_3(x))
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        return x


class SeqNiN(nn.Module):
    def __init__(self, category):
        super(SeqNiN, self).__init__()
        self.net = self.nin(category)

    def nin(self, category):
        # 是由于alexnet改造而来
        net = nn.Sequential(
            self.nin_block(1, 96, kernel_size=11, strides=4, padding=0),
            nn.MaxPool2d(3, stride=2),
            self.nin_block(96, 256, kernel_size=5, strides=1, padding=2),
            nn.MaxPool2d(3, stride=2),
            self.nin_block(256, 384, kernel_size=3, strides=1, padding=1),
            nn.MaxPool2d(3, stride=2), nn.Dropout(0.5),
            self.nin_block(384, category, kernel_size=3, strides=1, padding=1),
            nn.AdaptiveAvgPool2d((1, 1)),  # 最后输出层的平均池化
            nn.Flatten()
        )
        return net

    def nin_block(self, in_channels, out_channels, kernel_size, strides, padding):
        """一个卷积块"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, strides, padding),
            nn.ReLU(), nn.Conv2d(out_channels, out_channels, kernel_size=1),  # padding=0 stride=1
            nn.ReLU(), nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.ReLU()
        )