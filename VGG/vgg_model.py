# -*- coding = utf-8 -*-
# @Time :2021/6/22 3:12 下午
# @Author: XZL
# @File : vgg_model.py
# @Software: PyCharm
import torch
import torch.nn as nn
from torch.nn import functional as F
from utils import *


class VGG(nn.Module):
    def __int__(self, input_channels, output_channels):
        super(VGG, self).__int__()
        # 3 * 224 * 224
        self.conv1_1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)  # 64 * 224 * 224
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # 128 * 112 * 112
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)  # 256 * 56 * 56
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)  # 512 * 28 * 28
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)  # 512 * 14 * 14
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.maxpool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        # view 512 * 7 * 7

        self.fc1 = nn.Linear(512 * 7 * 7, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, output_channels)

    def forward(self, x):
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = self.maxpool1(x)
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = self.maxpool2(x)
        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = F.relu(self.conv3_3(x))
        x = self.maxpool3(x)
        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))
        x = F.relu(self.conv4_3(x))
        x = self.maxpool4(x)
        x = F.relu(self.conv5_1(x))
        x = F.relu(self.conv5_2(x))
        x = F.relu(self.conv5_3(x))
        x = self.maxpool5(x)

        x = x.view(x.size(0), -1)  # 展平
        x = x.relu(self.fc1(x))
        x = x.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.softmax(x, dim=1)  # 用了softmax就得用softmax with cross-entropy loss
        return x


class SeqVgg(nn.Module):
    def __init__(self):
        super(SeqVgg, self).__init__()
        self.vgg16_conv_arch = ((2, 64), (2, 128), (3, 256), (3, 512), (3, 512))  # VGG16-2
        self.net = self.vgg(self.vgg16_conv_arch)

    def vgg(self, conv_arch):
        conv_blks = []
        in_channels = 1
        # 构造所有卷积块结构
        for (num_convs, out_channels) in conv_arch:
            conv_blks.append(
                self.vgg_block(num_convs, in_channels, out_channels)
            )
            in_channels = out_channels

        net = nn.Sequential(*conv_blks, nn.Flatten(),
                            nn.Linear(out_channels * 7 * 7, 4096), nn.ReLU(),
                            nn.Dropout(0.5), nn.Linear(4096, 4096), nn.ReLU(),
                            nn.Dropout(0.5), nn.Linear(4096, 10)
                            )
        return net

    def vgg_block(self, num_convs, in_channels, out_channels):
        """vgg卷积块 用块的方式优雅的实现"""
        layers = []
        for _ in range(num_convs):
            layers.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
            )
            layers.append(nn.ReLU())
            in_channels = out_channels
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        return nn.Sequential(*layers)  # *layers表示把列表中元素拆分方传入 字典前面加两个星号，是将字典的值解开成独立的元素作为形参。

    def forward(self, x):
        return self.net(x)


list = [1, 2, 3]
print(list)
# vgg = SeqVgg().net
# summary(vgg)
