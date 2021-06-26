# -*- coding = utf-8 -*-
# @Time :2021/6/23 2:55 下午
# @Author: XZL
# @File : nin_model.py
# @Software: PyCharm
import torch
from torch import nn
from utils import *


class NiN(nn.Module):
    def __init__(self):
        super(NiN, self).__init__()
        pass


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
