# -*- coding = utf-8 -*-
# @Time :2021/6/20 10:06 下午
# @Author: XZL
# @File : alexnet_model.py
# @Software: PyCharm
import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F


class AlexNet(nn.Module):
    """标准AlexNet网络"""

    def __init__(self):
        super(AlexNet, self).__init__()
        self.C1 = nn.Conv2d(in_channels=1, out_channels=96, kernel_size=11, stride=4)
        self.C2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2)
        self.C3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1)
        self.C4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1)
        self.C5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1)
        self.FC1 = nn.Linear(in_features=256 * 6 * 6, out_features=4096)
        self.FC2 = nn.Linear(in_features=4096, out_features=4096)
        self.FC3 = nn.Linear(in_features=4096, out_features=10)

    def forward(self, x):
        x = F.relu(self.C1(x))
        x = F.max_pool2d(x, 3, 2)
        x = F.relu(self.C2(x))
        x = F.max_pool2d(x, 3, 2)
        x = F.relu(self.C3(x))
        x = F.relu(self.C4(x))
        x = F.relu(self.C5(x))
        x = F.max_pool2d(x, 3, 2)
        x = x.view(x.size(0), -1)  # 展平
        x = F.relu(self.FC1(x))
        x = F.dropout(x, p=0.5)
        x = F.relu(self.FC2(x))
        x = F.dropout(x, p=0.5)
        x = self.FC3(x)
        return x


class SeqAlexNet(nn.Module):
    def __init__(self):
        super(SeqAlexNet, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=11, stride=4), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2), nn.Flatten(),
            nn.Linear(9216, 4096), nn.ReLU(), nn.Dropout(p=0.5),
            nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(p=0.5),
            nn.Linear(4096, 10)
        )

    def forward(self, x):
        return self.net(x)


class SeqAlexNetEnhance(nn.Module):
    """尝试全掉AlexNet全连接层，用1*1卷积和全局池化层替代"""

    def __init__(self):
        super(SeqAlexNetEnhance, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(256, 4096, kernel_size=1), nn.ReLU(), nn.Dropout2d(p=0.5),
            # padding=0 stride=1
            nn.Conv2d(4096, 4096, kernel_size=1), nn.ReLU(), nn.Dropout2d(p=0.5),
            nn.Conv2d(4096, 10, kernel_size=1),
            nn.AdaptiveAvgPool2d((1, 1)),  # 最后输出层的平均池化为1*1
            nn.Flatten()
        )

    def forward(self, x):
        return self.net(x)


class VNA(nn.Module):
    """按照VGG的思路写的网络，epoch-10精确度最高"""
    def __init__(self):
        super(VNA, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            # nn.BatchNorm2d(64, 0.8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 64*112*112
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 128*56*56
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 256*28*28
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # # 512*14*14
            # nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512, 0.8), nn.ReLU(),
            # nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512, 0.8), nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            # 512*7*7
            nn.Flatten(),
            nn.Linear(5 * 5 * 512, 4096), nn.ReLU(), nn.Dropout2d(p=0.5),
            nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout2d(p=0.5),
            nn.Linear(4096, 10)

            # nn.Conv2d(384, 4096, kernel_size=1), nn.ReLU(), nn.Dropout2d(p=0.5),
            # # padding=0 stride=1
            # nn.Conv2d(4096, 4096, kernel_size=1), nn.ReLU(), nn.Dropout2d(p=0.5),
            # nn.Conv2d(4096, 10, kernel_size=1),
            # nn.AdaptiveAvgPool2d((1, 1)),  # 最后输出层的平均池化为1*1
            # nn.Flatten()
        )

    def forward(self, x):
        return self.net(x)
