# -*- coding = utf-8 -*-
# @Time :2021/6/8 6:31 下午
# @Author: XZL
# @File : lenet_model.py
# @Software: PyCharm
import torch.nn as nn
from torch.nn import functional as F


# 在序列模型中的降维
class Reshape(nn.Module):
    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        return x


class LeNet(nn.Module):
    """
    适用于Fashionmnist的数据集的标准LeNet网络
    """

    def __init__(self):
        # 继承父类的init，初始化父类中的方法
        super(LeNet, self).__init__()
        # 这边只写卷积和全连接结构，池化以及激励函数放在forward中
        self.C1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2)
        self.C3 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.F1 = nn.Linear(16 * 5 * 5, 120)
        self.F2 = nn.Linear(120, 84)
        self.OUT = nn.Linear(84, 10)

    def forward(self, x):
        """
        relu和sigmod不同的是在relu在小于0是直接设置为负数，在使用relu激活函数时要注意学习率不能过大
        """
        x = F.relu(self.C1(x))
        x = F.avg_pool2d(x, 2)
        x = F.relu(self.C3(x))
        x = F.avg_pool2d(x, 2)
        # 输入全连接层时降为1维向量
        x = x.view(x.size(0), -1)
        x = F.relu(self.F1(x))
        x = F.relu(self.F2(x))
        x = self.OUT(x)
        return x


class LeNetSeq(nn.Module):
    """
    sequential构造的LeNet网络
    """

    def __init__(self):
        super(LeNetSeq, self).__init__()
        self.net = nn.Sequential(
            Reshape(),
            nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2),  # 默认stride = kernelSize
            nn.Conv2d(6, 16, kernel_size=5), nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2), nn.Flatten(),  # 展平操作一维
            nn.Linear(16 * 5 * 5, 120), nn.ReLU(inplace=True),
            nn.Linear(120, 84), nn.ReLU(inplace=True),
            nn.Linear(84, 10),
        )

    def forward(self, x):
        return self.net(x)

# sq = LetNetSeq().net
# summary(sq)
