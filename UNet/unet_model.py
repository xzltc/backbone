# -*- coding = utf-8 -*-
# @Time :2021/2/28 下午4:11
# @Author: XZL
# @File : unet_model.py
# @Software: PyCharm
from unet_part import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):  # 输入通道数、输出类别、是否开启双线性差值
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DubleConv(n_channels, 64)  # 卷积核已经定了 不用管尺寸 管理层数就行
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.up1 = Up(1024, 512, bilinear)  # 输出256，另一半256是融合的
        self.up2 = Up(512, 256, bilinear)
        self.up3 = Up(256, 128, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.ouc = OutConv(64, n_classes)

    def forward(self, x):
        # 后面要融合，记录所有x
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        # 上采样并融合
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.ouc(x)


if __name__ == '__main__':
    net = UNet(n_channels=3, n_classes=1)
    print(net)
