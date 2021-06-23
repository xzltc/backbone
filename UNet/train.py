# -*- coding = utf-8 -*-
# @Time :2021/2/28 下午8:43
# @Author: XZL
# @File : train.py
# @Software: PyCharm
from unet_model import UNet
from dataset import BUSI_Loader
import torch.utils.data as Data
from torch import optim
import torch.nn as nn
import numpy as np
import torch


def train_net(net, device, data_path, epochs=50, batch_size=1, lr=0.00001):
    busi_dataset = BUSI_Loader(data_path)
    train_loader = Data.DataLoader(dataset=busi_dataset,
                                   batch_size=batch_size,
                                   shuffle=True)

    # 定义RMSprop算法
    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    # 定义Loss算法
    criterion = nn.BCEWithLogitsLoss()
    # best_loss统计，初始化为正无穷
    best_loss = float('inf')
    # 训练epochs次
    for epoch in range(epochs):
        IoU = 0.0
        IoU_frame = 0  # 参与IoU计算的数目
        # 训练模式
        net.train()
        # 按照batch_size开始训练
        for image, label in train_loader:
            optimizer.zero_grad()
            # 将数据拷贝到device中
            image = image.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.float32)
            # 使用网络参数，输出预测结果
            pred = net(image)
            # 计算loss
            loss = criterion(pred, label)
            print('Loss/train', loss.item())
            # 保存loss值最小的网络参数
            if loss < best_loss:
                best_loss = loss
                torch.save(net.state_dict(), 'best_model_200.pth')
                print('-----best loss updated [' + str(best_loss.item()) + ']-----')
            if label.sum().item() > 0:
                IoU += iou(pred, label)
                IoU_frame += 1
            # 更新参数
            loss.backward()
            optimizer.step()
        print('-/-/-/-/-/-epoch: %d is done [IoU: %f] -/-/-/-/-' % (epoch + 1, IoU / len(train_loader)))


def iou(input, target):
    # clone()深拷贝保留梯度 detach()去除梯度
    i = input.clone()
    t = target.clone()
    i[i >= 0] = 1
    i[i < 0] = 0
    t[t > 0] = 1
    intersection = np.logical_and(target, i.detach()).numpy()[0, 0, :, :]  # 并
    # print(intersection.any())
    union = np.logical_or(target, i.detach()).numpy()[0, 0, :, :]  # 交
    iou = np.sum(intersection) / np.sum(union)  # 求交并比
    return iou


if __name__ == "__main__":
    # 选择设备，有cuda用cuda，没有就用cpu
    device = torch.device('cpu')
    # 加载网络，图片单通道1，分类为1。
    net = UNet(n_channels=1, n_classes=1)
    # 将网络拷贝到deivce中
    net.to(device=device)
    # 指定训练集地址，开始训练
    data_path = "data/train/"
    train_net(net, device, data_path)
