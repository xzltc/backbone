# -*- coding = utf-8 -*-
# @Time :2021/3/1 上午10:42
# @Author: XZL
# @File : predict.py
# @Software: PyCharm
import glob
import re

import numpy as np
import torch
import cv2
from unet_model import UNet

# test_path = './data/test/malignant (123).png'

if __name__ == "__main__":
    device = torch.device('cpu')
    # 加载网络，图片单通道，分类为1。
    net = UNet(n_channels=1, n_classes=1)
    # 将网络拷贝到deivce中
    net.to(device=device)
    # 加载模型参数
    net.load_state_dict(torch.load('best_model_200.pth', map_location=device))
    # 测试模式
    net.eval()

    image_path = sorted(glob.glob('data/test/*.png'), key=lambda x: int(re.findall("[0-9]+", x)[0]))
    # img = cv2.imread(image_path)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img = cv2.resize(img, (572, 572))
    # img = img.reshape(1, img.shape[0], img.shape[1])
    # img_tensor = torch.from_numpy(img)

    for test_path in image_path:
        # 保存结果地址
        save_res_path = (test_path.split('.')[0]).replace('test', 's_out') + '_res.png'
        img = cv2.imread(test_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_h, img_w = img.shape
        img_res = cv2.resize(img, (572, 572))
        img_res = img_res.reshape(1, img_res.shape[0], img_res.shape[1])

        # 转为tensor
        img_tensor = torch.from_numpy(img_res)
        # 将tensor拷贝到device中，只用cpu就是拷贝到cpu中，用cuda就是拷贝到cuda中。
        img_tensor = img_tensor.to(device=device, dtype=torch.float32)
        # 预测
        pred = net(img_tensor.unsqueeze(0))
        # 提取结果
        pred = np.array(pred.data.cpu()[0])[0]
        # 处理结果
        pred[pred >= 0] = 255
        pred[pred < 0] = 0
        # 保存图片
        re_sa_img = cv2.resize(pred, (img_w, img_h))

        blend = cv2.addWeighted(img, 1.0, re_sa_img, 0.6, 0.0, dtype=cv2.CV_32F).astype(np.uint8)
        cv2.imshow('blend', blend)
        # cv2.imwrite(save_res_path, blend)
        cv2.waitKey(0)

    # # 读取所有图片路径
    # tests_path = glob.glob('data/test/*.png')
    # # 遍历所有图片
    # for test_path in tests_path:
    #     # 保存结果地址
    #     save_res_path = test_path.split('.')[0] + '_res.png'
    #     # 读取图片
    #     img = cv2.imread(test_path)
    #     # 转为灰度图
    #     img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    #     # 转为batch为1，通道为1，大小为512*512的数组
    #     img = img.reshape(1, 1, img.shape[0], img.shape[1])
    #     # 转为tensor
    #     img_tensor = torch.from_numpy(img)
    #     # 将tensor拷贝到device中，只用cpu就是拷贝到cpu中，用cuda就是拷贝到cuda中。
    #     img_tensor = img_tensor.to(device=device, dtype=torch.float32)
    #     # 预测
    #     pred = net(img_tensor)
    #     # 提取结果
    #     pred = np.array(pred.data.cpu()[0])[0]
    #     # 处理结果
    #     pred[pred >= 0.5] = 255
    #     pred[pred < 0.5] = 0
    #     # 保存图片
    #     cv2.imwrite(save_res_path, pred)
