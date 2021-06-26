# -*- coding = utf-8 -*-
# @Time :2021/6/8 8:21 下午
# @Author: XZL
# @File : utils.py
# @Software: PyCharm
import torch
from torch import nn
from d2l import torch as d2l
import sys
import time
import torchvision
import numpy as np
from IPython import display
from matplotlib import pyplot as plt
from torchvision import transforms
from torch.utils import data

utils = sys.modules[__name__]


def try_gpu(i=0):
    """Return gpu(i) if exists, otherwise return cpu()."""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


def get_dataloader_workers():
    """Use 4 processes to read the data."""
    return 4


def use_svg_display():
    """Use the svg format to display a plot in Jupyter."""
    display.set_matplotlib_formats('svg')


# Defined in file: ./chapter_preliminaries/calculus.md
def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """Set the axes for matplotlib."""
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()


class Accumulator:
    """For accumulating sums over `n` variables."""

    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class Timer:
    """记录运行时间，用定义类的方式"""

    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        """Start the timer."""
        self.tik = time.time()

    def stop(self):
        """Stop the timer and record the time in a list."""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """Return the average time."""
        return sum(self.times) / len(self.times)

    def sum(self):
        """Return the sum of time."""
        return sum(self.times)

    def cumsum(self):
        """Return the accumulated time."""
        return np.array(self.times).cumsum().tolist()


class Animator:
    """For plotting data in animation."""

    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        # Incrementally plot multiple lines
        if legend is None:
            legend = []
        d2l.use_svg_display()
        self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # Use a lambda function to capture arguments
        self.config_axes = lambda: d2l.set_axes(self.axes[
                                                    0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # Add multiple data points into the figure
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        # display.display(self.fig)
        # display.clear_output(wait=True)

    def show_plot(self):
        """因为plot只能显示一次，所以在结束时添加显示操作 """
        plt.show()


def summary(net):
    """
    打印Sequence模型中每一层的情况
    :param net: model
    """

    x = torch.rand(size=(1, 1, 32, 32), dtype=torch.float32)  # lenet
    x = torch.rand(size=(1, 1, 227, 227), dtype=torch.float32)  # alexnet
    for layer in net:
        x = layer(x)
        print(layer.__class__.__name__, "output shape: \t", x.shape)


def accuracy(y_hat, y):
    """Compute the number of correct predictions."""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = argmax(y_hat, axis=1)
    cmp = astype(y_hat, y.dtype) == y
    return float(reduce_sum(astype(cmp, y.dtype)))


def evaluate_accuracy_gpu(net, data_iter, device=None):
    """
    衡量精确度
    :param net: model
    :param data_iter:
    :param device: cpu or gpu
    """
    if isinstance(net, torch.nn.Module):
        net.eval()  # 评估模式，关闭dropout， .train()模式是开启#
        if not device:
            device = next(iter(net.parameters())).device
    metric = Accumulator(2)
    for X, y in data_iter:
        if isinstance(X, list):
            X = [x.to(device) for x in x]
        else:
            X = X.to(device)
        y = y.to(device)
        # 1.net(x)计算应用网络后的结果 2.计算测试集精确度 3.添加到metric中
        metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]


def train_ch6(net, train_iter, test_iter, num_epochs, lr, device):
    """在GPU或者CPU上训练和验证的一个标准流程"""

    def init_weights(m):
        """
        xavier_uniform方法初始化权重
        :param m: 当前的一层
        """
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)

    net.apply(init_weights)  # model.apply(fn) 递归在每一层调用fn
    print('training on', device)
    net.to(device)  # 将模型挪到指定设备上cpu&gpu
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)  # 随机梯度下降 net.parameters()查看网络参数
    loss = nn.CrossEntropyLoss()  # 交叉熵loss 多类分类问题
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs],
                        legend=['train loss', 'train acc', 'test acc'])
    t_time, timer, num_batches = Timer(), Timer(), len(train_iter)  # 初始化时间，train_iter?
    t_time.start()  # 验证整个训练执行的总时间
    for epoch in range(num_epochs):
        # Sum of training loss, sum of training accuracy, no. of examples
        metric = Accumulator(3)
        net.train()
        for i, (X, y) in enumerate(train_iter):  # 每次取一个batch出来
            timer.start()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)  # 把batch挪到对应的设备
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (train_l, train_acc, None))
                print('e:', epoch + 1, "- add loss:", train_l)  # .item()一个元素tensor可以得到元素值
                # print(f'current loss {train_l:.3f}')
                print(timer.avg())
        # 每次结束验证集的测试
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
    t_time.stop()
    animator.show_plot()
    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)} all time {t_time.sum()}')


def load_data_fashion_mnist(path, batch_size, resize=None, download=False):
    """下载数据集并加载到dataloader实现复用的操作"""
    trans = [transforms.ToTensor(),
             transforms.RandomHorizontalFlip(),
             transforms.RandomVerticalFlip(),
             transforms.RandomAutocontrast()
             ]  # 对传入的图像先做一个ToTensor()的操作
    # 如果需要resize，则传入参数
    if resize:
        trans.insert(0, transforms.Resize(resize))
    # wow标准写法，多个transforms则在这里进行组合 大于两个给List
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(root=path,
                                                    train=True,
                                                    transform=trans,
                                                    download=download)
    mnist_test = torchvision.datasets.FashionMNIST(root=path,
                                                   train=False,
                                                   transform=trans,
                                                   download=download)
    # (train dataloader,test dataloader)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True,
                            num_workers=get_dataloader_workers()),
            data.DataLoader(mnist_test, batch_size, shuffle=False,
                            num_workers=get_dataloader_workers()))


def load_data_mnist(path, batch_size, resize=None, download=False):
    """下载数据集并加载到dataloader实现复用的操作"""
    trans = [transforms.ToTensor(),
             transforms.RandomHorizontalFlip(),  # 随机水平翻转
             transforms.RandomVerticalFlip(),  # 随机垂直翻转
             ]
    # 如果需要resize，则传入参数
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.MNIST(root=path,
                                             train=True,
                                             transform=trans,
                                             download=download)
    mnist_test = torchvision.datasets.MNIST(root=path,
                                            train=False,
                                            transform=trans,
                                            download=download)
    # (train dataloader,test dataloader)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True,
                            num_workers=get_dataloader_workers()),
            data.DataLoader(mnist_test, batch_size, shuffle=False,
                            num_workers=get_dataloader_workers()))


def load_data_cifar_10(path, batch_size, resize=None, download=False):
    """下载数据集并加载到dataloader实现复用的操作"""
    trans = [transforms.ToTensor(),
             # transforms.RandomHorizontalFlip(),
             # transforms.RandomVerticalFlip(),
             # transforms.ColorJitter(brightness=0.5),
             # transforms.ColorJitter(contrast=0.5),
             # transforms.ColorJitter(hue=0.5)
             ]  # 对传入的图像先做一个ToTensor()的操作
    # 如果需要resize，则传入参数
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    CIFAR10_train = torchvision.datasets.CIFAR10(root=path,
                                                 train=True,
                                                 transform=trans,
                                                 download=download)
    CIFAR10_test = torchvision.datasets.CIFAR10(root=path,
                                                train=False,
                                                transform=trans,
                                                download=download)
    # (train dataloader,test dataloader)
    return (data.DataLoader(CIFAR10_train, batch_size, shuffle=True,
                            num_workers=1),
            data.DataLoader(CIFAR10_test, batch_size, shuffle=False,
                            num_workers=1))


reduce_sum = lambda x, *args, **kwargs: x.sum(*args, **kwargs)
argmax = lambda x, *args, **kwargs: x.argmax(*args, **kwargs)
astype = lambda x, *args, **kwargs: x.type(*args, **kwargs)
