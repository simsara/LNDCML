# 首先要引入相关的包
# 引入torch.nn并指定别名
import torch.nn as nn

import torch.nn.functional as F
import torch.optim


class Net(nn.Module):
    def __init__(self):
        # nn.Module子类的函数必须在构造函数中执行父类的构造函数
        super(Net, self).__init__()

        # 卷积层 '1'表示输入图片为单通道， '6'表示输出通道数，'3'表示卷积核为3*3
        self.conv1 = nn.Conv2d(1, 6, 3)
        # 线性层，输入1350个特征，输出10个特征
        self.fc1 = nn.Linear(1350, 10)  # 这里的1350是如何计算的呢？这就要看后面的forward函数

    # 正向传播
    def forward(self, x):
        # 卷积 -> 激活 -> 池化
        x = self.conv1(x)  # 根据卷积的尺寸计算公式，计算结果是30，具体计算公式后面第二章第四节 卷积神经网络 有详细介绍。
        x = F.relu(x)
        x = F.max_pool2d(x, (2, 2))  # 我们使用池化层，计算结果是15
        x = F.relu(x)
        # reshape，‘-1’表示自适应
        # 这里做的就是压扁的操作 就是把后面的[1, 6, 15, 15]压扁，变为 [1, 1350]
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        return x


def print_param(n):
    for parameters in n.parameters():
        print(parameters)


if __name__ == '__main__':
    net = Net()
    print_param(net)
    print('------------------------------------------------------------')
    input = torch.randn(1, 1, 32, 32)  # 这里的对应前面fforward的输入是32
    for epoch in range(10):
        out = net(input)
        optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
        # 先梯度清零(与net.zero_grad()效果一样)
        optimizer.zero_grad()
        y = torch.arange(0, 10).view(1, 10).float()
        criterion = nn.MSELoss()
        loss = criterion(out, y)
        loss.backward()
        print('Epoch: %d, loss: %3.2f' % (epoch, loss.item()))
        # 更新参数
        optimizer.step()
    print('------------------------------------------------------------')
    print_param(net)
