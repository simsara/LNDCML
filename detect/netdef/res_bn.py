import torch
import torch.nn as nn
import torch.nn.functional as F


class Bottleneck(nn.Module):
    def __init__(self, last_planes, in_planes, out_planes, dense_depth, stride, first_layer, attention_module=None):
        super(Bottleneck, self).__init__()
        self.out_planes = out_planes
        self.dense_depth = dense_depth
        self.last_planes = last_planes
        self.in_planes = in_planes

        self.conv1 = nn.Conv3d(last_planes, in_planes, kernel_size=1, bias=False)  # 24 24
        self.bn1 = nn.BatchNorm3d(in_planes)
        # group = 8，将原输入分为8组，每组channel重复用out_channels/8次
        # groups决定了将原输入分为几组，而每组channel重用几次，由out_channels/groups计算得到
        self.conv2 = nn.Conv3d(in_planes, in_planes, kernel_size=3, stride=stride, padding=1, groups=8, bias=False)
        self.bn2 = nn.BatchNorm3d(in_planes)
        #
        self.conv3 = nn.Conv3d(in_planes, out_planes + dense_depth, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(out_planes + dense_depth)

        self.shortcut = nn.Sequential()
        if first_layer:
            self.shortcut = nn.Sequential(
                nn.Conv3d(last_planes, out_planes + dense_depth, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_planes + dense_depth)
            )
        if attention_module is not None:
            self.attention = attention_module(channel=out_planes, reduction=2)
        else:
            self.attention = None

    # 一个DPN模块
    def forward(self, x):
        """
        y = G([x[:d]+F(x)[:d],x[d:],F(x)[d:]])
        y是一个DPN块的最后特征
        G是relu激活函数
        F是卷积层函数
        x是一个DPN块的输入
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        x = self.shortcut(x)  # residual learning
        d = self.out_planes
        output_add_part = out[:, :d, :, :, :]
        output_dense_part = out[:, d:, :, :, :]
        # SE
        if self.attention is not None:
            output_add_part = self.attention(output_add_part)
        # [x[:d]+F(x)[:d],x[d:],F(x)[d:]]
        out = torch.cat([x[:, :d, :, :, :] + output_add_part, x[:, d:, :, :, :], output_dense_part], 1)
        out = F.relu(out)  # y
        return out
