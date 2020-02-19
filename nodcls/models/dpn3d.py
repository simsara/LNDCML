'''Dual Path Networks in PyTorch.'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class Bottleneck(nn.Module):
    def __init__(self, last_planes, in_planes, out_planes, dense_depth, stride, first_layer):
        super(Bottleneck, self).__init__()
        self.out_planes = out_planes
        self.dense_depth = dense_depth
        self.last_planes = last_planes
        self.in_planes = in_planes

        self.conv1 = nn.Conv3d(last_planes, in_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(in_planes)
        self.conv2 = nn.Conv3d(in_planes, in_planes, kernel_size=3, stride=stride, padding=1, groups=32, bias=False)
        self.bn2 = nn.BatchNorm3d(in_planes)
        self.conv3 = nn.Conv3d(in_planes, out_planes + dense_depth, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(out_planes + dense_depth)

        self.shortcut = nn.Sequential()
        if first_layer:
            self.shortcut = nn.Sequential(
                nn.Conv3d(last_planes, out_planes + dense_depth, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_planes + dense_depth)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        x = self.shortcut(x)
        d = self.out_planes
        out = torch.cat([x[:, :d, :, :] + out[:, :d, :, :], x[:, d:, :, :], out[:, d:, :, :]], 1)
        out = F.relu(out)
        return out


class DPN(nn.Module):
    def __init__(self, cfg):
        super(DPN, self).__init__()
        in_planes, out_planes = cfg['in_planes'], cfg['out_planes']
        num_blocks, dense_depth = cfg['num_blocks'], cfg['dense_depth']

        self.conv1 = nn.Conv3d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.last_planes = 64

        self.layer1 = self._make_layer(in_planes[0], out_planes[0], num_blocks[0], dense_depth[0], stride=1)
        self.layer2 = self._make_layer(in_planes[1], out_planes[1], num_blocks[1], dense_depth[1], stride=2)
        self.layer3 = self._make_layer(in_planes[2], out_planes[2], num_blocks[2], dense_depth[2], stride=2)
        self.layer4 = self._make_layer(in_planes[3], out_planes[3], num_blocks[3], dense_depth[3], stride=2)
        self.linear = nn.Linear(out_planes[3] + (num_blocks[3] + 1) * dense_depth[3], 2)  # 10)

    def _make_layer(self, in_planes, out_planes, num_blocks, dense_depth, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i, stride in enumerate(strides):
            layers.append(Bottleneck(self.last_planes, in_planes, out_planes, dense_depth, stride, i == 0))
            self.last_planes = out_planes + (i + 2) * dense_depth
            # print '_make_layer', i, layers[-1].size()
        return nn.Sequential(*layers)

    def forward(self, x):  # 1 * 32
        out = F.relu(self.bn1(self.conv1(x)))  # 32 * 64
        out = self.layer1(out)  # 32 * 320
        out = self.layer2(out)  # 16 * 672
        out = self.layer3(out)  # 8 * 1528
        out = self.layer4(out)  # 4 * 2560
        out = F.avg_pool3d(out, 4)  # 1 * 2560
        out_1 = out.view(out.size(0), -1)
        out = self.linear(out_1)  # 1 * 2
        return out, out_1


def DPN92_3D():
    cfg = {
        'in_planes': (96, 192, 384, 768),
        'out_planes': (256, 512, 1024, 2048),
        'num_blocks': (3, 4, 20, 3),
        'dense_depth': (16, 32, 24, 128)
    }
    return DPN(cfg)


if __name__ == '__main__':
    test_net = DPN92_3D()
    dummy = torch.randn(1, 1, 32, 32, 32)
    test_out, test_out_1 = test_net(dummy)
    print(test_out)
