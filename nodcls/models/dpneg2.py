import torch
from torch import nn
from torch.nn import functional as F

from nodcls.models import get_common_config
from nodcls.models.bn import Bottleneck


class EmbeddedGaussian(nn.Module):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(EmbeddedGaussian, self).__init__()
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        conv_nd = nn.Conv3d
        max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
        bn = nn.BatchNorm3d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)
        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x):
        '''
        :param x: (b, c, t, h, w)
        :return:
        '''

        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z


class DPN_EG(nn.Module):
    def __init__(self, cfg):
        super(DPN_EG, self).__init__()
        in_planes, out_planes = cfg['in_planes'], cfg['out_planes']
        num_blocks, dense_depth = cfg['num_blocks'], cfg['dense_depth']

        self.conv1 = nn.Conv3d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.last_planes = 64

        self.layer1 = self._make_layer(in_planes[0], out_planes[0], num_blocks[0], dense_depth[0], stride=1)
        self.eg1 = EmbeddedGaussian(self.last_planes, self.last_planes // 8)
        self.layer2 = self._make_layer(in_planes[1], out_planes[1], num_blocks[1], dense_depth[1], stride=2)
        self.eg2 = EmbeddedGaussian(self.last_planes, self.last_planes // 8)
        self.layer3 = self._make_layer(in_planes[2], out_planes[2], num_blocks[2], dense_depth[2], stride=2)
        self.eg3 = EmbeddedGaussian(self.last_planes, self.last_planes // 8)
        self.layer4 = self._make_layer(in_planes[3], out_planes[3], num_blocks[3], dense_depth[3], stride=2)
        self.linear = nn.Linear(out_planes[3] + (num_blocks[3] + 1) * dense_depth[3], 2)  # 10)
        self.drop = nn.Dropout3d(p=0.5, inplace=False)

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
        out = self.eg1(out)
        out = self.layer2(out)  # 16 * 672
        out = self.eg2(out)
        out = self.layer3(out)  # 8 * 1528
        out = self.eg3(out)  # 8 * 1528
        out = self.layer4(out)  # 4 * 2560
        out = self.drop(out)
        out = F.avg_pool3d(out, 4)  # 1 * 2560
        out_1 = out.view(out.size(0), -1)
        out = self.linear(out_1)  # 1 * 2
        return out, out_1


def get_model():
    return DPN_EG(get_common_config())


if __name__ == '__main__':
    test_net = get_model()
    dummy = torch.randn(1, 1, 32, 32, 32)
    test_out, test_out_1 = test_net(dummy)
    print(test_out)
