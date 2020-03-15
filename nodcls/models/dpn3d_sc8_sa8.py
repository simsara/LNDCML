'''Dual Path Networks in PyTorch.'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from nodcls.models import get_common_config


class Self_Attn3D(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim, query_channel, key_channel):
        super(Self_Attn3D, self).__init__()
        self.chanel_in = in_dim
        self.query_conv = nn.Conv3d(in_channels=in_dim, out_channels=query_channel, kernel_size=1)
        self.key_conv = nn.Conv3d(in_channels=in_dim, out_channels=key_channel, kernel_size=1)
        self.value_conv = nn.Conv3d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, w, h, z = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, w * h * z).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, w * h * z)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, w * h * z)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, w, h, z)

        out = self.gamma * out + x
        return out


class AttentionLayer(nn.Module):
    def __init__(self, channel, reduction=64, multiply=True):
        super(AttentionLayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )
        self.multiply = multiply

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        if self.multiply == True:
            return x * y
        else:
            return y


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
        # print 'bottleneck_0', x.size(), self.last_planes, self.in_planes, 1
        out = F.relu(self.bn1(self.conv1(x)))
        # print 'bottleneck_1', out.size(), self.in_planes, self.in_planes, 3
        out = F.relu(self.bn2(self.conv2(out)))
        # print 'bottleneck_2', out.size(), self.in_planes, self.out_planes+self.dense_depth, 1
        out = self.bn3(self.conv3(out))
        # print 'bottleneck_3', out.size()
        x = self.shortcut(x)
        d = self.out_planes
        # print 'bottleneck_4', x.size(), self.last_planes, self.out_planes+self.dense_depth, d
        out = torch.cat([x[:, :d, :, :] + out[:, :d, :, :], x[:, d:, :, :], out[:, d:, :, :]], 1)
        # print 'bottleneck_5', out.size()
        out = F.relu(out)
        return out


class DPN(nn.Module):
    def __init__(self, cfg):
        super(DPN, self).__init__()
        in_planes, out_planes = cfg['in_planes'], cfg['out_planes']
        num_blocks, dense_depth = cfg['num_blocks'], cfg['dense_depth']

        # self.in_planes = in_planes
        # self.out_planes = out_planes
        # self.num_blocks = num_blocks
        # self.dense_depth = dense_depth

        self.conv1 = nn.Conv3d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.last_planes = 64
        self.layer1 = self._make_layer(in_planes[0], out_planes[0], num_blocks[0], dense_depth[0], stride=1)
        self.layer2 = self._make_layer(in_planes[1], out_planes[1], num_blocks[1], dense_depth[1], stride=2)
        self.layer3 = self._make_layer(in_planes[2], out_planes[2], num_blocks[2], dense_depth[2], stride=2)
        self.layer4 = self._make_layer(in_planes[3], out_planes[3], num_blocks[3], dense_depth[3], stride=2)
        self.linear = nn.Linear(out_planes[3] + (num_blocks[3] + 1) * dense_depth[3], 2)  # 10)

        self.sp_attention = nn.Sequential(
            nn.Conv3d(1528, 1528 / 8, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Tanh(),
            nn.Conv3d(1528 / 8, 1, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm3d(2560)
        )
        self.sa = Self_Attn3D(1528, 1528 // 8, 1528 // 8)
        self.sc_attention = AttentionLayer(channel=1528)

    def _make_layer(self, in_planes, out_planes, num_blocks, dense_depth, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i, stride in enumerate(strides):
            layers.append(Bottleneck(self.last_planes, in_planes, out_planes, dense_depth, stride, i == 0))
            self.last_planes = out_planes + (i + 2) * dense_depth
            # print '_make_layer', i, layers[-1].size()
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.sc_attention(out)
        out_2 = self.sa(out)
        out = self.layer4(out_2)
        out = F.avg_pool3d(out, 4)
        out_1 = out.view(out.size(0), -1)
        out = self.linear(out_1)
        return out, out_2


def get_model():
    return DPN(get_common_config())
