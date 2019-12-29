'''Dual Path Networks in PyTorch.'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from detect.netdef import get_common_config
from detect.netdef.res_bn import Bottleneck
from detect.netdef.loss import Loss
from detect.netdef.pbb import GetPBB

config = get_common_config()


class DPN(nn.Module):
    def __init__(self, cfg, attention=None):
        super(DPN, self).__init__()
        self.attention = attention

        # 得到参数
        in_planes, out_planes = cfg['in_planes'], cfg['out_planes']
        num_blocks, dense_depth = cfg['num_blocks'], cfg['dense_depth']

        # [96*96*96*1] [conv 24 3*3*3] [96*96*96*24]
        self.conv1 = nn.Conv3d(1, 24, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(24)
        self.last_planes = 24

        # 2*4个DPN blocks
        self.layer1 = self._make_layer(in_planes[0], out_planes[0], num_blocks[0], dense_depth[0], stride=2)  # stride=1
        self.layer2 = self._make_layer(in_planes[1], out_planes[1], num_blocks[1], dense_depth[1], stride=2)
        self.layer3 = self._make_layer(in_planes[2], out_planes[2], num_blocks[2], dense_depth[2], stride=2)
        self.layer4 = self._make_layer(in_planes[3], out_planes[3], num_blocks[3], dense_depth[3], stride=2)

        self.last_planes = 216  # 本来self.last_planess是120的，不符合网络设计了，于是自己指定为96+120

        # 2个DPN
        self.layer5 = self._make_layer(128, 128, num_blocks[2], dense_depth[2], stride=1)
        self.last_planes = 224 + 3

        # 2个DPN
        self.layer6 = self._make_layer(224, 224, num_blocks[1], dense_depth[1], stride=1)

        # Linear(in_features=120, out_features=2)，定义一个linear层 TODO 好像没用到？
        self.linear = nn.Linear(out_planes[3] + (num_blocks[3] + 1) * dense_depth[3], 2)  # 10)

        self.last_planes = 120

        # 第一个反卷积
        self.path1 = nn.Sequential(
            nn.ConvTranspose3d(self.last_planes, self.last_planes, kernel_size=2, stride=2),
            nn.BatchNorm3d(self.last_planes),
            nn.ReLU(inplace=True))

        self.last_planes = 152
        # 第二个反卷积
        self.path2 = nn.Sequential(
            nn.ConvTranspose3d(self.last_planes, self.last_planes, kernel_size=2, stride=2),
            nn.BatchNorm3d(self.last_planes),
            nn.ReLU(inplace=True))

        self.drop = nn.Dropout3d(p=0.5, inplace=False)
        self.output = nn.Sequential(nn.Conv3d(248, 64, kernel_size=1),
                                    nn.ReLU(),
                                    # nn.Dropout3d(p = 0.3),
                                    nn.Conv3d(64, 5 * len(config['anchors']), kernel_size=1))

    # 两个DPN模块
    def _make_layer(self, in_planes, out_planes, num_blocks, dense_depth, stride):
        strides = [stride] + [1] * (num_blocks - 1)  # [4,1]=[4]+[1]*(2-1)
        layers = []
        for i, stride in enumerate(strides):  # i=0,stride=4 i=1,stride=1
            layers.append(
                Bottleneck(self.last_planes, in_planes, out_planes, dense_depth, stride, i == 0, self.attention))
            self.last_planes = out_planes + (i + 2) * dense_depth  # 每经过两个DPN模块，last_planes增加24
        return nn.Sequential(*layers)  # 将每一个模块按顺序送入到nn.Sequential中

    def forward(self, x, coord):
        out0 = F.relu(self.bn1(self.conv1(x)))  # [24 3*3*3] 96*96*96*24
        out1 = self.layer1(out0)  # [2DPN] 48*48*48*48
        out2 = self.layer2(out1)  # [2DPN] 24*24*24*72
        out3 = self.layer3(out2)  # [2DPN] 12*12*12*96
        out4 = self.layer4(out3)  # [2DPN] 6*6*6*120

        out5 = self.path1(out4)  # [Deconv] 12*12*12*120
        out6 = self.layer5(torch.cat((out3, out5), 1))  # [2DPN] 12*12*12*(96+120) -> 12*12*12*(128+24=152)
        out7 = self.path2(out6)  # [Deconv] 24*24*24*152
        out8 = self.layer6(torch.cat((out2, out7, coord), 1))  # [2DPN] 24*24*24*(224+3) -> 24*24*24*(224+24=248)
        comb2 = self.drop(out8)  # 24*24*24*248
        out = self.output(comb2)  # 24*24*24*(5*3)
        size = out.size()
        out = out.view(out.size(0), out.size(1), -1)  # 展开
        # out = out.transpose(1, 4).transpose(1, 2).transpose(2, 3).contiguous()
        # 交换维度后再展开 num * 24 * 24 * 24 * 3 * 5
        out = out.transpose(1, 2).contiguous().view(size[0], size[2], size[3], size[4], len(config['anchors']), 5)
        return out  # , out_1


def DPN92_3D(attention_module=None):
    cfg = {
        'in_planes': (24, 48, 72, 96),  # (96,192,384,768) feature map的个数
        'out_planes': (24, 48, 72, 96),
        'num_blocks': (2, 2, 2, 2),  # 2DPN为一个blocks
        'dense_depth': (8, 8, 8, 8)  # d=8
    }
    return DPN(cfg, attention_module)


def get_model(attention_module=None):
    net = DPN92_3D(attention_module)
    loss = Loss(config['num_hard'])  # 使用hard mining策略
    get_pbb = GetPBB(config)  # probability of bounding box
    return config, net, loss, get_pbb
