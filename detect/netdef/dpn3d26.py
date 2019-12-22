'''Dual Path Networks in PyTorch.'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from detect.netdef.loss import Loss
from detect.netdef.pbb import GetPBB


config = {}
config['anchors'] = [5., 10., 20.] #[ 10.0, 30.0, 60.]
config['chanel'] = 1
config['crop_size'] = [96, 96, 96]
config['stride'] = 4
config['max_stride'] = 16
config['num_neg'] = 800
config['th_neg'] = 0.02
config['th_pos_train'] = 0.5
config['th_pos_val'] = 1
config['num_hard'] = 2
config['bound_size'] = 12
config['reso'] = 1
config['sizelim'] = 2.5 #3 #6. #mm
config['sizelim2'] = 10 #30
config['sizelim3'] = 20 #40
config['aug_scale'] = True
config['r_rand_crop'] = 0.3
config['pad_value'] = 170
config['augtype'] = {'flip':True,'swap':False,'scale':True,'rotate':False}
config['side_len'] = 144
config['margin'] = 32
#config['augtype'] = {'flip':True,'swap':False,'scale':True,'rotate':False}
config['blacklist'] = ['868b024d9fa388b7ddab12ec1c06af38',
                       '990fbe3f0a1b53878669967b9afd1441',
                       'adc3bbc63d40f8761c59be10f1e504c3']
debug = False #True# #True

class Bottleneck(nn.Module):
    def __init__(self, last_planes, in_planes, out_planes, dense_depth, stride, first_layer):
        super(Bottleneck, self).__init__()
        self.out_planes = out_planes
        self.dense_depth = dense_depth
        self.last_planes = last_planes
        self.in_planes = in_planes

        self.conv1 = nn.Conv3d(last_planes, in_planes, kernel_size=1, bias=False) # 24 24
        self.bn1 = nn.BatchNorm3d(in_planes)
        # group = 8，将原输入分为8组，每组channel重复用out_channels/8次
        # groups决定了将原输入分为几组，而每组channel重用几次，由out_channels/groups计算得到
        self.conv2 = nn.Conv3d(in_planes, in_planes, kernel_size=3, stride=stride, padding=1, groups=8, bias=False)
        self.bn2 = nn.BatchNorm3d(in_planes)
        #
        self.conv3 = nn.Conv3d(in_planes, out_planes+dense_depth, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(out_planes+dense_depth)

        self.shortcut = nn.Sequential()
        if first_layer:
            self.shortcut = nn.Sequential(
                nn.Conv3d(last_planes, out_planes+dense_depth, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_planes+dense_depth)
            )

    # 一个DPN模块
    def forward(self, x):
        """
        y = G([x[:d]+F(x)[:d],x[d:],F(x)[d:]])
        y是一个DPN块的最后特征
        G是relu激活函数
        F是卷积层函数
        x是一个DPN块的输入
        """
        if debug: print('bottleneck_0', x.size(), self.last_planes, self.in_planes, 1)
        out = F.relu(self.bn1(self.conv1(x)))
        if debug: print('bottleneck_1', out.size(), self.in_planes, self.in_planes, 3)
        out = F.relu(self.bn2(self.conv2(out)))
        if debug: print('bottleneck_2', out.size(), self.in_planes, self.out_planes+self.dense_depth, 1)
        out = self.bn3(self.conv3(out))
        if debug: print('bottleneck_3', out.size())
        x = self.shortcut(x) # residual learning
        d = self.out_planes
        if debug: print('bottleneck_4', x.size(), self.last_planes, self.out_planes+self.dense_depth, d)
        out = torch.cat([x[:,:d,:,:]+out[:,:d,:,:], x[:,d:,:,:], out[:,d:,:,:]], 1)  # [x[:d]+F(x)[:d],x[d:],F(x)[d:]]
        if debug: print('bottleneck_5', out.size())
        out = F.relu(out)  # y
        return out

class DPN(nn.Module):
    def __init__(self, cfg):
        super(DPN, self).__init__()

        #得到参数
        in_planes, out_planes = cfg['in_planes'], cfg['out_planes']
        num_blocks, dense_depth = cfg['num_blocks'], cfg['dense_depth']

        # [96*96*96*1] [conv 24 3*3*3] [96*96*96*24]
        self.conv1 = nn.Conv3d(1, 24, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(24)
        self.last_planes = 24

        # 2*4个DPN blocks
        self.layer1 = self._make_layer(in_planes[0], out_planes[0], num_blocks[0], dense_depth[0], stride=2)#stride=1
        self.layer2 = self._make_layer(in_planes[1], out_planes[1], num_blocks[1], dense_depth[1], stride=2)
        self.layer3 = self._make_layer(in_planes[2], out_planes[2], num_blocks[2], dense_depth[2], stride=2)
        self.layer4 = self._make_layer(in_planes[3], out_planes[3], num_blocks[3], dense_depth[3], stride=2)

        self.last_planes = 216  # 本来self.last_planess是120的，不符合网络设计了，于是自己指定为96+120

        # 2个DPN
        self.layer5 = self._make_layer(128, 128, num_blocks[2], dense_depth[2], stride=1)
        self.last_planes = 224+3

        # 2个DPN
        self.layer6 = self._make_layer(224, 224, num_blocks[1], dense_depth[1], stride=1)

        # Linear(in_features=120, out_features=2)，定义一个linear层 TODO 好像没用到？
        self.linear = nn.Linear(out_planes[3]+(num_blocks[3]+1)*dense_depth[3], 2)#10)

        self.last_planes = 120

        # 第一个反卷积
        self.path1 = nn.Sequential(
            nn.ConvTranspose3d(self.last_planes, self.last_planes, kernel_size = 2, stride = 2),
            nn.BatchNorm3d(self.last_planes),
            nn.ReLU(inplace = True))

        self.last_planes = 152
        # 第二个反卷积
        self.path2 = nn.Sequential(
            nn.ConvTranspose3d(self.last_planes, self.last_planes, kernel_size = 2, stride = 2),
            nn.BatchNorm3d(self.last_planes),
            nn.ReLU(inplace = True))

        self.drop = nn.Dropout3d(p = 0.5, inplace = False)
        self.output = nn.Sequential(nn.Conv3d(248, 64, kernel_size = 1),
                                    nn.ReLU(),
                                    #nn.Dropout3d(p = 0.3),
                                   nn.Conv3d(64, 5 * len(config['anchors']), kernel_size = 1))

    # 两个DPN模块
    def _make_layer(self, in_planes, out_planes, num_blocks, dense_depth, stride):
        strides = [stride] + [1]*(num_blocks-1)  # [4,1]=[4]+[1]*(2-1)
        layers = []
        for i,stride in enumerate(strides):  # i=0,stride=4 i=1,stride=1
            #if debug: print(i, self.last_planes, in_planes, out_planes, dense_depth)
            layers.append(Bottleneck(self.last_planes, in_planes, out_planes, dense_depth, stride, i==0))
            self.last_planes = out_planes + (i+2) * dense_depth  # 每经过两个DPN模块，last_planes增加24
        return nn.Sequential(*layers)  # 将每一个模块按顺序送入到nn.Sequential中

    def forward(self, x, coord):
        if debug: print('0', x.size(), 64)  # 24*24*24*8
        out0 = F.relu(self.bn1(self.conv1(x)))  # [24 3*3*3] 96*96*96*24
        if debug: print('1', out0.size())
        out1 = self.layer1(out0)  # [2DPN] 48*48*48*48
        if debug: print('2', out1.size())
        out2 = self.layer2(out1)  # [2DPN] 24*24*24*72
        if debug: print('3', out2.size())
        out3 = self.layer3(out2)  # [2DPN] 12*12*12*96
        if debug: print('4', out3.size())
        out4 = self.layer4(out3)  # [2DPN] 6*6*6*120
        if debug: print('5', out4.size())

        out5 = self.path1(out4)  # [Deconv] 12*12*12*120
        if debug: print('6', out5.size(), torch.cat((out3, out5), 1).size())
        out6 = self.layer5(torch.cat((out3, out5), 1))  # [2DPN] 12*12*12*(96+120) -> 12*12*12*(128+24=152)
        if debug: print('7', out6.size())
        out7 = self.path2(out6)  # [Deconv] 24*24*24*152
        if debug: print('8', out7.size(), torch.cat((out2, out7), 1).size())  # 24*24*24*152 24*24*24*224
        out8 = self.layer6(torch.cat((out2, out7, coord), 1))  # [2DPN] 24*24*24*(224+3) -> 24*24*24*(224+24=248)
        if debug: print('9', out8.size())
        comb2 = self.drop(out8)  # 24*24*24*248
        out = self.output(comb2) # 24*24*24*(5*3)
        if debug: print('10', out.size())
        size = out.size()
        out = out.view(out.size(0), out.size(1), -1)  # 展开
        if debug: print('11', out.size())  # 13824*(5*3)
        #out = out.transpose(1, 4).transpose(1, 2).transpose(2, 3).contiguous()
        # 交换维度后再展开 num * 24 * 24 * 24 * 3 * 5
        out = out.transpose(1, 2).contiguous().view(size[0], size[2], size[3], size[4], len(config['anchors']), 5)
        if debug: print('12', out.size())
        return out#, out_1

def DPN92_3D():
    cfg = {
        'in_planes': (24,48,72,96),  # (96,192,384,768) feature map的个数
        'out_planes': (24,48,72,96),
        'num_blocks': (2,2,2,2),  # 2DPN为一个blocks
        'dense_depth': (8,8,8,8)  # d=8
    }
    return DPN(cfg)

def get_model():
    net = DPN92_3D()
    net = torch.nn.DataParallel(net.cuda())  # 使用多个GPU进行训练
    loss = Loss(config['num_hard'])  # 使用hard mining策略
    get_pbb = GetPBB(config)  # probability of bounding box
    return config, net, loss, get_pbb

if __name__ == '__main__':
    test_net = DPN92_3D()
    dummy = torch.randn(1, 1, 96, 96, 96)
    coord = torch.randn(1, 3, 24, 24, 24)
    test_net(dummy, coord)
