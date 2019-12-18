import torch
from torch import nn

from detect.netdef.loss import Loss
from detect.netdef.pbb import GetPBB
from detect.netdef.res_block import ResidualBlock

config = {}
config['anchors'] = [5., 10., 20.]
config['channel'] = 1
config['crop_size'] = [96, 96, 96]
config['stride'] = 4
config['max_stride'] = 16
config['num_neg'] = 800
config['th_neg'] = 0.02
config['th_pos_train'] = 0.5
config['th_pos_val'] = 1
config['num_hard'] = 2  # 负anchor数量
config['bound_size'] = 12
config['reso'] = 1
config['sizelim'] = 2.5  # 3 #6. #mm
config['sizelim2'] = 10  # 30
config['sizelim3'] = 20  # 40
config['aug_scale'] = True
config['r_rand_crop'] = 0.3
config['pad_value'] = 170
config['augtype'] = {'flip': True, 'swap': False, 'scale': True, 'rotate': False}
config['side_len'] = 144
config['margin'] = 32

config['augtype'] = {'flip': True, 'swap': False, 'scale': True, 'rotate': False}
config['blacklist'] = ['868b024d9fa388b7ddab12ec1c06af38', '990fbe3f0a1b53878669967b9afd1441',
                       'adc3bbc63d40f8761c59be10f1e504c3']


# 这个是论文中用来对比的Res18网络
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # The first few layers consumes the most memory, so use simple convolution to save memory.
        # Call these layers preBlock, i.e., before the residual blocks of later layers.
        # 最开始的块，经过两层卷积，size 96 channel 24
        self.preBlock = nn.Sequential(
            nn.Conv3d(1, 24, kernel_size=3, padding=1),
            nn.BatchNorm3d(24),
            nn.ReLU(inplace=True),
            nn.Conv3d(24, 24, kernel_size=3, padding=1),
            nn.BatchNorm3d(24),
            nn.ReLU(inplace=True))

        # downsampling
        # 手写清晰点
        self.forw1 = nn.Sequential(
            ResidualBlock(24, 32),
            ResidualBlock(32, 32)
        )
        self.forw2 = nn.Sequential(
            ResidualBlock(32, 64),
            ResidualBlock(64, 64)
        )
        self.forw3 = nn.Sequential(
            ResidualBlock(64, 64),
            ResidualBlock(64, 64),
            ResidualBlock(64, 64)
        )
        self.forw4 = nn.Sequential(
            ResidualBlock(64, 64),
            ResidualBlock(64, 64),
            ResidualBlock(64, 64)
        )

        self.back2 = nn.Sequential(
            ResidualBlock(131, 128),
            ResidualBlock(128, 128),
            ResidualBlock(128, 128)
        )
        self.back3 = nn.Sequential(
            ResidualBlock(128, 64),
            ResidualBlock(64, 64),
            ResidualBlock(64, 64)
        )

        # Res Block 要用的max pool
        self.maxpool1 = nn.MaxPool3d(kernel_size=2, stride=2, return_indices=True)
        self.maxpool2 = nn.MaxPool3d(kernel_size=2, stride=2, return_indices=True)
        self.maxpool3 = nn.MaxPool3d(kernel_size=2, stride=2, return_indices=True)
        self.maxpool4 = nn.MaxPool3d(kernel_size=2, stride=2, return_indices=True)

        # upsampling
        # out = (in - 1) * stride + out_padding - 2 * padding + kernel_size
        # kernel_size, stride 和 padding 采用跟池化一样参数
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose3d(64, 64, kernel_size=2, stride=2),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True))
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose3d(64, 64, kernel_size=2, stride=2),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True))
        self.drop = nn.Dropout3d(p=0.5, inplace=False)
        self.output = nn.Sequential(nn.Conv3d(128, 64, kernel_size=1),
                                    nn.ReLU(),
                                    nn.Conv3d(64, 5 * len(config['anchors']), kernel_size=1))

    def forward(self, x, coord):  # coord ???
        out = self.preBlock(x)  # size 96 channel 24

        res_block_1_pool, indices1 = self.maxpool1(out)  # size 48 channel 24
        out1 = self.forw1(res_block_1_pool)  # size 48 channel 32

        res_block_2_pool, indices2 = self.maxpool2(out1)  # size 24 channel 32
        out2 = self.forw2(res_block_2_pool)  # size 24 channel 64

        res_block_3_pool, indices3 = self.maxpool3(out2)  # size 12 channel 64
        out3 = self.forw3(res_block_3_pool)  # size 12 channel 64

        res_block_4_pool, indices3 = self.maxpool4(out3)  # size 6 channel 64
        out4 = self.forw4(res_block_4_pool)  # size 6 channel 64

        rev3 = self.deconv1(out4)  # size 12 channel 64
        concat3 = torch.cat((rev3, out3), 1)  # size 12 channel 64+64
        comb3 = self.back3(concat3)  # size 12 channel 64

        rev2 = self.deconv2(comb3)  # size 24 channel 64
        concat2 = torch.cat((rev2, out2, coord), 1)  # size 24 channel 128
        comb2 = self.back2(concat2)  # size 24 channel 128

        drop2 = self.drop(comb2)  # size 24 channel 128
        out = self.output(drop2)  # size 24 channel 3 * 5
        size = out.size()
        out = out.view(out.size(0), out.size(1), -1)
        # 交换维度后再展开 num * 24 * 24 * 24 * 3 * 5
        out = out.transpose(1, 2).contiguous() \
            .view(size[0], size[2], size[3], size[4], len(config['anchors']), 5)
        return out


if __name__ == '__main__':
    test_net = Net()
    dummy = torch.randn(1, 1, 96, 96, 96)
    coord = torch.randn(1, 3, 24, 24, 24)
    test_net(dummy, coord)


def get_model():
    net = Net()
    para_net = torch.nn.DataParallel(net.cuda())
    loss = Loss(config['num_hard'])
    get_pbb = GetPBB(config)
    return config, para_net, loss, get_pbb
