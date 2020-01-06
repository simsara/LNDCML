import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(MLP, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=bias),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(x)


class ChannelGate(nn.Module):
    def __init__(self, channel, reduction=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.pool_types = pool_types
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_mlp = MLP(channel, reduction)
        self.avg_mlp = MLP(channel, reduction)

    def forward(self, x):
        b, c, _, _, _ = x.size()
        channel_att_sum = None

        for pool_type in self.pool_types:
            channel_att_raw = None
            if pool_type == 'avg':
                pool = self.avg_pool(x).view(b, c)
                channel_att_raw = self.avg_mlp(pool)
            elif pool_type == 'max':
                pool = self.max_pool(x).view(b, c)
                channel_att_raw = self.max_mlp(pool)
            if channel_att_raw is None:
                continue
            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw
        scale = channel_att_sum.view(b, c, 1, 1, 1).expand_as(x)
        return x * scale


class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        self.spatial = nn.Sequential(
            nn.Conv3d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=(7 - 1) // 2),
            nn.Sigmoid()
        )

    def forward(self, x):
        max = torch.max(x, 1)[0]
        mean = torch.mean(x, 1)
        x_compress = torch.cat((max.unsqueeze(1), mean.unsqueeze(1)), dim=1)
        scale = self.spatial(x_compress)
        return x * scale


class CBAM(nn.Module):
    def __init__(self, channel, reduction=16, pool_types=['avg', 'max'], sequence=1):
        super(CBAM, self).__init__()
        self.seq = sequence
        self.ChannelGate = ChannelGate(channel, reduction, pool_types)
        self.SpatialGate = SpatialGate()
        self.conv = nn.Conv3d(channel, channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm3d(channel)

    def forward(self, x):
        if self.seq == 1:
            return self.SpatialGate(self.ChannelGate(x))
        elif self.seq == 2:
            return self.ChannelGate(self.SpatialGate(x))
        else:
            c_out = self.ChannelGate(x)
            s_out = self.SpatialGate(x)
            out = c_out + s_out
            return F.relu(self.bn(self.conv(out)))
