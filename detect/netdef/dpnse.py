'''Dual Path Networks in PyTorch.'''
import torch

import detect.netdef.dpn_basic as dpn
from detect.netdef.se import SE3


def get_model():
    return dpn.get_model(SE3, reduction=2)


if __name__ == '__main__':
    _, test_net, _, _ = get_model()
    dummy = torch.randn(1, 1, 96, 96, 96)
    coord = torch.randn(1, 3, 24, 24, 24)
    test_net(dummy, coord)
