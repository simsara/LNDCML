import os

import pynvml as pynvml

from utils.log import get_logger

log = get_logger(__name__)


def get_free_ratio(idx):
    handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
    use = pynvml.nvmlDeviceGetUtilizationRates(handle)
    ratio = 0.5 * (float(use.gpu + float(use.memory)))
    return ratio


def get_free_ids():
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()
    available = []
    for i in range(device_count):
        if get_free_ratio(i) < 70:
            available.append(i)
    gpus = []
    for g in available:
        gpus.append(g)
    return gpus


def set_gpu(gpuinput):
    free_ids = get_free_ids()
    if gpuinput == 'all':
        gpus = free_ids
    else:
        gpus = gpuinput
        if gpuinput not in gpus:
            raise ValueError('gpu %s is being used' % gpuinput)
    log.info('using gpu %s' % gpus)
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(i) for i in gpus)
    return len(gpus)


if __name__ == '__main__':
    log.info(set_gpu('all'))
