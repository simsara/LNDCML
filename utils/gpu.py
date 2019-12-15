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


def set_gpu(input: str):
    free_ids = get_free_ids()
    to_use = []
    if input == 'all':
        to_use = free_ids
    else:
        input_arr = input.split(',')
        for i in input_arr:
            if i in free_ids:
                to_use.append(i)
            else:
                log.warn('gpu %s is being used' % i)

    if len(to_use) == 0:
        raise ValueError('No available gpu')
    log.info('using gpu %s' % to_use)
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(i) for i in to_use)
    return len(to_use)


if __name__ == '__main__':
    log.info(set_gpu('all'))
