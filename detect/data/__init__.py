import collections

import numpy as np
import torch


def collate(batch):
    if torch.is_tensor(batch[0]):  # batch_size = 1
        return [b.unsqueeze(0) for b in batch]  # 1 * split
    elif isinstance(batch[0], np.ndarray):
        return batch
    elif isinstance(batch[0], int):
        return torch.LongTensor(batch)
    elif isinstance(batch[0], collections.Iterable):
        transposed = zip(*batch)
        return [collate(samples) for samples in transposed]
