import math
import os
import time

import numpy as np
import torch
from scipy.ndimage import rotate
from torch.utils.data import Dataset

from detect.data.crop import Crop
from detect.data.label_mapping import LabelMapping
from utils.log import get_logger

log = get_logger(__name__)


class DataBowl3Detector(Dataset):
    def __init__(self, data_dir, sample_prefix_list, config, phase='train', split_combo=None):
        assert (phase == 'train' or phase == 'val' or phase == 'test')
        self.phase = phase
        self.max_stride = config['max_stride']
        self.stride = config['stride']
        sizelim = config['sizelim'] / config['reso']
        sizelim2 = config['sizelim2'] / config['reso']
        sizelim3 = config['sizelim3'] / config['reso']
        self.blacklist = config['blacklist']
        self.isScale = config['aug_scale']
        self.r_rand = config['r_rand_crop']
        self.augtype = config['augtype']
        self.pad_value = config['pad_value']
        self.split_combo = split_combo

        self.img_file_names = [os.path.join(data_dir, '%s_clean.npy' % idx) for idx in sample_prefix_list]  # 处理完的图像
        labels = []  # 按顺序的标签集
        log.info('Total file size: %s' % len(sample_prefix_list))
        for idx in sample_prefix_list:
            label_file_name = os.path.join(data_dir, '%s_label.npy' % idx)  # 结节label
            label_data = np.load(label_file_name, allow_pickle=True)
            if np.all(label_data == 0):  # 没有结节
                label_data = np.array([])
            labels.append(label_data)

        self.sample_bboxes = labels
        if self.phase != 'test':
            self.bboxes = []
            for i, l in enumerate(labels):
                if len(l) == 0:
                    continue
                for t in l:
                    label_with_no = np.concatenate([[i], t])
                    if t[3] > sizelim:
                        self.bboxes.append([label_with_no])
                    if t[3] > sizelim2:
                        self.bboxes += [[label_with_no]] * 2
                    if t[3] > sizelim3:
                        self.bboxes += [[label_with_no]] * 4
            self.bboxes = np.concatenate(self.bboxes, axis=0)  # TODO 多放这么多有什么用

        self.crop = Crop(config)
        self.label_mapping = LabelMapping(config, self.phase)

    def __getitem__(self, idx, split=None):
        t = time.time()
        np.random.seed(int(str(t % 1)[2:7]))  # seed according to time

        is_random_img = False
        is_random = False
        if self.phase != 'test':
            if idx >= len(self.bboxes):  # TODO 随机
                is_random = True
                idx = idx % len(self.bboxes)
                is_random_img = np.random.randint(2)

        if self.phase != 'test':
            if not is_random_img:
                bbox = self.bboxes[idx]  # idx, x, y, z, d
                img_file_name = self.img_file_names[int(bbox[0])]
                img_data = np.load(img_file_name)
                bboxes = self.sample_bboxes[int(bbox[0])]
                is_scale = self.augtype['scale'] and (self.phase == 'train')
                sample, target, bboxes, coord = self.crop(img_data, bbox[1:], bboxes, is_scale, is_random)
                if self.phase == 'train' and not is_random:  # 随机增强
                    sample, target, bboxes, coord = augment(sample, target, bboxes, coord,
                                                            ifflip=self.augtype['flip'],
                                                            ifrotate=self.augtype['rotate'],
                                                            ifswap=self.augtype['swap'])
            else:
                rand_idx = np.random.randint(len(self.img_file_names))
                img_file_name = self.img_file_names[rand_idx]
                img_data = np.load(img_file_name)
                bboxes = self.sample_bboxes[rand_idx]
                sample, target, bboxes, coord = self.crop(img_data, [], bboxes, is_scale=False, is_rand=True)
            label = self.label_mapping(sample.shape[1:], target, bboxes, img_file_name)
            sample = (sample.astype(np.float32) - 128) / 128
            return torch.from_numpy(sample), torch.from_numpy(label), coord
        else:
            img_data = np.load(self.img_file_names[idx])
            bboxes = self.sample_bboxes[idx]
            nz, nh, nw = img_data.shape[1:]
            pz = int(np.ceil(float(nz) / self.stride)) * self.stride
            ph = int(np.ceil(float(nh) / self.stride)) * self.stride
            pw = int(np.ceil(float(nw) / self.stride)) * self.stride
            img_data = np.pad(img_data, [[0, 0], [0, pz - nz], [0, ph - nh], [0, pw - nw]], 'constant',
                              constant_values=self.pad_value)

            xx, yy, zz = np.meshgrid(np.linspace(-0.5, 0.5, img_data.shape[1] / self.stride),
                                     np.linspace(-0.5, 0.5, img_data.shape[2] / self.stride),
                                     np.linspace(-0.5, 0.5, img_data.shape[3] / self.stride), indexing='ij')
            coord = np.concatenate([xx[np.newaxis, ...], yy[np.newaxis, ...], zz[np.newaxis, :]], 0).astype('float32')
            img_data, nzhw = self.split_combo.split(img_data)

            coord2, nzhw2 = self.split_combo.split(coord,  # python3 python2
                                                   side_len=int(self.split_combo.side_len / self.stride),
                                                   max_stride=int(self.split_combo.max_stride / self.stride),
                                                   margin=int(self.split_combo.margin / self.stride))
            assert np.all(nzhw == nzhw2)
            img_data = (img_data.astype(np.float32) - 128) / 128
            return torch.from_numpy(img_data), bboxes, torch.from_numpy(coord2), np.array(nzhw)

    def __len__(self):
        if self.phase == 'train':
            return math.floor(len(self.bboxes) / (1 - self.r_rand))  # 随机
        elif self.phase == 'val':
            return len(self.bboxes)
        else:
            return len(self.sample_bboxes)


def augment(sample, target, bboxes, coord, ifflip=True, ifrotate=True, ifswap=True):
    #                     angle1 = np.random.rand()*180
    if ifrotate:
        validrot = False
        counter = 0
        while not validrot:
            newtarget = np.copy(target)
            angle1 = np.random.rand() * 180
            size = np.array(sample.shape[2:4]).astype('float')
            rotmat = np.array([[np.cos(angle1 / 180 * np.pi), -np.sin(angle1 / 180 * np.pi)],
                               [np.sin(angle1 / 180 * np.pi), np.cos(angle1 / 180 * np.pi)]])
            newtarget[1:3] = np.dot(rotmat, target[1:3] - size / 2) + size / 2
            if np.all(newtarget[:3] > target[3]) and np.all(newtarget[:3] < np.array(sample.shape[1:4]) - newtarget[3]):
                validrot = True
                target = newtarget
                sample = rotate(sample, angle1, axes=(2, 3), reshape=False)
                coord = rotate(coord, angle1, axes=(2, 3), reshape=False)
                for box in bboxes:
                    box[1:3] = np.dot(rotmat, box[1:3] - size / 2) + size / 2
            else:
                counter += 1
                if counter == 3:
                    break
    if ifswap:
        if sample.shape[1] == sample.shape[2] and sample.shape[1] == sample.shape[3]:
            axisorder = np.random.permutation(3)
            sample = np.transpose(sample, np.concatenate([[0], axisorder + 1]))
            coord = np.transpose(coord, np.concatenate([[0], axisorder + 1]))
            target[:3] = target[:3][axisorder]
            bboxes[:, :3] = bboxes[:, :3][:, axisorder]

    if ifflip:
        #         flipid = np.array([np.random.randint(2),np.random.randint(2),np.random.randint(2)])*2-1
        flipid = np.array([1, np.random.randint(2), np.random.randint(2)]) * 2 - 1
        sample = np.ascontiguousarray(sample[:, ::flipid[0], ::flipid[1], ::flipid[2]])
        coord = np.ascontiguousarray(coord[:, ::flipid[0], ::flipid[1], ::flipid[2]])
        for ax in range(3):
            if flipid[ax] == -1:
                target[ax] = np.array(sample.shape[ax + 1]) - target[ax]
                bboxes[:, ax] = np.array(sample.shape[ax + 1]) - bboxes[:, ax]
    return sample, target, bboxes, coord
