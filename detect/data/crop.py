import warnings

import numpy as np
from scipy.ndimage import zoom


class Crop(object):
    def __init__(self, config):
        self.crop_size = config['crop_size']
        self.bound_size = config['bound_size']
        self.stride = config['stride']
        self.pad_value = config['pad_value']

    def __call__(self, imgs, target, bboxes, is_scale=False, is_rand=False):
        if is_scale:
            diam_lim = [8., 120.]
            scale_lim = [0.75, 1.25]
            scale_range = [
                np.min([np.max([(diam_lim[0] / target[3]), scale_lim[0]]), 1]),
                np.max([np.min([(diam_lim[1] / target[3]), scale_lim[1]]), 1])
            ]  # 直径在规定范围内的缩放范围
            scale = np.random.rand() * (scale_range[1] - scale_range[0]) + scale_range[0]  # 随机缩放比例
            crop_size = (np.array(self.crop_size).astype('float') / scale).astype('int')  # 裁剪大小 d小crop大 d大crop小
        else:
            crop_size = self.crop_size
        bound_size = self.bound_size
        target = np.copy(target)  # 目标结节
        bboxes = np.copy(bboxes)  # 该CT所有结节

        start = []
        for i in range(3):  # x, y, z
            if not is_rand:  # 从结节里面剪
                r = target[3] / 2
                s = np.floor(target[i] - r) + 1 - bound_size
                e = np.ceil(target[i] + r) + 1 + bound_size - crop_size[i]  # 如果s<e，说明结节直接小，crop可以包括整个结节
            else:  # 0.3的几率在整片CT里面随机剪
                s = np.max([imgs.shape[i + 1] - crop_size[i] / 2, imgs.shape[i + 1] / 2 + bound_size])  # 右边缘 or 中心点
                e = np.min([crop_size[i] / 2, imgs.shape[i + 1] / 2 - bound_size])  # 左边缘 or 中心点
                target = np.array([np.nan, np.nan, np.nan, np.nan])
            if s > e:  # 0.3的情况或者0.7里面直径太小导致不满足crop_size
                start.append(np.random.randint(e, s))  # 随机一个
            else:  # 0.7里面直径比较大，会撑满整个crop，rand用于随机移动结节中心点，不让其完全在crop的中心
                start.append(int(target[i]) - crop_size[i] / 2 + np.random.randint(-bound_size / 2, bound_size / 2))

        normstart = np.array(start).astype('float32') / np.array(imgs.shape[1:]) - 0.5  # 映射到 -0.5 ~ 0.5
        normsize = np.array(crop_size).astype('float32') / np.array(imgs.shape[1:])  # 长度
        xx, yy, zz = np.meshgrid(np.linspace(normstart[0], normstart[0] + normsize[0], self.crop_size[0] / self.stride),
                                 np.linspace(normstart[1], normstart[1] + normsize[1], self.crop_size[1] / self.stride),
                                 np.linspace(normstart[2], normstart[2] + normsize[2], self.crop_size[2] / self.stride),
                                 indexing='ij')  # 3维网格
        # 应该是从crop_size=96到网络输出24的映射方法
        coord = np.concatenate([xx[np.newaxis, ...], yy[np.newaxis, ...], zz[np.newaxis, :]], 0).astype('float32')

        pad = []
        pad.append([0, 0])
        for i in range(3):
            left_pad = max(0, -start[i])
            right_pad = max(0, start[i] + crop_size[i] - imgs.shape[i + 1])
            pad.append([left_pad, right_pad])
        crop = imgs[:,
               max(start[0], 0):min(start[0] + crop_size[0], imgs.shape[1]),
               max(start[1], 0):min(start[1] + crop_size[1], imgs.shape[2]),
               max(start[2], 0):min(start[2] + crop_size[2], imgs.shape[3])]
        crop = np.pad(crop, pad, 'constant', constant_values=self.pad_value)
        for i in range(3):  # 根据裁剪位置调整结节坐标
            target[i] = target[i] - start[i]
        for i in range(len(bboxes)):  # 根据裁剪位置调整bbox坐标
            for j in range(3):
                bboxes[i][j] = bboxes[i][j] - start[j]

        if is_scale:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                crop = zoom(crop, [1, scale, scale, scale], order=1)
            newpad = self.crop_size[0] - crop.shape[1:][0]
            if newpad < 0:  # 放大了 剪回去
                crop = crop[:, :-newpad, :-newpad, :-newpad]
            elif newpad > 0:  # 缩小了 填上
                pad2 = [[0, 0], [0, newpad], [0, newpad], [0, newpad]]
                crop = np.pad(crop, pad2, 'constant', constant_values=self.pad_value)
            for i in range(4):  # 坐标缩放
                target[i] = target[i] * scale
            for i in range(len(bboxes)):
                for j in range(4):  # bbox 缩放
                    bboxes[i][j] = bboxes[i][j] * scale
        return crop, target, bboxes, coord
