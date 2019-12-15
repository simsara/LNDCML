import math
import random

import numpy as np

from utils.log import get_logger

log = get_logger(__name__)


class LabelMapping(object):
    def __init__(self, config, phase):
        self.stride = np.array(config['stride'])
        self.num_neg = int(config['num_neg'])
        self.th_neg = config['th_neg']
        self.anchors = np.asarray(config['anchors'])
        self.phase = phase
        if phase == 'train':
            self.th_pos = config['th_pos_train']
        elif phase == 'val':
            self.th_pos = config['th_pos_val']

    def __call__(self, input_size, target, bboxes, filename):
        stride = self.stride
        num_neg = self.num_neg
        th_neg = self.th_neg
        anchors = self.anchors
        th_pos = self.th_pos

        output_size = []
        for i in range(3):
            if input_size[i] % stride != 0:
                err_msg = 'Output size doesnt fit stride [%s]' % filename
                log.info(err_msg)
                raise ValueError(err_msg)
            # assert(input_size[i] % stride == 0)
            output_size.append(math.floor(input_size[i] / stride))

        label = -1 * np.ones(output_size + [len(anchors), 5], np.float32)
        offset = ((stride.astype('float')) - 1) / 2
        oz = np.arange(offset, offset + stride * (output_size[0] - 1) + 1, stride)
        oh = np.arange(offset, offset + stride * (output_size[1] - 1) + 1, stride)
        ow = np.arange(offset, offset + stride * (output_size[2] - 1) + 1, stride)

        for bbox in bboxes:
            for i, anchor in enumerate(anchors):
                iz, ih, iw = select_samples(bbox, anchor, th_neg, oz, oh, ow)
                label[iz, ih, iw, i, 0] = 0

        if self.phase == 'train' and self.num_neg > 0:
            neg_z, neg_h, neg_w, neg_a = np.where(label[:, :, :, :, 0] == -1)
            neg_idcs = random.sample(range(len(neg_z)), min(num_neg, len(neg_z)))
            neg_z, neg_h, neg_w, neg_a = neg_z[neg_idcs], neg_h[neg_idcs], neg_w[neg_idcs], neg_a[neg_idcs]
            label[:, :, :, :, 0] = 0
            label[neg_z, neg_h, neg_w, neg_a, 0] = -1

        if np.isnan(target[0]):
            return label
        iz, ih, iw, ia = [], [], [], []
        for i, anchor in enumerate(anchors):
            iiz, iih, iiw = select_samples(target, anchor, th_pos, oz, oh, ow)
            iz.append(iiz)
            ih.append(iih)
            iw.append(iiw)
            ia.append(i * np.ones((len(iiz),), np.int64))
        iz = np.concatenate(iz, 0)
        ih = np.concatenate(ih, 0)
        iw = np.concatenate(iw, 0)
        ia = np.concatenate(ia, 0)
        flag = True
        if len(iz) == 0:
            pos = []
            for i in range(3):
                pos.append(max(0, int(np.round((target[i] - offset) / stride))))
            idx = np.argmin(np.abs(np.log(target[3] / anchors)))
            pos.append(idx)
            flag = False
        else:
            idx = random.sample(range(len(iz)), 1)[0]
            pos = [iz[idx], ih[idx], iw[idx], ia[idx]]
        dz = (target[0] - oz[pos[0]]) / anchors[pos[3]]
        dh = (target[1] - oh[pos[1]]) / anchors[pos[3]]
        dw = (target[2] - ow[pos[2]]) / anchors[pos[3]]
        dd = np.log(target[3] / anchors[pos[3]])
        label[pos[0], pos[1], pos[2], pos[3], :] = [1, dz, dh, dw, dd]
        return label


def select_samples(bbox, anchor, th, oz, oh, ow):
    z, h, w, d = bbox
    max_overlap = min(d, anchor)
    min_overlap = np.power(max(d, anchor), 3) * th / max_overlap / max_overlap
    if min_overlap > max_overlap:
        return np.zeros((0,), np.int64), np.zeros((0,), np.int64), np.zeros((0,), np.int64)
    else:
        s = z - 0.5 * np.abs(d - anchor) - (max_overlap - min_overlap)
        e = z + 0.5 * np.abs(d - anchor) + (max_overlap - min_overlap)
        mz = np.logical_and(oz >= s, oz <= e)
        iz = np.where(mz)[0]

        s = h - 0.5 * np.abs(d - anchor) - (max_overlap - min_overlap)
        e = h + 0.5 * np.abs(d - anchor) + (max_overlap - min_overlap)
        mh = np.logical_and(oh >= s, oh <= e)
        ih = np.where(mh)[0]

        s = w - 0.5 * np.abs(d - anchor) - (max_overlap - min_overlap)
        e = w + 0.5 * np.abs(d - anchor) + (max_overlap - min_overlap)
        mw = np.logical_and(ow >= s, ow <= e)
        iw = np.where(mw)[0]

        if len(iz) == 0 or len(ih) == 0 or len(iw) == 0:
            return np.zeros((0,), np.int64), np.zeros((0,), np.int64), np.zeros((0,), np.int64)

        lz, lh, lw = len(iz), len(ih), len(iw)
        iz = iz.reshape((-1, 1, 1))
        ih = ih.reshape((1, -1, 1))
        iw = iw.reshape((1, 1, -1))
        iz = np.tile(iz, (1, lh, lw)).reshape((-1))
        ih = np.tile(ih, (lz, 1, lw)).reshape((-1))
        iw = np.tile(iw, (lz, lh, 1)).reshape((-1))
        centers = np.concatenate([
            oz[iz].reshape((-1, 1)),
            oh[ih].reshape((-1, 1)),
            ow[iw].reshape((-1, 1))], axis=1)

        r0 = anchor / 2
        s0 = centers - r0
        e0 = centers + r0

        r1 = d / 2
        s1 = bbox[:3] - r1
        s1 = s1.reshape((1, -1))
        e1 = bbox[:3] + r1
        e1 = e1.reshape((1, -1))

        overlap = np.maximum(0, np.minimum(e0, e1) - np.maximum(s0, s1))

        intersection = overlap[:, 0] * overlap[:, 1] * overlap[:, 2]
        union = anchor * anchor * anchor + d * d * d - intersection

        iou = intersection / union

        mask = iou >= th
        # if th > 0.4:
        #   if np.sum(mask) == 0:
        #      log.info(['iou not large', iou.max()])
        # else:
        #    log.info(['iou large', iou[mask]])
        iz = iz[mask]
        ih = ih[mask]
        iw = iw[mask]
        return iz, ih, iw
