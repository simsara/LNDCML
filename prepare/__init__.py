import os

import SimpleITK as SimpleITK

import numpy as np
import pandas
from scipy.ndimage.interpolation import zoom
from scipy.ndimage.morphology import binary_dilation, generate_binary_structure
from skimage.morphology import convex_hull_image

from utils import env, threadpool
from utils.log import get_logger

from tools import *

log = get_logger(__name__)


def process_mask(mask):
    convex_mask = np.copy(mask)
    for i_layer in range(convex_mask.shape[0]):
        mask1 = np.ascontiguousarray(mask[i_layer])
        if np.sum(mask1) > 0:
            mask2 = convex_hull_image(mask1)  # 计算凸包
            if np.sum(mask2) > 1.5 * np.sum(mask1):  # 凸包面积是之前的1.5倍
                mask2 = mask1
        else:
            mask2 = mask1
        convex_mask[i_layer] = mask2
    struct = generate_binary_structure(3, 1)  # 3*3*3 的方块
    dilated_mask = binary_dilation(convex_mask, structure=struct, iterations=10)  # 膨胀10次
    return dilated_mask


def lum_trans(img):
    lung_win = np.array([-1200., 600.])
    new_img = (img - lung_win[0]) / (lung_win[1] - lung_win[0])
    new_img[new_img < 0] = 0
    new_img[new_img > 1] = 1
    new_img = (new_img * 255).astype('uint8')
    return new_img

def resample(imgs, spacing, new_spacing, order=2):
    if len(imgs.shape) == 3:
        new_shape = np.round(imgs.shape * spacing / new_spacing)
        true_spacing = spacing * imgs.shape / new_shape
        resize_factor = new_shape / imgs.shape
        imgs = zoom(imgs, resize_factor, mode='nearest', order=order)
        return imgs, true_spacing
    elif len(imgs.shape) == 4:
        n = imgs.shape[-1]
        newimg = []
        for i in range(n):
            slice = imgs[:, :, :, i]
            newslice, true_spacing = resample(slice, spacing, new_spacing)
            newimg.append(newslice)
        newimg = np.transpose(np.array(newimg), [1, 2, 3, 0])
        return newimg, true_spacing
    else:
        raise ValueError('wrong shape')


def save_file(file_name, data):
    if os.path.exists(file_name):
        if env.get('prepare_cover_data') == '1':
            os.remove(file_name)
        else:
            return
    np.save(file_name, data)


def save_npy_luna(id, annos, filelist, luna_segment, luna_data, savepath):
    is_label = True
    is_clean = True
    resolution = np.array([1, 1, 1])
    #     resolution = np.array([2,2,2])
    name = filelist[id]

    # 因为两张图是对应的，所以后三个变量可以复用
    slice_img, origin, spacing, is_flip = load_itk_image(os.path.join(luna_data, name + '.mhd'))
    mask_img, origin, spacing, is_flip = load_itk_image(os.path.join(luna_segment, name + '.mhd'))
    # show_image(slice_img)
    # show_image(mask_img)
    if is_flip:
        mask_img = mask_img[:, ::-1, ::-1]  # 原图被翻转，翻转回来，(slice, w, h)
    new_shape = np.round(np.array(mask_img.shape) * spacing / resolution).astype('int')  # 获取mask在新分辨率下的尺寸
    m1 = mask_img == 3  # 二值图 LUNA16的掩码有两种值，3和4
    m2 = mask_img == 4  # 其它置位0
    mask_img = m1 + m2  # 合并

    xx, yy, zz = np.where(mask_img)  # 返回的是>0的坐标
    box = np.array([[np.min(xx), np.max(xx)], [np.min(yy), np.max(yy)], [np.min(zz), np.max(zz)]])  # 取每个维度的上下限
    box = box * np.expand_dims(spacing, 1) / np.expand_dims(resolution, 1)  # 按新分辨率调整box
    box = np.floor(box).astype('int')
    margin = 5
    extend_box = np.vstack(
        [np.max([[0, 0, 0], box[:, 0] - margin], axis=0),
         np.min([new_shape, box[:, 1] + 2 * margin], axis=0).T]
    ).T  # 稍微把box扩展一下

    if is_clean:
        dm1 = process_mask(m1)  # 对掩码采取膨胀操作，去除肺部黑洞
        dm2 = process_mask(m2)
        dilated_mask = dm1 + dm2
        mask_img = m1 + m2

        extra_mask = dilated_mask ^ mask_img  # 提取轮廓
        bone_thresh = 210
        pad_value = 170

        if is_flip:
            slice_img = slice_img[:, ::-1, ::-1]
        slice_img = lum_trans(slice_img)  # 对原始数据阈值化，并归一化
        slice_img = slice_img * dilated_mask + pad_value * (1 - dilated_mask).astype('uint8')  # 170对应归一化后的水，掩码外的区域补充为水
        bones = (slice_img * extra_mask) > bone_thresh  # 210对应归一化后的骨头，凡是大于骨头的区域都填充为水
        slice_img[bones] = pad_value

        sliceim1, _ = resample(slice_img, spacing, resolution, order=1)
        sliceim2 = sliceim1[extend_box[0, 0]:extend_box[0, 1],
                   extend_box[1, 0]:extend_box[1, 1],
                   extend_box[2, 0]:extend_box[2, 1]]
        slice_img = sliceim2[np.newaxis, ...]
        save_file(os.path.join(savepath, name + '_clean.npy'), slice_img)
        save_file(os.path.join(savepath, name + '_spacing.npy'), spacing)
        save_file(os.path.join(savepath, name + '_extendbox.npy'), extend_box)
        save_file(os.path.join(savepath, name + '_origin.npy'), origin)
        save_file(os.path.join(savepath, name + '_mask.npy'), mask_img)

    if is_label:
        this_annos = np.copy(annos[annos[:, 0] == name])  # 读取该病例对应标签
        label = []
        if len(this_annos) > 0:
            for c in this_annos:
                pos = world_to_voxel(c[1:4][::-1], origin=origin, spacing=spacing)
                if is_flip:
                    pos[1:] = mask_img.shape[1:3] - pos[1:]
                label.append(np.concatenate([pos, [c[4] / spacing[1]]]))

        label = np.array(label)
        if len(label) == 0:
            label2 = np.array([[0, 0, 0, 0]])
        else:
            label2 = np.copy(label).T
            label2[:3] = label2[:3] * np.expand_dims(spacing, 1) / np.expand_dims(resolution, 1)  # 对标签应用新的分辨率
            label2[3] = label2[3] * spacing[1] / resolution[1]  # 对直径应用新的分辨率
            label2[:3] = label2[:3] - np.expand_dims(extend_box[:, 0], 1)  # 将box外的长度砍掉，也就是相对于box的坐标
            label2 = label2[:4].T
        label_file_name = os.path.join(savepath, '%s_label.npy' % name)
        save_file(label_file_name, label2)

    log.info('Process done: %s' % name)


def prepare_luna():
    luna_segment = env.get('luna_segment')  # 存放CT掩码的路径
    save_path = env.get('preprocess_result_path')  # 存放预处理后数据的路径
    luna_data = env.get('luna_data')  # LUNA16的原始数据
    luna_label = env.get('luna_label')  # 存放所有病例标签的文件 内容格式: id, x, y, z, r

    log.info('starting preprocessing luna')

    annos = np.array(pandas.read_csv(luna_label))
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    for setidx in range(10):
        log.info('process subset%d' % setidx)
        subset_dir = os.path.join(luna_data, 'subset' + str(setidx))
        tosave_dir = os.path.join(save_path, 'subset' + str(setidx))
        filelist = [f.split('.mhd')[0] for f in os.listdir(subset_dir) if f.endswith('.mhd')]
        if not os.path.exists(tosave_dir):
            os.mkdir(tosave_dir)

        for i in range(len(filelist)):
            threadpool.submit(save_npy_luna, id=i, annos=annos, filelist=filelist,
                              luna_segment=luna_segment, luna_data=subset_dir,
                              savepath=tosave_dir)
            if i == 1:
                break
        break  # 先搞一张图
    threadpool.join()  # 等线程跑完


if __name__ == '__main__':
    prepare_luna()
