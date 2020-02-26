import os

import numpy as np
import pandas
import scipy
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import zoom
from scipy.ndimage.morphology import binary_dilation, generate_binary_structure
from skimage import measure
from skimage.morphology import convex_hull_image

from utils import env, threadpool, file
from utils.log import get_logger
from utils.tools import load_itk_image, world_to_voxel

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
    save_path = file.get_preprocess_result_path()  # 存放预处理后数据的路径
    luna_data = file.get_luna_data_path()  # LUNA16的原始数据
    luna_label = file.get_luna_csv_name('annotations.csv')  # 存放所有病例标签的文件 内容格式: id, x, y, z, r

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


def binarize_per_slice(image, spacing, intensity_th=-600, sigma=1, area_th=30, eccen_th=0.99, bg_patch_size=10):
    bw = np.zeros(image.shape, dtype=bool)

    # prepare a mask, with all corner values set to nan
    image_size = image.shape[1]
    log.info('image_size: %d' % image_size)
    grid_axis = np.linspace(-image_size / 2 + 0.5, image_size / 2 - 0.5, image_size)
    x, y = np.meshgrid(grid_axis, grid_axis)
    d = (x ** 2 + y ** 2) ** 0.5
    nan_mask = (d < image_size / 2).astype(float)
    nan_mask[nan_mask == 0] = np.nan
    for i in range(image.shape[0]):
        # Check if corner pixels are identical, if so the slice  before Gaussian filtering
        if len(np.unique(image[i, 0:bg_patch_size, 0:bg_patch_size])) == 1:
            current_bw = gaussian_filter(np.multiply(image[i].astype('float32'), nan_mask), sigma, truncate=2.0) < intensity_th
        else:
            current_bw = gaussian_filter(image[i].astype('float32'), sigma, truncate=2.0) < intensity_th

        # select proper components
        label = measure.label(current_bw)
        properties = measure.regionprops(label)
        valid_label = set()
        for prop in properties:
            if prop.area * spacing[1] * spacing[2] > area_th and prop.eccentricity < eccen_th:
                valid_label.add(prop.label)
        current_bw = np.in1d(label, list(valid_label)).reshape(label.shape)
        bw[i] = current_bw

    return bw


def all_slice_analysis(bw, spacing, cut_num=0, vol_limit=[0.68, 8.2], area_th=6e3, dist_th=62):
    # in some cases, several top layers need to be removed first
    if cut_num > 0:
        bw0 = np.copy(bw)
        bw[-cut_num:] = False
    label = measure.label(bw, connectivity=1)
    # remove components access to corners
    mid = int(label.shape[2] / 2)
    bg_label = set([label[0, 0, 0], label[0, 0, -1], label[0, -1, 0], label[0, -1, -1], \
                    label[-1 - cut_num, 0, 0], label[-1 - cut_num, 0, -1], label[-1 - cut_num, -1, 0],
                    label[-1 - cut_num, -1, -1], \
                    label[0, 0, mid], label[0, -1, mid], label[-1 - cut_num, 0, mid], label[-1 - cut_num, -1, mid]])
    for l in bg_label:
        label[label == l] = 0

    # select components based on volume
    properties = measure.regionprops(label)
    for prop in properties:
        if prop.area * spacing.prod() < vol_limit[0] * 1e6 or prop.area * spacing.prod() > vol_limit[1] * 1e6:
            label[label == prop.label] = 0

    # prepare a distance map for further analysis
    x_axis = np.linspace(-label.shape[1] / 2 + 0.5, label.shape[1] / 2 - 0.5, label.shape[1]) * spacing[1]
    y_axis = np.linspace(-label.shape[2] / 2 + 0.5, label.shape[2] / 2 - 0.5, label.shape[2]) * spacing[2]
    x, y = np.meshgrid(x_axis, y_axis)
    d = (x ** 2 + y ** 2) ** 0.5
    vols = measure.regionprops(label)
    valid_label = set()
    # select components based on their area and distance to center axis on all slices
    for vol in vols:
        single_vol = label == vol.label
        slice_area = np.zeros(label.shape[0])
        min_distance = np.zeros(label.shape[0])
        for i in range(label.shape[0]):
            slice_area[i] = np.sum(single_vol[i]) * np.prod(spacing[1:3])
            min_distance[i] = np.min(single_vol[i] * d + (1 - single_vol[i]) * np.max(d))

        if np.average([min_distance[i] for i in range(label.shape[0]) if slice_area[i] > area_th]) < dist_th:
            valid_label.add(vol.label)

    bw = np.in1d(label, list(valid_label)).reshape(label.shape)

    # fill back the parts removed earlier
    if cut_num > 0:
        # bw1 is bw with removed slices, bw2 is a dilated version of bw, part of their intersection is returned as final mask
        bw1 = np.copy(bw)
        bw1[-cut_num:] = bw0[-cut_num:]
        bw2 = np.copy(bw)
        bw2 = scipy.ndimage.binary_dilation(bw2, iterations=cut_num)
        bw3 = bw1 & bw2
        label = measure.label(bw, connectivity=1)
        label3 = measure.label(bw3, connectivity=1)
        l_list = list(set(np.unique(label)) - {0})
        valid_l3 = set()
        for l in l_list:
            indices = np.nonzero(label == l)
            l3 = label3[indices[0][0], indices[1][0], indices[2][0]]
            if l3 > 0:
                valid_l3.add(l3)
        bw = np.in1d(label3, list(valid_l3)).reshape(label3.shape)

    return bw, len(valid_label)


def fill_hole(bw):
    # fill 3d holes
    label = measure.label(~bw)
    # idendify corner components
    bg_label = set([label[0, 0, 0], label[0, 0, -1], label[0, -1, 0], label[0, -1, -1], \
                    label[-1, 0, 0], label[-1, 0, -1], label[-1, -1, 0], label[-1, -1, -1]])
    bw = ~np.in1d(label, list(bg_label)).reshape(label.shape)

    return bw


def extract_main(bw, cover=0.95):
    for i in range(bw.shape[0]):
        current_slice = bw[i]
        label = measure.label(current_slice)
        properties = measure.regionprops(label)
        properties.sort(key=lambda x: x.area, reverse=True)
        area = [prop.area for prop in properties]
        count = 0
        sum = 0
        while sum < np.sum(area) * cover:
            sum = sum + area[count]
            count = count + 1
        filter = np.zeros(current_slice.shape, dtype=bool)
        for j in range(count):
            bb = properties[j].bbox
            filter[bb[0]:bb[2], bb[1]:bb[3]] = filter[bb[0]:bb[2], bb[1]:bb[3]] | properties[j].convex_image
        bw[i] = bw[i] & filter

    label = measure.label(bw)
    properties = measure.regionprops(label)
    properties.sort(key=lambda x: x.area, reverse=True)
    bw = label == properties[0].label

    return bw


def fill_2d_hole(bw):
    for i in range(bw.shape[0]):
        current_slice = bw[i]
        label = measure.label(current_slice)
        properties = measure.regionprops(label)
        for prop in properties:
            bb = prop.bbox
            current_slice[bb[0]:bb[2], bb[1]:bb[3]] = current_slice[bb[0]:bb[2], bb[1]:bb[3]] | prop.filled_image
        bw[i] = current_slice

    return bw


def two_lung_only(bw, spacing, max_iter=22, max_ratio=4.8):
    found_flag = False
    iter_count = 0
    bw0 = np.copy(bw)
    while not found_flag and iter_count < max_iter:
        label = measure.label(bw, connectivity=2)
        properties = measure.regionprops(label)
        properties.sort(key=lambda x: x.area, reverse=True)
        if len(properties) > 1 and properties[0].area / properties[1].area < max_ratio:
            found_flag = True
            bw1 = label == properties[0].label
            bw2 = label == properties[1].label
        else:
            bw = scipy.ndimage.binary_erosion(bw)
            iter_count = iter_count + 1

    if found_flag:
        d1 = scipy.ndimage.morphology.distance_transform_edt(bw1 == False, sampling=spacing)
        d2 = scipy.ndimage.morphology.distance_transform_edt(bw2 == False, sampling=spacing)
        bw1 = bw0 & (d1 < d2)
        bw2 = bw0 & (d1 > d2)

        bw1 = extract_main(bw1)
        bw2 = extract_main(bw2)

    else:
        bw1 = bw0
        bw2 = np.zeros(bw.shape).astype('bool')

    bw1 = fill_2d_hole(bw1)
    bw2 = fill_2d_hole(bw2)
    bw = bw1 | bw2

    return bw1, bw2, bw


if __name__ == '__main__':
    prepare_luna()
