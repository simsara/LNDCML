import os

import SimpleITK as SimpleITK
import numpy as np
import pandas
from scipy.ndimage.interpolation import zoom
from scipy.ndimage.morphology import binary_dilation, generate_binary_structure
from skimage.morphology import convex_hull_image

from utils import env


def load_itk_image(filename):
    with open(filename) as f:  # auto close
        contents = f.readlines()
        line = [k for k in contents if k.startswith('TransformMatrix')][0]
        transform_m = np.array(line.split(' = ')[1].split(' ')).astype('float')
        transform_m = np.round(transform_m)
        if np.any(transform_m != np.array([1, 0, 0, 0, 1, 0, 0, 0, 1])):
            is_flip = True
        else:
            is_flip = False

    itk_image = SimpleITK.ReadImage(filename)
    numpy_image = SimpleITK.GetArrayFromImage(itk_image)

    numpy_origin = np.array(list(reversed(itk_image.GetOrigin())))
    numpy_spacing = np.array(list(reversed(itk_image.GetSpacing())))

    return numpy_image, numpy_origin, numpy_spacing, is_flip


def process_mask(mask):
    convex_mask = np.copy(mask)
    for i_layer in range(convex_mask.shape[0]):
        mask1 = np.ascontiguousarray(mask[i_layer])
        if np.sum(mask1) > 0:
            mask2 = convex_hull_image(mask1)
            if np.sum(mask2) > 1.5 * np.sum(mask1):
                mask2 = mask1
        else:
            mask2 = mask1
        convex_mask[i_layer] = mask2
    struct = generate_binary_structure(3, 1)
    dilated_mask = binary_dilation(convex_mask, structure=struct, iterations=10)
    return dilated_mask


def lumTrans(img):
    lungwin = np.array([-1200., 600.])
    newimg = (img - lungwin[0]) / (lungwin[1] - lungwin[0])
    newimg[newimg < 0] = 0
    newimg[newimg > 1] = 1
    newimg = (newimg * 255).astype('uint8')
    return newimg


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


def worldToVoxelCoord(worldCoord, origin, spacing):
    stretchedVoxelCoord = np.absolute(worldCoord - origin)
    voxelCoord = stretchedVoxelCoord / spacing
    return voxelCoord


def save_npy_luna(id, annos, filelist, luna_segment, luna_data, savepath):
    is_label = True
    is_clean = True
    resolution = np.array([1, 1, 1])
    #     resolution = np.array([2,2,2])
    name = filelist[id]

    sliceim, origin, spacing, is_flip = load_itk_image(os.path.join(luna_data, name + '.mhd'))

    Mask, origin, spacing, is_flip = load_itk_image(os.path.join(luna_segment, name + '.mhd'))
    if is_flip:
        Mask = Mask[:, ::-1, ::-1]
    newshape = np.round(np.array(Mask.shape) * spacing / resolution).astype('int')
    m1 = Mask == 3
    m2 = Mask == 4
    Mask = m1 + m2

    xx, yy, zz = np.where(Mask)
    box = np.array([[np.min(xx), np.max(xx)], [np.min(yy), np.max(yy)], [np.min(zz), np.max(zz)]])
    box = box * np.expand_dims(spacing, 1) / np.expand_dims(resolution, 1)
    box = np.floor(box).astype('int')
    margin = 5
    extendbox = np.vstack(
        [np.max([[0, 0, 0], box[:, 0] - margin], 0), np.min([newshape, box[:, 1] + 2 * margin], axis=0).T]).T

    this_annos = np.copy(annos[annos[:, 0] == (name)])

    if is_clean:
        convex_mask = m1
        dm1 = process_mask(m1)
        dm2 = process_mask(m2)
        dilated_mask = dm1 + dm2
        Mask = m1 + m2

        extramask = dilated_mask ^ Mask
        bone_thresh = 210
        pad_value = 170

        if is_flip:
            sliceim = sliceim[:, ::-1, ::-1]
            print('flip!')
        sliceim = lumTrans(sliceim)
        sliceim = sliceim * dilated_mask + pad_value * (1 - dilated_mask).astype('uint8')
        bones = (sliceim * extramask) > bone_thresh
        sliceim[bones] = pad_value

        sliceim1, _ = resample(sliceim, spacing, resolution, order=1)
        sliceim2 = sliceim1[extendbox[0, 0]:extendbox[0, 1],
                   extendbox[1, 0]:extendbox[1, 1],
                   extendbox[2, 0]:extendbox[2, 1]]
        sliceim = sliceim2[np.newaxis, ...]
        np.save(os.path.join(savepath, name + '_clean.npy'), sliceim)
        np.save(os.path.join(savepath, name + '_spacing.npy'), spacing)
        np.save(os.path.join(savepath, name + '_extendbox.npy'), extendbox)
        np.save(os.path.join(savepath, name + '_origin.npy'), origin)
        np.save(os.path.join(savepath, name + '_mask.npy'), Mask)

    if is_label:
        this_annos = np.copy(annos[annos[:, 0] == (name)])
        label = []
        if len(this_annos) > 0:

            for c in this_annos:
                pos = worldToVoxelCoord(c[1:4][::-1], origin=origin, spacing=spacing)
                if is_flip:
                    pos[1:] = Mask.shape[1:3] - pos[1:]
                label.append(np.concatenate([pos, [c[4] / spacing[1]]]))

        label = np.array(label)
        if len(label) == 0:
            label2 = np.array([[0, 0, 0, 0]])
        else:
            label2 = np.copy(label).T
            label2[:3] = label2[:3] * np.expand_dims(spacing, 1) / np.expand_dims(resolution, 1)
            label2[3] = label2[3] * spacing[1] / resolution[1]
            label2[:3] = label2[:3] - np.expand_dims(extendbox[:, 0], 1)
            label2 = label2[:4].T
        np.save(os.path.join(savepath, name + '_label.npy'), label2)

    print(name)


def prepare_luna():
    luna_segment = env.get('luna_segment')  # 存放CT掩码的路径
    save_path = env.get('preprocess_result_path')  # 存放预处理后数据的路径
    luna_data = env.get('luna_data')  # LUNA16的原始数据
    luna_label = env.get('luna_label')  # 存放所有病例标签的文件 内容格式: id, x, y, z, r

    print('starting preprocessing luna')

    annos = np.array(pandas.read_csv(luna_label))
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    for setidx in range(1):  # 先用1个文件夹测试
        print('process subset%d' % setidx)
        subset_dir = os.path.join(luna_data, 'subset' + str(setidx))
        tosave_dir = os.path.join(save_path, 'subset' + str(setidx))
        filelist = [f.split('.mhd')[0] for f in os.listdir(subset_dir) if f.endswith('.mhd')]
        if not os.path.exists(tosave_dir):
            os.mkdir(tosave_dir)

        for i in range(len(filelist)):
            save_npy_luna(id=i, annos=annos, filelist=filelist,
                          luna_segment=luna_segment, luna_data=subset_dir,
                          savepath=tosave_dir)


if __name__ == '__main__':
    prepare_luna()
