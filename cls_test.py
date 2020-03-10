import json
import os

import SimpleITK as sitk
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from torch.autograd import Variable

from nodcls import corp_size, get_transform, lunanod, get_net, cls_resources_dir
from prepare import binarize_per_slice, all_slice_analysis, fill_hole, two_lung_only, process_mask, lum_trans, resample
from utils import file, env
from utils.log import get_logger

log = get_logger(__name__)
test_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'test_resources')
save_file_prefix = 'cls_preprocess'
cls_annotations_json = 'cls-annotations.json'


def read_nii(idx):
    itkimage = sitk.ReadImage(os.path.join(test_dir, '%d.nii.gz' % idx))
    numpyImage = sitk.GetArrayFromImage(itkimage)

    numpyOrigin = np.array(list(reversed(itkimage.GetOrigin())))  # 世界坐标系的原点坐标
    numpySpacing = np.array(list(reversed(itkimage.GetSpacing())))  # 两个像素点之间的真实距离
    return numpyImage, numpyOrigin, numpySpacing


def make_mask(case_pixels, origin, spacing):
    bw = binarize_per_slice(case_pixels, spacing)  # 1-对每一张切片进行二值化处理
    log.info('二值化图像 %s' % str(bw.shape))
    bw_plt = bw  # *255
    # log.info(bw_plt[slice_plt,:,:])
    # plt.imshow(bw_plt[slice_plt,:,:],'gray')
    # plt.show()
    flag = 0
    cut_num = 0
    cut_step = 2
    bw0 = np.copy(bw)
    while flag == 0 and cut_num < bw.shape[0]:
        bw = np.copy(bw0)
        bw, flag = all_slice_analysis(bw, spacing, cut_num=cut_num, vol_limit=[0.68, 7.5])
        cut_num = cut_num + cut_step

    bw = fill_hole(bw)
    # plt.imshow(bw[slice_plt,:,:],'gray')
    # plt.show()
    log.info('shape: %s' % str(bw.shape))
    bw1, bw2, bw = two_lung_only(bw, spacing)
    # plt.imshow(bw[slice_plt, :, :], 'gray')
    # plt.show()
    return bw, bw1, bw2


def preprocess(mask, mask1, mask2, idx):
    resolution = np.array([1, 1, 1])
    sliceim, origin, spacing = read_nii(idx)  # 加载原始数据
    newshape = np.round(np.array(mask.shape) * spacing / resolution).astype('int')  # 获取mask在新分辨率下的尺寸
    xx, yy, zz = np.where(mask)  # 确定mask的边界
    box = np.array([[np.min(xx), np.max(xx)], [np.min(yy), np.max(yy)], [np.min(zz), np.max(zz)]])
    box = box * np.expand_dims(spacing, 1) / np.expand_dims(resolution, 1)  # 对边界即掩码的最小外部长方体应用新的分辨率
    box = np.floor(box).astype('int')
    margin = 5
    extend_box = np.vstack(
        [np.max([[0, 0, 0], box[:, 0] - margin], axis=0),
         np.min([newshape, box[:, 1] + 2 * margin], axis=0).T]
    ).T  # 稍微把box扩展一下

    dm1 = process_mask(mask1)  # 对mask采取膨胀操作，去除肺部空洞
    dm2 = process_mask(mask2)
    dilated_mask = dm1 + dm2
    mask = mask1 + mask2
    # plt.imshow(dilated_mask[slice_plt, :, :], 'gray')
    # plt.show()

    extra_mask = dilated_mask ^ mask
    # plt.imshow(extra_mask[slice_plt, :, :], 'gray')
    # plt.show()
    log.info('extra_mask_shape %s' % str(extra_mask.shape))
    bone_thresh = 210
    pad_value = 170

    final_mask = extra_mask * sliceim
    # plt.imshow(final_mask[slice_plt, :, :])
    # plt.show()

    sliceim = lum_trans(sliceim)  # 对原始数据阈值归一化
    sliceim = sliceim * dilated_mask + pad_value * (1 - dilated_mask).astype('uint8')  # 170对应归一化之后的水，掩码外的部分补充为水
    bones = (sliceim * extra_mask) > bone_thresh  # 210对应归一化后的骨头，凡事大于骨头的区域都填充为水
    sliceim[bones] = pad_value
    # plt.imshow(sliceim[slice_plt, :, :],'gray')
    # plt.show()

    sliceim, _ = resample(sliceim, spacing, resolution, order=1)  # 对原始数据重采样，采用新的分辨率
    sliceim = sliceim[extend_box[0, 0]:extend_box[0, 1],  # 将extend_box内数据取出作为最后的结果
              extend_box[1, 0]:extend_box[1, 1],
              extend_box[2, 0]:extend_box[2, 1]]
    log.info('sliceim_shape %s' % str(sliceim.shape))
    plt.imshow(sliceim[155, :, :], 'gray')
    plt.show()
    sliceim = sliceim[np.newaxis, ...]

    np.save(os.path.join(test_dir, '%s_%d_clean.npy' % (save_file_prefix, idx)), sliceim)

    return sliceim


def corp(clean_file_prefix):
    save_path = file.get_cls_corp_path()
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    annotations = None
    with open(os.path.join(test_dir, cls_annotations_json), 'r') as f:
        annotations = json.load(f)

    for idx in range(len(annotations)):
        fname = '%s-%d' % (clean_file_prefix, idx)
        i = int(annotations[idx]['i'])
        data = np.load(os.path.join(test_dir, '%s_%d_clean.npy' % (clean_file_prefix, i)))
        crdx = int(float(annotations[idx]['x']))
        crdy = int(float(annotations[idx]['y']))
        crdz = int(float(annotations[idx]['z']))
        bgx = max(0, crdx - corp_size // 2)
        bgy = max(0, crdy - corp_size // 2)
        bgz = max(0, crdz - corp_size // 2)
        cropdata = np.ones((corp_size, corp_size, corp_size)) * 170  # 170对应归一化后的水
        cropdatatmp = np.array(data[0, bgx:bgx + corp_size, bgy:bgy + corp_size, bgz:bgz + corp_size])
        cropdata[
        corp_size // 2 - cropdatatmp.shape[0] // 2:corp_size // 2 - cropdatatmp.shape[0] // 2 + cropdatatmp.shape[0], \
        corp_size // 2 - cropdatatmp.shape[1] // 2:corp_size // 2 - cropdatatmp.shape[1] // 2 + cropdatatmp.shape[1], \
        corp_size // 2 - cropdatatmp.shape[2] // 2:corp_size // 2 - cropdatatmp.shape[2] // 2 + cropdatatmp.shape[2]
        ] = np.array(2 - cropdatatmp)
        assert cropdata.shape[0] == corp_size and cropdata.shape[1] == corp_size and cropdata.shape[2] == corp_size
        np.save(os.path.join(save_path, fname + '.npy'), cropdata)
        log.info('Saved: %s. Shape: %s.' % (fname, str(cropdata.shape)))


def get_file_list(args, clean_file_prefix):
    tefnamelst = []
    telabellst = []
    tefeatlst = []

    annotations = None
    with open(os.path.join(test_dir, cls_annotations_json), 'r') as f:
        annotations = json.load(f)

    mxx = mxy = mxz = mxd = 0
    for idx in range(len(annotations)):
        srsid = '%s-%d' % (clean_file_prefix, idx)
        x = int(float(annotations[idx]['x']))
        y = int(float(annotations[idx]['y']))
        z = int(float(annotations[idx]['z']))
        d = int(float(annotations[idx]['d']))

        mxx = max(abs(float(x)), mxx)
        mxy = max(abs(float(y)), mxy)
        mxz = max(abs(float(z)), mxz)
        mxd = max(abs(float(d)), mxd)
        # crop raw pixel as feature
        corp_file = os.path.join(file.get_cls_corp_path(), '%s.npy' % srsid)
        data = np.load(corp_file)
        bgx = data.shape[0] // 2 - corp_size // 2
        bgy = data.shape[1] // 2 - corp_size // 2
        bgz = data.shape[2] // 2 - corp_size // 2
        data = np.array(data[bgx:bgx + corp_size, bgy:bgy + corp_size, bgz:bgz + corp_size])
        feat = np.hstack((np.reshape(data, (-1,)) / 255, float(d)))

        tefnamelst.append(srsid + '.npy')
        telabellst.append(0)
        tefeatlst.append(feat)

    for idx in range(len(tefeatlst)):
        tefeatlst[idx][-1] /= mxd
    log.info('[Existed] Size of test files: %d.' % len(tefnamelst))

    return tefnamelst, telabellst, tefeatlst


def get_loader(args, clean_file_prefix):
    preprocesspath = file.get_cls_corp_path()
    _, transform_test = get_transform()
    tefnamelst, telabellst, tefeatlst = get_file_list(args, clean_file_prefix)

    testset = lunanod(preprocesspath, tefnamelst, telabellst, tefeatlst, train=False, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                             shuffle=False, num_workers=args.workers)
    log.info('[Used] Size of test files: %d.' % len(testset))
    return testloader


def cls(clean_file_prefix):
    args = env.get_args()
    test_loader = get_loader(args, clean_file_prefix)
    net, loss, opt = get_net(args)
    net.eval()
    gbm = GradientBoostingClassifier(random_state=0)
    trainfeat = np.load(os.path.join(cls_resources_dir, 'train_feat.npy'))
    trainlabel = np.load(os.path.join(cls_resources_dir, 'train_label.npy'))
    gbm.fit(trainfeat, trainlabel)
    with torch.no_grad():
        correct = 0
        total = 0
        test_loss = 0
        test_size = len(test_loader.dataset)
        testfeat = np.zeros((test_size, 2560 + corp_size * corp_size * corp_size + 1))
        testlabel = np.zeros((test_size,))
        idx = 0
        for batch_idx, (inputs, targets, feat) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs), Variable(targets)
            outputs, dfeat = net(inputs)
            # add feature into the array
            testfeat[idx:idx + len(targets), :2560] = np.array(dfeat.data.cpu().numpy())
            for i in range(len(targets)):
                testfeat[idx + i, 2560:] = np.array(Variable(feat[i]).data.cpu().numpy())
                testlabel[idx + i] = np.array(targets[i].data.cpu().numpy())
            idx += len(targets)
            loss_val = loss(outputs, targets)
            test_loss += loss_val.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()
        accout = round(correct.data.cpu().numpy() / total, 4)
        test_result = gbm.predict(testfeat)
        prob_result = gbm.predict_proba(testfeat)
        compare_result = (test_result == testlabel)
        gbtteacc = round(np.mean(compare_result), 4)
        df = pd.DataFrame(data=[prob_result[:, 0], prob_result[:, 1], test_result, testlabel, compare_result]).T
        df.to_excel(os.path.join(cls_resources_dir, 'cls_nii_output.xls'))
        log.info('Test Loss: %.3f | Acc: %.3f%% (%d/%d) | Gbt: %.3f' % (test_loss / (batch_idx + 1), 100. * accout,
                                                                        correct, total, gbtteacc))


def full_pipeline():
    for i in range(50):
        if not os.path.exists(os.path.join(test_dir, '%d.nii.gz' % i)):
            continue
        case_pixels, origin, spacing = read_nii(i)
        mask, mask1, mask2 = make_mask(case_pixels, origin, spacing)
        preprocess_file = preprocess(mask, mask1, mask2, i)
    # detect_test.detector(save_file_prefix)
    # detect_test.show_result(preprocess_file, save_file_prefix)
    corp(save_file_prefix)
    cls(save_file_prefix)


if __name__ == '__main__':
    full_pipeline()
