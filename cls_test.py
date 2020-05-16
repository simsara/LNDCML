import json
import math
import os

import matplotlib.pyplot as plt
import SimpleITK as sitk
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import GradientBoostingClassifier
from torch import nn
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

import detect
import nodcls
from detect import netdef, SplitCombine, DataBowl3Detector, data
from nodcls import corp_size, get_transform, lunanod, cls_resources_dir
from prepare import binarize_per_slice, all_slice_analysis, fill_hole, two_lung_only, process_mask, lum_trans, resample
from utils import file, env, gpu, tools
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
    # plt.imshow(sliceim[155, :, :], 'gray')
    # plt.show()
    sliceim = sliceim[np.newaxis, ...]

    np.save(os.path.join(test_dir, '%s_%d_clean.npy' % (save_file_prefix, idx)), sliceim)

    return sliceim


def corp():
    save_path = file.get_cls_corp_path()
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    annotations = None
    with open(os.path.join(test_dir, cls_annotations_json), 'r') as f:
        annotations = json.load(f)

    for idx in range(len(annotations)):
        fname = '%s-%d' % (save_file_prefix, idx)
        i = int(annotations[idx]['i'])
        data = np.load(os.path.join(test_dir, '%s_%d_clean.npy' % (save_file_prefix, i)))
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


def get_file_list(args):
    tefnamelst = []
    telabellst = []
    tefeatlst = []

    annotations = None
    with open(os.path.join(test_dir, cls_annotations_json), 'r') as f:
        annotations = json.load(f)

    mxx = mxy = mxz = mxd = 0
    for idx in range(len(annotations)):
        srsid = '%s-%d' % (save_file_prefix, idx)
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
        if not os.path.exists(corp_file):
            continue
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


def get_loader(args):
    preprocesspath = file.get_cls_corp_path()
    _, transform_test = get_transform()
    tefnamelst, telabellst, tefeatlst = get_file_list(args, save_file_prefix)

    testset = lunanod(preprocesspath, tefnamelst, telabellst, tefeatlst, train=False, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                             shuffle=False, num_workers=args.workers)
    log.info('[Used] Size of test files: %d.' % len(testset))
    return testloader


def get_cls_net(args):
    gpu.set_gpu(args.gpu)
    model = nodcls.models.get_model(args.model)
    net = model.get_model()
    checkpoint = torch.load(os.path.join(test_dir, 'cls.ckpt'))
    net.load_state_dict(checkpoint['state_dict'])
    net = torch.nn.DataParallel(net).cuda()
    loss = CrossEntropyLoss()
    return net, loss


def cls():
    args = env.get_args()
    test_loader = get_loader(args)
    net, loss = get_cls_net(args)
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


def get_train_net(args):
    gpu.set_gpu(args.gpu)
    torch.manual_seed(0)
    model = netdef.get_model('dpnse')
    config, net, loss, get_pbb = model.get_model()
    checkpoint = torch.load(os.path.join(test_dir, 'detect.ckpt'))
    net.load_state_dict(checkpoint['state_dict'])
    net = torch.nn.DataParallel(net).cuda()  # 使用多个GPU进行训练
    loss = loss.cuda()
    log.info("we have %s GPUs" % torch.cuda.device_count())
    return config, net, loss, get_pbb


def detector():
    args = env.get_args()
    log.info(args)
    net_config, net, loss, get_pbb = get_train_net(args)

    testdatadir = test_dir  # 预处理结果路径
    testfilelist = [f[:-10] for f in os.listdir(testdatadir) if
                    f.startswith(save_file_prefix) and f.endswith('_clean.npy')]  # 文件名列表

    split_combine = SplitCombine(net_config['side_len'], net_config['max_stride'], net_config['stride'],
                                 net_config['margin'], net_config['pad_value'])
    dataset = DataBowl3Detector(
        testdatadir,
        testfilelist,
        net_config,
        phase='prod',
        split_combine=split_combine)
    test_loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=data.collate,
        pin_memory=False)
    detect.test(test_loader, net, get_pbb, args, net_config, test_dir)


def generate_nii_annotations():
    for i in range(50):
        pbb_name = os.path.join(test_dir, '%s_%d_pbb.npy' % (save_file_prefix, i))
        if not os.path.exists(pbb_name):
            continue
        raw_data = np.load(os.path.join(test_dir, '%s_%d_clean.npy' % (save_file_prefix, i)))
        pbb = np.load(pbb_name)
        pbb = np.array(pbb[pbb[:, 0] > 0])
        pbb = tools.nms(pbb, 0.1)
        log.info('Detection Results according to confidence')
        for idx in range(pbb.shape[0]):
            z, x, y = int(pbb[idx, 1]), int(pbb[idx, 2]), int(pbb[idx, 3])
            #     print z,x,y
            dat0 = np.array(raw_data[0, z, :, :])
            dat0[max(0, x - 10):min(dat0.shape[0], x + 10), max(0, y - 10)] = 255
            dat0[max(0, x - 10):min(dat0.shape[0], x + 10), min(dat0.shape[1], y + 10)] = 255
            dat0[max(0, x - 10), max(0, y - 10):min(dat0.shape[1], y + 10)] = 255
            dat0[min(dat0.shape[0], x + 10), max(0, y - 10):min(dat0.shape[1], y + 10)] = 255
            plt.imshow(dat0, 'gray')
            plt.show()
            print(pbb[idx])
        pass


def full_pipeline():
    for i in range(0, 50):
        if not os.path.exists(os.path.join(test_dir, '%d.nii.gz' % i)):
            continue
        try:
            case_pixels, origin, spacing = read_nii(i)
            mask, mask1, mask2 = make_mask(case_pixels, origin, spacing)
            preprocess(mask, mask1, mask2, i)
        except Exception as e:
            log.error("Fail to convert %d" % i)
    detector()
    # detect_test.show_result(preprocess_file, save_file_prefix)
    corp()
    cls()


def valid_int(i):
    if i is None:
        return 5
    f = float(i)
    if math.isnan(f):
        return 5
    return int(f)

def check_with_doctor():
    col_names = ['seriesuid', 'coordX', 'coordY', 'coordZ', 'diameter_mm', 'malignant']
    doc_col_names = ['seriesuid', 'coordX', 'coordY', 'coordZ', 'diameter_mm', 'malignant', 'd1', 'd2', 'd3', 'd4']

    _, transform_test = nodcls.get_transform()

    csv_file = os.path.join(file.get_cls_data_path(), 'annotationdetclsconvfnl_v3.csv')
    csv_data = pd.read_csv(csv_file, names=col_names)

    doc_csv_file = os.path.join(file.get_cls_data_path(), 'annotationdetclssgm_doctor.csv')
    doc_csv_data = pd.read_csv(doc_csv_file, names=doc_col_names)

    pid_map = {}
    id_l = doc_csv_data.seriesuid.tolist()[1:]
    d1 = doc_csv_data.d1.tolist()[1:]
    d2 = doc_csv_data.d2.tolist()[1:]
    d3 = doc_csv_data.d3.tolist()[1:]
    d4 = doc_csv_data.d4.tolist()[1:]
    for idx in range(len(id_l)):
        fname = id_l[idx]
        sc1 = valid_int(d1[idx])
        sc2 = valid_int(d2[idx])
        sc3 = valid_int(d3[idx])
        sc4 = valid_int(d4[idx])
        sc = [0, sc1, sc2, sc3, sc4]
        arr = []
        if fname not in pid_map:
            pid_map[fname] = []
        pid_map[fname].append(sc)

    id_l = csv_data.seriesuid.tolist()[1:]
    x_l = csv_data.coordX.tolist()[1:]
    y_l = csv_data.coordY.tolist()[1:]
    z_l = csv_data.coordZ.tolist()[1:]
    d_l = csv_data.diameter_mm.tolist()[1:]
    m_l = csv_data.malignant.tolist()[1:]

    args = env.get_args()
    net, _, _ = nodcls.get_net(args)
    net.eval()
    log.info('Net loaded')

    gbm = GradientBoostingClassifier(random_state=0)
    trainfeat = np.load(os.path.join(cls_resources_dir, 'train_feat.npy'))
    trainlabel = np.load(os.path.join(cls_resources_dir, 'train_label.npy'))
    gbm.fit(trainfeat, trainlabel)
    log.info('GBM loaded')

    soft = nn.Softmax(dim=-1)

    for idx in range(len(id_l)):
        fname = id_l[idx]
        pid = fname.split('-')[0]

        zz = float(x_l[idx])
        xx = float(y_l[idx])
        yy = float(z_l[idx])
        dd = float(d_l[idx])
        mm = int(float(m_l[idx]))

        if pid not in pid_map:
            log.error('Cant find score of %s' % fname)
            continue

        sc = pid_map[pid].pop(0)

        subset_num = file.get_subset_num(pid)
        log.info('Handling %s. Fold num %d. Score: %s' % (pid, subset_num, sc))
        if subset_num != args.cls_test_fold_num:
            continue
        if mm != 1:
            continue

        wrong = 0
        for i in range(1, 5):
            if sc[i] < 3:
                wrong = i
                break

        # crop raw pixel as feature
        corp_file = os.path.join(file.get_cls_corp_path(), '%s.npy' % fname)
        if not os.path.exists(corp_file):
            log.error('Cant find corp for %s' % fname)
            continue
        data = np.load(corp_file)
        bgx = data.shape[0] // 2 - corp_size // 2
        bgy = data.shape[1] // 2 - corp_size // 2
        bgz = data.shape[2] // 2 - corp_size // 2
        data = np.array(data[bgx:bgx + corp_size, bgy:bgy + corp_size, bgz:bgz + corp_size])
        feat = np.hstack((np.reshape(data, (-1,)) / 255, float(dd)))

        tefnamelst = ['%s.npy' % fname]
        telabellst = [mm]
        tefeatlst = [feat]

        testset = lunanod(file.get_cls_corp_path(),
                          tefnamelst, telabellst, tefeatlst,
                          train=False, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False,
                                                  num_workers=args.workers)

        correct = 0
        prob = 0
        with torch.no_grad():
            test_size = len(test_loader.dataset)
            testfeat = np.zeros((test_size, 2560 + corp_size * corp_size * corp_size + 1))
            testlabel = np.zeros((test_size,))
            test_idx = 0
            for batch_idx, (inputs, targets, feat) in enumerate(test_loader):
                inputs, targets = inputs.cuda(), targets.cuda()
                inputs, targets = Variable(inputs), Variable(targets)
                outputs, dfeat = net(inputs)
                # add feature into the array
                testfeat[test_idx:test_idx + len(targets), :2560] = np.array(dfeat.data.cpu().numpy())
                for i in range(len(targets)):
                    testfeat[test_idx + i, 2560:] = np.array(Variable(feat[i]).data.cpu().numpy())
                    testlabel[test_idx + i] = np.array(targets[i].data.cpu().numpy())
                test_idx += len(targets)
                _, predicted = torch.max(outputs.data, 1)
                correct += predicted.eq(targets.data).cpu().sum()
                prob_list = soft(outputs)
                prob = prob_list[0][1]

        coord = np.asarray([xx, yy, zz])
        clean_file_name = file.get_clean_file_path_name(pid)

        mhd_file_name = file.get_mhd_file_path_name(pid)
        slice_img, origin, spacing, is_flip = tools.load_itk_image(mhd_file_name)

        mask_file_name = file.get_mask_file_path_name(pid)
        mask_img = np.load(mask_file_name)

        space_file_name = file.get_space_file_path_name(pid)
        spacing = np.load(space_file_name)

        origin_file_name = file.get_origin_file_path_name(pid)
        origin = np.load(origin_file_name)

        extend_box_file_name = file.get_extend_file_path_name(pid)
        extend_box = np.load(extend_box_file_name)

        label_file_name = file.get_label_file_path_name(pid)
        label_data = np.load(label_file_name, allow_pickle=True)

        x = int(xx)
        y = int(yy)
        z = int(zz)

        preprocess_file = np.load(clean_file_name)
        #     print z,x,y
        dat0 = np.array(preprocess_file[0, z, :, :])
        dat0[max(0, x - 10):min(dat0.shape[0], x + 10), max(0, y - 10)] = 255
        dat0[max(0, x - 10):min(dat0.shape[0], x + 10), min(dat0.shape[1], y + 10)] = 255
        dat0[max(0, x - 10), max(0, y - 10):min(dat0.shape[1], y + 10)] = 255
        dat0[min(dat0.shape[0], x + 10), max(0, y - 10):min(dat0.shape[1], y + 10)] = 255
        plt.imshow(dat0, 'gray')
        png_dir = os.path.join(cls_resources_dir, 'png')
        if not os.path.exists(png_dir):
            os.mkdir(png_dir)
        img_filename = os.path.join(png_dir, '%s_%.2f_%d.png' % (pid, prob, wrong))
        plt.savefig(img_filename)
        plt.show()


if __name__ == '__main__':
    detector()
