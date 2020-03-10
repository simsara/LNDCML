import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from detect import netdef, data
from detect import test
from detect.data.dataset import DataBowl3Detector
from detect.data.split_combine import SplitCombine
from prepare import *
from utils import tools, gpu

slice_plt = 70
test_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'test_resources')
save_file_prefix = 'detect_preprocess'


# 生成mask
def make_mask(raw_file_path):
    case_pixels, origin, spacing, isflip = tools.load_itk_image(raw_file_path)
    # log.info('case_pixels : ',case_pixels)
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


# 预处理生成npy文件
def preprocess(filepath, mask, mask1, mask2):
    resolution = np.array([1, 1, 1])
    sliceim, origin, spacing, isflip = tools.load_itk_image(filepath)  # 加载原始数据
    if isflip:
        mask = mask[:, ::-1, ::-1]
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

    if isflip:
        sliceim = sliceim[:, ::-1, ::-1]
        log.info('flip!')

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

    np.save(os.path.join(test_dir, "%s_clean.npy" % save_file_prefix), sliceim)

    return sliceim


def init_net(args):
    gpu.set_gpu(args.gpu)
    torch.manual_seed(0)
    model = netdef.get_model(args.model)
    config, net, loss, get_pbb = model.get_model()
    checkpoint = torch.load(os.path.join(test_dir, 'test.ckpt'))
    net.load_state_dict(checkpoint['state_dict'])
    net = torch.nn.DataParallel(net).cuda()  # 使用多个GPU进行训练
    loss = loss.cuda()
    log.info("we have %s GPUs" % torch.cuda.device_count())
    return config, net, loss, get_pbb


# 执行检测，生成结果npy文件
def detector(file_name):
    args = env.get_args()
    log.info(args)
    net_config, net, loss, get_pbb = init_net(args)

    testdatadir = test_dir  # 预处理结果路径
    testfilelist = [file_name]  # 文件名列表

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
    test(test_loader, net, get_pbb, args, net_config, test_dir)


def show_result(preprocess_file, file_prefix):
    pbb = np.load(os.path.join(test_dir, '%s_pbb.npy' % file_prefix))
    pbb = np.array(pbb[pbb[:, 0] > 0])
    pbb = tools.nms(pbb, 0.1)
    log.info('Detection Results according to confidence')
    for idx in range(pbb.shape[0]):
        log.info('Index: %d. Pbb: %.4f' % (idx, pbb[idx, 0]))
        fig = plt.figure()
        z, x, y = int(pbb[idx, 1]), int(pbb[idx, 2]), int(pbb[idx, 3])
        #     print z,x,y
        dat0 = np.array(preprocess_file[0, z, :, :])
        dat0[max(0, x - 10):min(dat0.shape[0], x + 10), max(0, y - 10)] = 255
        dat0[max(0, x - 10):min(dat0.shape[0], x + 10), min(dat0.shape[1], y + 10)] = 255
        dat0[max(0, x - 10), max(0, y - 10):min(dat0.shape[1], y + 10)] = 255
        dat0[min(dat0.shape[0], x + 10), max(0, y - 10):min(dat0.shape[1], y + 10)] = 255
        plt.imshow(dat0, 'gray')
        plt.show()


def full_pipeline(mhd):
    raw_file = os.path.join(test_dir, mhd)
    log.info('start process ! ')
    mask, mask1, mask2 = make_mask(raw_file)
    preprocess_file = preprocess(raw_file, mask, mask1, mask2)

    detector(save_file_prefix)
    show_result(preprocess_file, save_file_prefix)


def run():
    args = env.get_args()
    full_pipeline(args.prd_file)


if __name__ == '__main__':
    full_pipeline('1.3.6.1.4.1.14519.5.2.1.6279.6001.100225287222365663678666836860.mhd')
