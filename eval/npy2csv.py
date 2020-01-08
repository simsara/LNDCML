import csv
import os
from concurrent.futures import Future

import numpy as np

from eval.CADevaluation import nodule_cad_evaluation
from eval.csv_label import pbb_csv_header
from utils import file
from utils.log import get_logger
from utils.threadpool import pool
from utils.tools import VoxelToWorldCoord, load_itk_image, nms

log = get_logger(__name__)

resolution = np.array([1, 1, 1])  # 分辨率 TODO 怎么跟网络保持一致
nms_thresh = 0.1  # 非极大值抑制的阈值设置


def convert_csv(bbox_name, bbox_path, detp):  # 给定pbb.npy的文件名，路径，阈值
    """
    对输出结节应用阈值和nms，输出的结果再转换为label一样的坐标体系，存在一个csv文件中
    """
    uid = bbox_name[:-8]
    mhd_file_name = file.get_mhd_file_path_name(uid)
    origin_file_name = file.get_origin_file_path_name(uid)
    space_file_name = file.get_space_file_path_name(uid)
    extend_file_name = file.get_extend_file_path_name(uid)
    sliceim, origin, spacing, is_flip = load_itk_image(mhd_file_name)  # 得到对应subset的原始数据
    origin = np.load(origin_file_name, mmap_mode='r')  # 得到对应subset预处理后的坐标原点
    spacing = np.load(space_file_name, mmap_mode='r')  # 体素间隙
    extendbox = np.load(extend_file_name, mmap_mode='r')  # 肺实质的拓展box
    pbb = np.load(os.path.join(bbox_path, bbox_name), mmap_mode='r')  # pbb.npy文件

    pbbold = np.array(pbb[pbb[:, 0] > detp])  # 根据阈值过滤掉概率低的
    pbbold = np.array(pbbold[pbbold[:, -1] > 3])  # add new 9 15 根据半径过滤掉小于3mm的

    pbb = nms(pbbold, nms_thresh)  # 对输出的结节进行nms
    pbb = np.array(pbb[:, :-1])  # 去掉直径

    # 对输出加上拓展box的坐标，其实就是恢复为原来的坐标
    pbb[:, 1:] = np.array(pbb[:, 1:] + np.expand_dims(extendbox[:, 0], 1).T)  # TODO
    # 将输出恢复为原来的分辨率，这样就对应了原始数据中的体素坐标
    pbb[:, 1:] = np.array(pbb[:, 1:] * np.expand_dims(resolution, 1).T / np.expand_dims(spacing, 1).T)  # TODO

    if is_flip:  # 如果有翻转，将坐标翻转回去
        mask_file_name = file.get_mask_file_path_name(uid)
        mask = np.load(mask_file_name, mmap_mode='r')  # 得到对应subset的mask
        pbb[:, 2] = mask.shape[1] - pbb[:, 2]
        pbb[:, 3] = mask.shape[2] - pbb[:, 3]
    pos = VoxelToWorldCoord(pbb[:, 1:], origin, spacing)  # 将输出转化为世界坐标

    row_list = []
    for nk in range(pos.shape[0]):  # 每一个结节：文件名，z,y,x，是结节的概率(经过sigmoid处理)
        row_list.append([uid, pos[nk, 2], pos[nk, 1], pos[nk, 0], 1 / (1 + np.exp(-pbb[nk, 0]))])
    log.info('[%s] Done' % uid)
    return row_list


def get_csv(args):  # 给定阈值
    """
    对输出的结果文件调用convert_csv函数处理
    每一个epoch生成一个csv文件，存放80多个测试病例的预测结节位置及概率
    """
    for ep in range(args.start_epoch, args.epochs + 1):  # 对每一个epoch
        if not epoch_exists(args, ep):
            continue
        bbox_path = file.get_net_bbox_save_path(args, ep)
        log.info('bbox path: %s' % bbox_path)
        for detp_thresh in args.eval_detp:
            save_file_name = file.get_predanno_file_name(args, ep, detp_thresh)
            log.info('ep: %d. detp: %3.2f. file: %s' % (ep, detp_thresh, save_file_name))
            f = open(save_file_name, 'w', newline='')
            file_writer = csv.writer(f)
            file_writer.writerow(pbb_csv_header)  # 写入的第一行为 用户id 结节坐标x,y,z 结节概率
            pbb_list = []
            for file_name in os.listdir(bbox_path):  # bboxpath目录下的所有文件和文件夹
                if file_name.endswith('_pbb.npy'):  # 找到以_pbb.npy结尾的文件(结节概率文件)，添加进文件列表
                    pbb_list.append(file_name)

            future_list = []
            result_list = []
            for pbb_file_name in pbb_list:
                if args.multi_process == 1:
                    future_list.add(pool.submit(convert_csv, bbox_name=pbb_file_name, bbox_path=bbox_path, detp=detp_thresh))
                else:
                    result_list.append(convert_csv(bbox_name=pbb_file_name, bbox_path=bbox_path, detp=detp_thresh))

            for future in future_list:  # type: Future
                predanno = future.result()
                result_list.append(predanno)

            for result in result_list:
                for row in result:
                    file_writer.writerow(row)

            f.close()
            log.info('Finished ep: %d. detp: %3.2f' % (ep, detp_thresh))


def get_froc_value(predanno_filename, output_dir, uid_list):
    annotations = file.get_luna_csv_name('annotations.csv')
    annotation_exclude = file.get_luna_csv_name('annotations_excluded.csv')

    return nodule_cad_evaluation(annotations, annotation_exclude, uid_list, predanno_filename, output_dir)


# 每个epoch都会对应一个csv文件，要选取一个最好的结果，选取标准为froc值
def get_froc(args):  # 阈值和epoch
    """
    根据pbb生成的csv
    """
    max_froc = 0
    max_ep = 0
    for ep in range(args.start_epoch, args.epochs + 1):  # 对每个epoch分别处理
        if not epoch_exists(args, ep):
            continue
        uid_list = np.load(file.get_uid_list_filename(args, ep))
        froc_list = []
        for detp_thresh in args.eval_detp:  # 对于阈值列表中的每一个阈值
            predanno = file.get_predanno_file_name(args, ep, detp_thresh)
            output_dir = file.get_eval_save_path(args, ep, detp_thresh)
            froc_list.append(get_froc_value(predanno_filename=predanno, output_dir=output_dir, uid_list=uid_list))

        if max(froc_list) > max_froc:
            max_ep = ep  # 更新maxep
            max_froc = max(froc_list)

        log.info('Epoch: %03d. Froc list: %s' % (ep, froc_list))

        # TODO 跳过了一个看不懂的循环
    log.info('Max froc: %3.10f. Max epoch: %03d' % (max_froc, max_ep))


def epoch_exists(args, epoch) -> bool:
    net = file.get_net_save_file_path_name(args, epoch)
    return os.path.exists(net)
