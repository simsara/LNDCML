# 根据网络名称和id获取保存路径
import os

from utils import env


def get_luna_data_path():
    return env.get('luna_data')


def get_preprocess_result_path():
    return env.get('preprocess_result_path')


def get_file_path_name_by_uid(parent_dir, uid, suffix):
    for subset in os.listdir(parent_dir):
        if not subset.startswith('subset'):
            continue
        subset_dir = os.path.join(parent_dir, subset)
        for f in os.listdir(subset_dir):
            if not f.endswith(suffix):
                continue
            if f.startswith(uid):
                return os.path.join(subset_dir, f)
    return None


def get_mhd_file_path_name(uid):
    return get_file_path_name_by_uid(get_luna_data_path(), uid, '.mhd')


def get_origin_file_path_name(uid):
    return get_file_path_name_by_uid(get_preprocess_result_path(), uid, '_origin.npy')


def get_space_file_path_name(uid):
    return get_file_path_name_by_uid(get_preprocess_result_path(), uid, '_spacing.npy')


def get_extend_file_path_name(uid):
    return get_file_path_name_by_uid(get_preprocess_result_path(), uid, '_extendbox.npy')


def get_mask_file_path_name(uid):
    return get_file_path_name_by_uid(get_preprocess_result_path(), uid, '_mask.npy')


def get_net_save_dir(args):
    parent_dir = env.get('net_save_dir')
    if not os.path.exists(parent_dir):
        os.mkdir(parent_dir)
    net_dir = os.path.join(parent_dir, args.model)
    if not os.path.exists(net_dir):
        os.mkdir(net_dir)
    net_id_dir = os.path.join(net_dir, args.id)
    if not os.path.exists(net_id_dir):
        os.mkdir(net_id_dir)
    return net_id_dir


def get_net_save_file_path_name(save_dir, epoch):
    return os.path.join(save_dir, '%03d.ckpt' % epoch)


def get_net_bbox_save_path(args):
    net_save_dir = get_net_save_dir(args)
    save_dir = os.path.join(net_save_dir, 'bbox')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    return save_dir


def get_predanno_file_name(args, detp_thresh):
    return os.path.join(get_net_bbox_save_path(args), 'predanno%sd3.csv' % str(detp_thresh))
