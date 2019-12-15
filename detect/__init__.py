import os
import time

import numpy as np
import torch
from torch.backends import cudnn
from torch.nn import DataParallel
from torch.utils.data import DataLoader

from detect import netdef
from detect.data.dataset import DataBowl3Detector
from utils import gpu, env
from utils.log import get_logger

log = get_logger(__name__)


def get_file_list(args):
    preprocess_result_path = env.get('luna_data')
    train_files = []
    test_files = []
    train_dir_count = 0
    test_dir_count = 0
    for subset in os.listdir(preprocess_result_path):
        if not subset.startswith('subset'):
            continue
        arr_to_append = None
        if train_dir_count < args.nd_train:
            arr_to_append = train_files
            train_dir_count += 1
        elif test_dir_count < args.nd_test:
            arr_to_append = test_files
            test_dir_count += 1
        if arr_to_append is None:
            break
        subset_dir = os.path.join(preprocess_result_path, subset)
        for f in os.listdir(subset_dir):
            if not f.endswith('.mhd'):  # TODO 没有抄黑名单
                continue
            file_name = os.path.join(subset, f[:-4])
            arr_to_append.append(file_name)
    log.info('Files for train: %s. Files for test: %s' % (len(train_files), len(test_files)))
    return train_files, test_files


def get_data_loader(data_dir, files, args, net_config):
    dataset = DataBowl3Detector(
        data_dir,
        files,
        net_config,
        phase='train')
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True)
    check_data(data_loader)
    log.info('Load file successfully.')
    return data_loader


def get_train_loader(args, net_config):
    train_files, test_files = get_file_list(args)
    data_dir = env.get('preprocess_result_path')
    return get_data_loader(data_dir, train_files, args, net_config)


def get_val_loader(args, net_config):
    train_files, test_files = get_file_list(args)
    data_dir = env.get('preprocess_result_path')
    return get_data_loader(data_dir, test_files, args, net_config)


def check_data(data_loader):
    for i, (data, target, coord) in enumerate(data_loader):  # check data consistency
        pass  # empty check


def get_learning_rate(args, epoch):
    if epoch <= args.epochs * 1 / 3:
        lr = args.lr
    elif epoch <= args.epochs * 2 / 3:
        lr = 0.1 * args.lr
    elif epoch <= args.epochs * 0.8:
        lr = 0.05 * args.lr
    else:
        lr = 0.01 * args.lr
    return lr


# 根据网络名称和id获取保存路径
def get_save_dir(args):
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


def get_save_file_name(save_dir, epoch):
    return os.path.join(save_dir, '%03d.ckpt' % epoch)


def try_resume(net, args):
    args.start_epoch = 1  # 定义开始的epoch
    save_dir = get_save_dir(args)
    if args.resume == 0:
        os.removedirs(save_dir)  # 不继续 将之前的记录删掉
        return
    resume_epoch = args.resume_epoch
    if resume_epoch == -1:
        file_list = os.listdir(save_dir)
        file_list.sort()
        if len(file_list) > 0:
            last_file_name = file_list[len(file_list) - 1]
            resume_epoch = int(last_file_name[:-5])
    file_name = get_save_file_name(save_dir, resume_epoch)
    args.start_epoch = resume_epoch
    if os.path.exists(file_name):
        checkpoint = torch.load(file_name)
        net.load_state_dict(checkpoint['state_dict'])
    else:
        log.info('No saved file. ID: %s. Epoch: %s' % (args.id, resume_epoch))


def common_init(args):
    torch.manual_seed(0)
    torch.cuda.set_device(0)

    model = netdef.get_model(args.model)
    config, net, loss, get_pbb = model.get_model()

    try_resume(net, args)

    n_gpu = gpu.set_gpu(args.gpu)
    args.n_gpu = n_gpu
    net = net.cuda()
    loss = loss.cuda()
    cudnn.benchmark = False
    net = DataParallel(net)
    return config, net, loss, get_pbb


def run_train():
    args = env.get_args()
    config, net, loss, get_pbb = common_init(args)
    train_loader = get_train_loader(args, config)
    val_loader = get_val_loader(args, config)
    optimizer = torch.optim.SGD(
        net.parameters(),
        args.lr,
        momentum=0.9,
        weight_decay=args.weight_decay)

    for epoch in range(max(args.start_epoch, 1), args.epochs + 1):
        train(train_loader, net, loss, epoch, optimizer, args)
        validate(val_loader, net, loss)


def train(data_loader, net, loss, epoch, optimizer, args):
    save_freq = args.save_freq
    save_dir = get_save_dir(args)
    start_time = time.time()

    net.train()
    lr = get_learning_rate(args, epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    metrics = []

    for i, (data, target, coord) in enumerate(data_loader):
        data = data.cuda()
        target = target.cuda()
        coord = coord.cuda()

        output = net(data, coord)
        loss_output = loss(output, target)
        optimizer.zero_grad()
        loss_output[0].backward()
        optimizer.step()

        loss_output[0] = loss_output[0].data[0]
        metrics.append(loss_output)
        log.info('Finish epoch [%d] file [%d]' % (epoch, i))

    if epoch % save_freq == 0:
        state_dict = net.module.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].cpu()

        torch.save({
            'epoch': epoch,
            'save_dir': save_dir,
            'state_dict': state_dict,
            'args': args},
            get_save_file_name(save_dir, epoch))
        log.info('Saved. Epoch [%d]' % epoch)

    end_time = time.time()

    metrics = np.asarray(metrics, np.float32)
    log.info('Epoch %03d (lr %.5f)' % (epoch, lr))
    log.info('Train:      tpr %3.2f, tnr %3.2f, total pos %d, total neg %d, time %3.2f' % (
        100.0 * np.sum(metrics[:, 6]) / np.sum(metrics[:, 7]),
        100.0 * np.sum(metrics[:, 8]) / np.sum(metrics[:, 9]),
        np.sum(metrics[:, 7]),
        np.sum(metrics[:, 9]),
        end_time - start_time))
    log.info('loss %2.4f, classify loss %2.4f, regress loss %2.4f, %2.4f, %2.4f, %2.4f' % (
        np.mean(metrics[:, 0]),
        np.mean(metrics[:, 1]),
        np.mean(metrics[:, 2]),
        np.mean(metrics[:, 3]),
        np.mean(metrics[:, 4]),
        np.mean(metrics[:, 5])))


def validate(data_loader, net, loss):
    start_time = time.time()
    net.eval()
    metrics = []
    for i, (data, target, coord) in enumerate(data_loader):
        data = data.cuda()
        target = target.cuda()
        coord = coord.cuda()

        output = net(data, coord)
        loss_output = loss(output, target, train=False)

        loss_output[0] = loss_output[0].data[0]
        metrics.append(loss_output)
    end_time = time.time()

    metrics = np.asarray(metrics, np.float32)
    log.info('Validation: tpr %3.2f, tnr %3.8f, total pos %d, total neg %d, time %3.2f' % (
        100.0 * np.sum(metrics[:, 6]) / np.sum(metrics[:, 7]),
        100.0 * np.sum(metrics[:, 8]) / np.sum(metrics[:, 9]),
        np.sum(metrics[:, 7]),
        np.sum(metrics[:, 9]),
        end_time - start_time))
    log.info('loss %2.4f, classify loss %2.4f, regress loss %2.4f, %2.4f, %2.4f, %2.4f' % (
        np.mean(metrics[:, 0]),
        np.mean(metrics[:, 1]),
        np.mean(metrics[:, 2]),
        np.mean(metrics[:, 3]),
        np.mean(metrics[:, 4]),
        np.mean(metrics[:, 5])))


if __name__ == '__main__':
    log.info(run_train())
