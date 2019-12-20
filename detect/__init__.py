import os
import shutil
import time
from collections import OrderedDict

import numpy as np
import torch
from torch.backends import cudnn
from torch.utils.data import DataLoader

from detect import netdef, data
from detect.data.dataset import DataBowl3Detector
from detect.data.split_combo import SplitComb
from utils import gpu, env
from utils.log import get_logger

from torch.autograd import Variable

log = get_logger(__name__)


def get_file_list(args):
    luna_data_dir = env.get('luna_data')
    train_files = []
    test_files = []
    train_dir_count = 0
    test_dir_count = 0
    for subset in os.listdir(luna_data_dir):
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
        subset_dir = os.path.join(luna_data_dir, subset)
        for f in os.listdir(subset_dir):
            if not f.endswith('.mhd'):  # TODO 没有抄黑名单
                continue
            file_name = os.path.join(subset, f[:-4])
            arr_to_append.append(file_name)
    log.info('Files for train: %s. Files for test: %s' % (len(train_files), len(test_files)))
    return train_files, test_files


def get_train_loader(args, net_config):
    train_files, test_files = get_file_list(args)
    data_dir = env.get('preprocess_result_path')

    dataset = DataBowl3Detector(
        data_dir,
        train_files,
        net_config,
        phase='train')
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True)

    return train_loader


def get_val_loader(args, net_config):
    train_files, test_files = get_file_list(args)
    data_dir = env.get('preprocess_result_path')

    dataset = DataBowl3Detector(
        data_dir,
        test_files,
        net_config,
        phase='val')
    val_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True)

    return val_loader


def get_learning_rate(args, epoch):
    lr = args.learning_rate
    if epoch <= args.epochs * 1 / 3:
        lr = lr
    elif epoch <= args.epochs * 2 / 3:
        lr = 0.1 * lr
    elif epoch <= args.epochs * 0.8:
        lr = 0.05 * lr
    else:
        lr = 0.01 * lr
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
        shutil.rmtree(save_dir, True)  # 不继续 将之前的记录删掉
        return
    resume_epoch = args.resume_epoch
    if resume_epoch == -1:
        file_list = os.listdir(save_dir)
        vali_file_list = [f for f in file_list if f.endswith('.ckpt')]
        vali_file_list.sort()
        if len(vali_file_list) > 0:
            last_file_name = vali_file_list[len(vali_file_list) - 1]
            resume_epoch = int(last_file_name[:-5])
    file_name = get_save_file_name(save_dir, resume_epoch)
    args.start_epoch = resume_epoch
    if os.path.exists(file_name):
        checkpoint = torch.load(file_name)
        new_state_dict = OrderedDict()
        for k, v in checkpoint['state_dict'].items():
            name = 'module.%s' % k  # add `module.`
            new_state_dict[name] = v
        net.load_state_dict(new_state_dict)
    else:
        log.info('No saved file. ID: %s. Epoch: %s' % (args.id, resume_epoch))


def common_init(args):
    gpu.set_gpu(args.gpu)
    torch.manual_seed(0)
    model = netdef.get_model(args.model)
    config, net, loss, get_pbb = model.get_model()
    loss = loss.cuda()
    cudnn.benchmark = False
    try_resume(net, args)
    log.info("we have %s GPUs" % torch.cuda.device_count())
    return config, net, loss, get_pbb


def run_train():
    args = env.get_args()
    config, net, loss, get_pbb = common_init(args)
    train_loader = get_train_loader(args, config)
    val_loader = get_val_loader(args, config)
    optimizer = torch.optim.SGD(
        net.parameters(),
        args.learning_rate,
        momentum=0.9,
        weight_decay=args.weight_decay)

    for epoch in range(max(args.start_epoch, 1), args.epochs + 1):  # 跑完所有的epoch
        train(train_loader, net, loss, epoch, optimizer, args)
        validate(val_loader, net, loss)


def train(data_loader, net, loss, epoch, optimizer, args):  # 跑一个epoch
    save_freq = args.save_freq
    save_dir = get_save_dir(args)
    start_time = time.time()

    net.train()
    lr = get_learning_rate(args, epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    metrics = []

    for i, (data, target, coord) in enumerate(data_loader):
        data = Variable(data.cuda())
        target = Variable(target.cuda())
        coord = Variable(coord.cuda())

        output = net(data, coord)
        loss_output = loss(output, target)
        optimizer.zero_grad()
        loss_output[0].backward()
        optimizer.step()

        loss_output[0] = loss_output[0].item()  # loss_output[0] = loss_output[0].data[0]

        if i % 10 == 0:
            log.info('Loss_output. File index [%d] loss %2.4f, classify loss %2.4f, regress loss %2.4f, %2.4f, %2.4f, %2.4f' %
                     (i, loss_output[0], loss_output[1], loss_output[2], loss_output[3], loss_output[4], loss_output[5]))

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
        data = Variable(data.cuda(), volatile=True)
        target = Variable(target.cuda(), volatile=True)
        coord = Variable(coord.cuda(), volatile=True)

        output = net(data, coord)
        loss_output = loss(output, target, train=False)

        loss_output[0] = loss_output[0].item()  # loss_output[0] = loss_output[0].data[0]
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


def get_test_loader(args, net_config):
    train_files, test_files = get_file_list(args)
    data_dir = env.get('preprocess_result_path')

    split_combo = SplitComb(net_config['side_len'], net_config['max_stride'], net_config['stride'],
                            net_config['margin'], net_config['pad_value'])
    dataset = DataBowl3Detector(
        data_dir,
        test_files,
        net_config,
        phase='test',
        split_combo=split_combo)
    test_loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=data.collate,
        pin_memory=False)

    return test_loader


def run_test():
    args = env.get_args()
    config, net, loss, get_pbb = common_init(args)

    test(get_test_loader(args, config), net, get_pbb, args, config)


def test(data_loader, net, get_pbb, args, net_config):
    net_save_dir = get_save_dir(args)
    save_dir = os.path.join(net_save_dir, 'bbox')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    net.eval()
    namelist = []
    split_combo = data_loader.dataset.split_combo
    for i_name, (data, target, coord, nzhw, nzhw2) in enumerate(data_loader):
        print(111111, nzhw)  # 9 8 10
        print(222222, nzhw2)  # 3 2 3

        target = [np.asarray(t, np.float32) for t in target]
        lbb = target[0]
        nzhw = nzhw[0]
        name = data_loader.dataset.img_file_names[i_name].split('/')[-1].split('_clean')[
            0]  # .split('-')[0]  wentao change
        data = data[0][0]
        coord = coord[0][0]
        # data2 = data2[0]
        # coord2 = coord2[0]

        # print(333333,data2.shape)  # 1 300 256 332
        # print(444444,coord2.shape) # 3 75 64 83

        isfeat = False
        if 'output_feature' in net_config:
            if net_config['output_feature']:
                isfeat = True
        n_per_run = args.n_test
        splitlist = list(range(0, len(data) + 1, n_per_run))  # python23 range
        if splitlist[-1] != len(data):
            splitlist.append(len(data))
        outputlist = []
        featurelist = []

        for i in range(len(splitlist) - 1):
            # input = Variable(data[splitlist[i]:splitlist[i+1]], volatile = True).cuda()
            # inputcoord = Variable(coord[splitlist[i]:splitlist[i+1]], volatile = True).cuda()
            input = data[splitlist[i]:splitlist[i + 1]]
            input = input.type(torch.FloatTensor).cuda()
            inputcoord = coord[splitlist[i]:splitlist[i + 1]].cuda()
            print(input.shape)
            if isfeat:
                output, feature = net(input, inputcoord)
                featurelist.append(feature.data.cpu().numpy())
            else:
                output = net(input, inputcoord)
            outputlist.append(output.data.cpu().numpy())
        output = np.concatenate(outputlist, 0)
        output = split_combo.combine(output, nzhw=nzhw)
        if isfeat:
            feature = np.concatenate(featurelist, 0).transpose([0, 2, 3, 4, 1])[:, :, :, :, :, np.newaxis]
            feature = split_combo.combine(feature, net_config['side_len'])[..., 0]

        thresh = args.testthresh  # -8 #-3
        pbb, mask = get_pbb(output, thresh, is_mask=True)
        if isfeat:
            feature_selected = feature[mask[0], mask[1], mask[2]]
            np.save(os.path.join(save_dir, name + '_feature.npy'), feature_selected)
        np.save(os.path.join(save_dir, name + '_pbb.npy'), pbb)
        np.save(os.path.join(save_dir, name + '_lbb.npy'), lbb)
    np.save(os.path.join(save_dir, 'namelist.npy'), namelist)


if __name__ == '__main__':
    log.info(run_train())
