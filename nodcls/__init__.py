import os
import random
import shutil
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import GradientBoostingClassifier
from torch import nn, optim
from torch.autograd import Variable

from nodcls import transforms
from nodcls.dataloader import lunanod
from nodcls.models.dpn3d import DPN92_3D
from utils import file, gpu, env
from utils.log import get_logger

log = get_logger(__name__)
corp_size = 32
col_names = ['seriesuid', 'coordX', 'coordY', 'coordZ', 'diameter_mm', 'malignant']


def preprocess():
    csv_file = os.path.join(file.get_cls_data_path(), 'annotationdetclsconvfnl_v3.csv')
    csv_data = pd.read_csv(csv_file, names=col_names)

    id_l = csv_data.seriesuid.tolist()[1:]
    x_l = csv_data.coordX.tolist()[1:]
    y_l = csv_data.coordY.tolist()[1:]
    z_l = csv_data.coordZ.tolist()[1:]
    d_l = csv_data.diameter_mm.tolist()[1:]
    m_l = csv_data.malignant.tolist()[1:]

    save_path = file.get_cls_corp_path()
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    for idx in range(len(id_l)):
        fname = id_l[idx]
        pid = fname.split('-')[0]
        crdx = int(float(x_l[idx]))
        crdy = int(float(y_l[idx]))
        crdz = int(float(z_l[idx]))
        dim = int(float(d_l[idx]))
        clean_file_name = file.get_clean_file_path_name(pid)
        data = np.load(clean_file_name)
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


def cal_mean_and_std():
    # Cal mean std
    corp_path = file.get_cls_corp_path()
    pixvlu, npix = 0, 0
    for fname in os.listdir(corp_path):
        if fname.endswith('.npy'):
            data = np.load(os.path.join(corp_path, fname))
            pixvlu += np.sum(data)
            npix += np.prod(data.shape)  # 连乘操作
    pixmean = pixvlu / float(npix)
    pixvlu = 0
    for fname in os.listdir(corp_path):
        if fname.endswith('.npy'):
            data = np.load(os.path.join(corp_path, fname)) - pixmean
            pixvlu += np.sum(data * data)
    pixstd = np.sqrt(pixvlu / float(npix))
    log.info('mean ' + str(pixmean) + ' std ' + str(pixstd))
    return pixmean, pixstd


def get_transform():
    pixmean, pixstd = cal_mean_and_std()
    transform_train = transforms.Compose([
        # transforms.RandomScale(range(28, 38)),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomYFlip(),
        transforms.RandomZFlip(),
        transforms.ZeroOut(4),
        transforms.ToTensor(),
        transforms.Normalize(pixmean, pixstd),  # need to cal mean and std, revise norm func
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(pixmean, pixstd),
    ])
    return transform_train, transform_test


def get_file_list():
    # load data list
    trfnamelst = []
    trlabellst = []
    trfeatlst = []

    tefnamelst = []
    telabellst = []
    tefeatlst = []

    csv_file = os.path.join(file.get_cls_data_path(), 'annotationdetclsconvfnl_v3.csv')
    dataframe = pd.read_csv(csv_file, names=col_names)
    alllst = dataframe['seriesuid'].tolist()[1:]
    labellst = dataframe['malignant'].tolist()[1:]
    crdxlst = dataframe['coordX'].tolist()[1:]
    crdylst = dataframe['coordY'].tolist()[1:]
    crdzlst = dataframe['coordZ'].tolist()[1:]
    dimlst = dataframe['diameter_mm'].tolist()[1:]
    # test id
    test_id_list = []

    foldnum = random.randint(0, 9)
    log.info('Using fold %d to test' % foldnum)
    subset_path = os.path.join(file.get_luna_data_path(), 'subset%d' % foldnum)
    for fname in os.listdir(subset_path):
        if fname.endswith('.mhd'):
            test_id_list.append(fname[:-4])
    
    log.info('Size of test id: %d' % len(test_id_list))
    mxx = mxy = mxz = mxd = 0
    for srsid, label, x, y, z, d in zip(alllst, labellst, crdxlst, crdylst, crdzlst, dimlst):
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
        if srsid.split('-')[0] in test_id_list:
            tefnamelst.append(srsid + '.npy')
            telabellst.append(int(label))
            tefeatlst.append(feat)
        else:
            trfnamelst.append(srsid + '.npy')
            trlabellst.append(int(label))
            trfeatlst.append(feat)

    for idx in range(len(trfeatlst)):
        trfeatlst[idx][-1] /= mxd
    for idx in range(len(tefeatlst)):
        tefeatlst[idx][-1] /= mxd
    log.info(
        '[Existed] Size of train files: %d. Size of test files: %d.' % (len(trfnamelst), len(tefnamelst)))  # 912 92

    return trfnamelst, trlabellst, trfeatlst, tefnamelst, telabellst, tefeatlst


def get_loader(args):
    train_file_size = args.cls_train_file_size
    preprocesspath = file.get_cls_corp_path()
    transform_train, transform_test = get_transform()
    trfnamelst, trlabellst, trfeatlst, tefnamelst, telabellst, tefeatlst = get_file_list()
    if train_file_size == -1 or train_file_size > len(trfnamelst):
        train_file_size = len(trfnamelst)
    trainset = lunanod(preprocesspath,
                       trfnamelst[0:train_file_size], trlabellst[0:train_file_size], trfeatlst[0:train_file_size],
                       train=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                              shuffle=True, num_workers=args.workers)

    testset = lunanod(preprocesspath, tefnamelst, telabellst, tefeatlst, train=False, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                             shuffle=False, num_workers=args.workers)
    log.info('[Used] Size of train files: %d. Size of test files: %d.' % (len(trainset), len(testset)))
    return trainloader, testloader


def try_resume(net, args, para: bool = False):
    save_dir = file.get_cls_net_save_dir(args)
    if args.resume == 0:
        args.start_epoch = 0
        shutil.rmtree(save_dir, True)  # 不继续 将之前的记录删掉
        return
    start_epoch = args.start_epoch
    if start_epoch == -1:
        file_list = os.listdir(save_dir)
        vali_file_list = [f for f in file_list if f.endswith('.ckpt')]
        vali_file_list.sort()
        if len(vali_file_list) > 0:
            last_file_name = vali_file_list[len(vali_file_list) - 1]
            start_epoch = int(last_file_name[:-5])
    file_name = file.get_cls_net_save_file_path_name(args, start_epoch)
    args.start_epoch = start_epoch
    if os.path.exists(file_name):
        log.info('Resuming model from: %s' % file_name)
        checkpoint = torch.load(file_name)
        if para:
            new_state_dict = OrderedDict()
            for k, v in checkpoint['state_dict'].items():
                name = 'module.%s' % k  # add `module.`
                new_state_dict[name] = v
            net.load_state_dict(new_state_dict)
        else:
            net.load_state_dict(checkpoint['state_dict'])
    else:
        log.info('No saved file. ID: %s. Epoch: %s' % (args.id, start_epoch))


def get_learning_rate(epoch, max_epoch):
    if epoch < 0.5 * max_epoch:
        lr = 0.01  # args.lr
    elif epoch < 0.8 * max_epoch:
        lr = 0.001
    else:
        lr = 0.0001
    return lr


def get_net(args):
    gpu.set_gpu(args.gpu)
    net = DPN92_3D()
    try_resume(net, args)
    net = torch.nn.DataParallel(net).cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=5e-4)
    return net, criterion, optimizer


def run_train():
    args = env.get_args()
    train_loader, test_loader = get_loader(args)
    net, loss, opt = get_net(args)
    save_dir = file.get_cls_net_save_dir(args)

    best_acc = 0  # best test accuracy
    best_acc_gbt = 0

    for epoch in range(max(args.start_epoch + 1, 1), args.epochs + 1):  # 跑完所有的epoch
        m = GradientBoostingClassifier(max_depth=1, random_state=0)
        train(net, loss, opt, train_loader, epoch, args.epochs, m)
        acc, gbt = test(net, loss, test_loader, m)

        if acc > best_acc:
            best_acc = acc

        if gbt > best_acc_gbt:
            state_dict = net.module.state_dict()
            for key in state_dict.keys():
                state_dict[key] = state_dict[key].cpu()
            torch.save({
                'epoch': epoch,
                'save_dir': save_dir,
                'state_dict': state_dict,
                'args': args,
                'acc': acc},
                file.get_cls_net_save_file_path_name(args, epoch))
            log.info('Saved epoch %d' % epoch)
            best_acc_gbt = gbt
    pass


def train(net, criterion, optimizer, train_loader, epoch, max_epoch, m):
    train_size = len(train_loader.dataset)
    net.train()
    lr = get_learning_rate(epoch, max_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    log.info('Training epoch: %d. lr: %.4f' % (epoch, lr))
    train_loss = 0
    correct = 0
    total = 0
    trainfeat = np.zeros((train_size, 2560 + corp_size * corp_size * corp_size + 1))
    trainlabel = np.zeros((train_size,))
    idx = 0
    for batch_idx, (inputs, targets, feat) in enumerate(train_loader):
        inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        inputs, targets = Variable(inputs, requires_grad=True), Variable(targets)
        outputs, dfeat = net(inputs)
        # add feature into the array
        trainfeat[idx:idx + len(targets), :2560] = np.array(dfeat.data.cpu().numpy())  # [4,2560]
        for i in range(len(targets)):
            trainfeat[idx + i, 2560:] = np.array(Variable(feat[i]).data.cpu().numpy())
            trainlabel[idx + i] = np.array(targets[i].data.cpu().numpy())
        idx += len(targets)

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
    accout = round(correct.data.cpu().numpy() / total, 4)

    m.fit(trainfeat, trainlabel)
    gbttracc = round(np.mean(m.predict(trainfeat) == trainlabel), 4)
    log.info('Train Loss: %.3f | Acc: %.3f%% (%d/%d) | Gbt: %.3f' % (train_loss / (batch_idx + 1), 100. * accout,
                                                                     correct, total, gbttracc))
    return m


def test(net, criterion, test_loader, m):
    test_size = len(test_loader.dataset)
    net.eval()
    accout, gbtteacc = 0, 0
    with torch.no_grad():
        test_loss = 0
        correct = 0
        total = 0
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

            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()
        accout = round(correct.data.cpu().numpy() / total, 4)
        gbtteacc = round(np.mean(m.predict(testfeat) == testlabel), 4)
        log.info('Test Loss: %.3f | Acc: %.3f%% (%d/%d) | Gbt: %.3f' % (test_loss / (batch_idx + 1), 100. * accout,
                                                                        correct, total, gbtteacc))
    return accout, gbtteacc


if __name__ == '__main__':
    preprocess()
