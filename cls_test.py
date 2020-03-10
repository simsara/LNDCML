import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import GradientBoostingClassifier
from torch.autograd import Variable

from nodcls import get_loader, get_net
from utils import env
from utils.log import get_logger

log = get_logger(__name__)
corp_size = 32
col_names = ['seriesuid', 'coordX', 'coordY', 'coordZ', 'diameter_mm', 'malignant']

args = env.get_args()
log.info(args)
net, criterion, opt = get_net(args)

checkpoint = torch.load('')
net.load_state_dict(checkpoint['state_dict'])
net = torch.nn.DataParallel(net).cuda()

_, test_loader = get_loader(args)
test_size = len(test_loader.dataset)
net.eval()
m = GradientBoostingClassifier(max_depth=1, random_state=0)

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
    df = pd.DataFrame(data=[m.predict(testfeat),testlabel]).T
    df.to_excel('cls_test_output.xls')
    log.info('Test Loss: %.3f | Acc: %.3f%% (%d/%d) | Gbt: %.3f' % (test_loss / (batch_idx + 1), 100. * accout,
                                                                        correct, total, gbtteacc))
