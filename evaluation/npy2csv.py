"""
此文件将输出的.npy格式的结果转化为.csv格式
"""
import numpy as np
import matplotlib
matplotlib.use('agg')  # agg渲染器，文件类型为.png，高质量渲染引擎
import matplotlib.pyplot as plt
from CADevaluation import noduleCADEvaluation
import os
import csv
from multiprocessing import Pool
import functools

from tools import *

fold = 1
#annotations_filename = '' # path for ground truth annotations for the fold
annotations_filename = './annotations/annotations.csv'
#annotations_excluded_filename = '' # path for excluded annotations for the fold
annotations_excluded_filename = './annotations/annotations_excluded.csv'
#seriesuids_filename = '' # path for seriesuid for the fold
seriesuids_filename = './annotations/seriesuids.csv'
results_path = '/home/lwq/netsave/dpn3d26/deeplung/' # val' #val' ft96'+'/val'#
sideinfopath = '/home/lwq/dataset/luna16/data/subset1/'  # subset'+str(fold)+'/'  +str(fold)
datapath = '/home/lwq/dataset/luna16/subset/subset1/'  # subset'+str(fold)+'/'

maxeps = 2  #03 #150 #100
eps = range(1, maxeps + 1,1)  # 6,7,1)#5,151,5)#5,151,5)#76,77,1)#40,41,1)#76,77,1)#1,101,1)#17,18,1)#38,39,1)#1, maxeps+1, 1) #maxeps+1, 1)
detp = [-2,-1]  # , -0.5, 0]#, 0.5, 1]#, 0.5, 1] #range(-1, 0, 1)
isvis = False  # True
nmsthresh = 0.1  # 非极大值抑制的阈值设置
nprocess = 1  # 4 线程的个数
use_softnms = False  # 是否使用soft_nms
frocarr = np.zeros((maxeps, len(detp)))  # TODO 这个不知道
firstline = ['seriesuid', 'coordX', 'coordY', 'coordZ', 'probability']


def convertcsv(bboxfname, bboxpath, detp):  # 给定pbb.npy的文件名，路径，阈值
    """
    对输出结节应用阈值和nms，输出的结果再转换为label一样的坐标体系，存在一个csv文件中
    """
    print(bboxfname)

    sliceim, origin, spacing, isflip = load_itk_image(datapath + bboxfname[:-8] + '.mhd')  # 得到对应subset的原始数据
    origin = np.load(sideinfopath + bboxfname[:-8] + '_origin.npy', mmap_mode='r')  # 得到对应subset预处理后的坐标原点
    spacing = np.load(sideinfopath + bboxfname[:-8] + '_spacing.npy', mmap_mode='r')  # 体素间隙
    resolution = np.array([1, 1, 1])  # 分辨率
    extendbox = np.load(sideinfopath + bboxfname[:-8] + '_extendbox.npy', mmap_mode='r')  # 肺实质的拓展box
    pbb = np.load(bboxpath + bboxfname, mmap_mode='r')  # pbb.npy文件
    print('pbb load : ' + str(pbb.shape))
    print('max origin pbb ' + str(max(pbb[:,0])))
    print('min origin pbb ' + str(min(pbb[:,0])))

    pbbold = np.array(pbb[pbb[:, 0] > detp])  # 根据阈值过滤掉概率低的
    print('pbb after detp: ' + str(pbbold.shape))
    pbbold = np.array(pbbold[pbbold[:, -1] > 3])  # add new 9 15 根据半径过滤掉小于3mm的
    print('pbb after >3 : ' + str(pbbold.shape))
    print('max pbb ' + str(max(pbbold[:,0])))
    print('min pbb ' + str(min(pbbold[:,0])))
    

    #pbbold = pbbold[np.argsort(-pbbold[:, 0])][:10000]  # 取概率值前1000的结节，不然直接进行nms太耗时
    #print("after sort bboxs : ",len(pbbold))
    pbb = nms(pbbold, nmsthresh)  # 对输出的结节进行nms
    print("after nms bboxs : ",len(pbb))
    #print(bboxfname, pbbold.shape, pbb.shape, pbbold.shape)
    pbb = np.array(pbb[:, :-1])  # 去掉直径
    # print pbb[:, 0]

    # 对输出加上拓展box的坐标，其实就是恢复为原来的坐标
    pbb[:, 1:] = np.array(pbb[:, 1:] + np.expand_dims(extendbox[:, 0], 1).T)  # TODO
    # 将输出恢复为原来的分辨率，这样就对应了原始数据中的体素坐标
    pbb[:, 1:] = np.array(pbb[:, 1:] * np.expand_dims(resolution, 1).T / np.expand_dims(spacing, 1).T)  # TODO

    if isflip:  # 如果有翻转，将坐标翻转回去
        Mask = np.load(sideinfopath + bboxfname[:-8] + '_mask.npy', mmap_mode='r')   # 得到对应subset的mask
        pbb[:, 2] = Mask.shape[1] - pbb[:, 2]
        pbb[:, 3] = Mask.shape[2] - pbb[:, 3]
    pos = VoxelToWorldCoord(pbb[:, 1:], origin, spacing)  # 将输出转化为世界坐标
    print('voxel to world finished!')
    print('最终世界坐标的shape',pos.shape)

    rowlist = []
    for nk in range(pos.shape[0]):  # 每一个结节：文件名，z,y,x，是结节的概率(经过sigmoid处理)
        rowlist.append([bboxfname[:-8], pos[nk, 2], pos[nk, 1], pos[nk, 0], 1 / (1 + np.exp(-pbb[nk, 0]))])
    print(len(rowlist), len(rowlist[0]))
    return rowlist  # bboxfname[:-8], pos[:K, 2], pos[:K, 1], pos[:K, 0], 1/(1+np.exp(-pbb[:K,0]))

# 输入：标签文件，无关标签文件，用户id文件，以及最后的结果文件（上面生成的csv文件）
# 输出：评分（标量）
def getfrocvalue(results_filename):
    return noduleCADEvaluation(annotations_filename, annotations_excluded_filename, seriesuids_filename,
                               results_filename, './result')  # vis=False)


p = Pool(nprocess)  # Pool可以有效提升多进程的执行效率

def getcsv(detp, eps):  # 给定阈值和epoch
    """
    对输出的结果文件调用convertcsv函数处理
    每一个epoch生成一个csv文件，存放80多个测试病例的预测结节位置及概率
    """
    for ep in eps:  # 对每一个epoch
        #bboxpath = results_path + str(ep) + '/'  # 找到每个epoch的路径
        bboxpath = results_path + 'bbox/'
        print(results_path)
        print(bboxpath)
        for detpthresh in detp:
            print('ep', ep, 'detp', detpthresh)
            f = open(bboxpath + 'predanno' + str(detpthresh) + 'd3.csv', 'w')
            fwriter = csv.writer(f)
            fwriter.writerow(firstline)  # 写入的第一行为 用户id 结节坐标x,y,z 结节概率
            fnamelist = []
            for fname in os.listdir(bboxpath):  # bboxpath目录下的所有文件和文件夹
                if fname.endswith('_pbb.npy'):  # 找到以_pbb.npy结尾的文件(结节概率文件)，添加进文件列表
                    fnamelist.append(fname)
            print(11111,len(fnamelist))

            # 使用functools.partial可以创建一个新的函数，这个新函数可以固定住原函数的部分参数，从而在调用时更简单
            # map()将fnamelist的元素并行地送给convertcsv(_,bboxpath,detpthresh)处理
            predannolist = p.map(functools.partial(convertcsv, bboxpath=bboxpath, detp=detpthresh), fnamelist)
            print(333,len(predannolist), len(predannolist[0]))
            for predanno in predannolist:
                # print(predanno)
                for row in predanno:
                    # print(row)
                    fwriter.writerow(row)
            f.close()
            print('-------------------finished!-------------------')


# 每个epoch都会对应一个csv文件，要选取一个最好的结果，选取标准为froc值
def getfroc(detp, eps):  # 阈值和epoch
    maxfroc = 0
    maxep = 0
    for ep in eps:  # 对每个epoch分别处理
        #bboxpath = results_path + str(ep) + '/'
        bboxpath = results_path + 'bbox/'
        print('result path : ',results_path)
        print('bbox path : ',bboxpath)
        predannofnamalist = []
        for detpthresh in detp:  # 对于阈值列表中的每一个阈值
            #predannofnamalist.append(bboxpath + 'predanno' + str(detpthresh) + 'pbb.csv')
            predannofnamalist.append(bboxpath + 'predanno' + str(detpthresh) + 'd3.csv')

        #froclist = p.map(getfrocvalue, predannofnamalist)  # 得到当前epoch的所有froc值
        #froclist = [getfrocvalue(predanno) for predanno in predannofnamalist]
        froclist = []
        for predanno in predannofnamalist:
            _, sens, _, _, _, _, _ = getfrocvalue(predanno)
            print(max(sens))
            froclist.append(max(sens))
        print('result path : ',results_path)
        print('bbox path : ',bboxpath)
        print('predannofnamalist : ')
        print(predannofnamalist)
        print('froclist : ')
        print(froclist)

        if maxfroc < max(froclist):  # 如果记录的maxfroc值小于当前epoch的froclist的最大值
            maxep = ep  # 更新maxep
            maxfroc = max(froclist)  # 更新maxfroc
        print('froclist : ')
        print(froclist)
        print('detp0 : ' + str(detp[0]))
        print('detp1 : ' + str(detp[1]))
        for detpthresh in detp:  # TODO 没看懂这个循环在干嘛
            print(len(froclist), int((detpthresh-detp[0])/(detp[1]-detp[0])))
            frocarr[(ep - eps[0]) / (eps[1] - eps[0]), int((detpthresh - detp[0]) / (detp[1] - detp[0]))] = froclist[int((detpthresh - detp[0]) / (detp[1] - detp[0]))]
            print('ep', ep, 'detp', detpthresh, froclist[int((detpthresh - detp[0]) / (detp[1] - detp[0]))])
    print(maxfroc, maxep)

getcsv(detp,eps)

print('-------------------get csv finied !-------------------')

getfroc(detp, eps)
p.close()

print('-------------------get froc finied !-------------------')



fig = plt.imshow(frocarr.T)
plt.colorbar()
plt.xlabel('# Epochs')
plt.ylabel('# Detection Prob.')
xtick = detp  # [36, 37, 38, 39, 40]
plt.yticks(range(len(xtick)), xtick)
ytick = eps  # range(51, maxeps+1, 2)
plt.xticks(range(len(ytick)), ytick)
plt.title('Average FROC')

plt.savefig(results_path + 'frocavg.png')
np.save(results_path + 'frocavg.npy', frocarr)

frocarr = np.load(results_path + 'frocavg.npy', 'r')
froc, x, y = 0, 0, 0
for i in range(frocarr.shape[0]):
    for j in range(frocarr.shape[1]):
        if froc < frocarr[i, j]:
            froc, x, y = frocarr[i, j], i, j
print(fold, froc, x, y)
# print(maxfroc)