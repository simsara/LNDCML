import os
import math
import sys
import matplotlib

matplotlib.use('agg')

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, LogFormatter, StrMethodFormatter, FixedFormatter
import sklearn.metrics as skl_metrics
import numpy as np

from evaluation.NoduleFinding import NoduleFinding

from tools import *

# matplotlib.rc('xtick', labelsize=18)
# matplotlib.rc('ytick', labelsize=18)
font = {'family': 'normal','size': 17}

matplotlib.rc('font', **font)
# Evaluation settings
bPerformBootstrapping = True
bNumberOfBootstrapSamples = 1000
bOtherNodulesAsIrrelevant = True
bConfidence = 0.95

seriesuid_label = 'seriesuid'
coordX_label = 'coordX'
coordY_label = 'coordY'
coordZ_label = 'coordZ'
diameter_mm_label = 'diameter_mm'
CADProbability_label = 'probability'

# plot settings
FROC_minX = 0.125  # Mininum value of x-axis of FROC curve
FROC_maxX = 8  # Maximum value of x-axis of FROC curve
bLogPlot = True


def generateBootstrapSet(scanToCandidatesDict, FROCImList):
    '''
    Generates bootstrapped version of set
    '''
    imageLen = FROCImList.shape[0]

    # get a random list of images using sampling with replacement
    rand_index_im = np.random.randint(imageLen, size=imageLen)
    FROCImList_rand = FROCImList[rand_index_im]

    # get a new list of candidates
    candidatesExists = False
    for im in FROCImList_rand:
        if im not in scanToCandidatesDict:
            continue

        if not candidatesExists:
            candidates = np.copy(scanToCandidatesDict[im])
            candidatesExists = True
        else:
            candidates = np.concatenate((candidates, scanToCandidatesDict[im]), axis=1)

    return candidates


def compute_mean_ci(interp_sens, confidence=0.95):
    sens_mean = np.zeros((interp_sens.shape[1]), dtype='float32')
    sens_lb = np.zeros((interp_sens.shape[1]), dtype='float32')
    sens_up = np.zeros((interp_sens.shape[1]), dtype='float32')

    Pz = (1.0 - confidence) / 2.0
    print(interp_sens.shape)
    for i in range(interp_sens.shape[1]):
        # get sorted vector
        vec = interp_sens[:, i]
        vec.sort()

        sens_mean[i] = np.average(vec)
        sens_lb[i] = vec[int(math.floor(Pz * len(vec)))]
        sens_up[i] = vec[int(math.floor((1.0 - Pz) * len(vec)))]

    return sens_mean, sens_lb, sens_up


def computeFROC_bootstrap(FROCGTList, FROCProbList, FPDivisorList, FROCImList, excludeList,
                          numberOfBootstrapSamples=1000, confidence=0.95):
    set1 = np.concatenate(([FROCGTList], [FROCProbList], [excludeList]), axis=0)

    fps_lists = []
    sens_lists = []
    thresholds_lists = []

    FPDivisorList_np = np.asarray(FPDivisorList)
    FROCImList_np = np.asarray(FROCImList)

    # Make a dict with all candidates of all scans
    scanToCandidatesDict = {}
    for i in range(len(FPDivisorList_np)):
        seriesuid = FPDivisorList_np[i]
        candidate = set1[:, i:i + 1]

        if seriesuid not in scanToCandidatesDict:
            scanToCandidatesDict[seriesuid] = np.copy(candidate)
        else:
            scanToCandidatesDict[seriesuid] = np.concatenate((scanToCandidatesDict[seriesuid], candidate), axis=1)

    for i in range(numberOfBootstrapSamples):
        # print 'computing FROC: bootstrap %d/%d' % (i,numberOfBootstrapSamples)
        # Generate a bootstrapped set
        btpsamp = generateBootstrapSet(scanToCandidatesDict, FROCImList_np)
        fps, sens, thresholds = computeFROC(btpsamp[0, :], btpsamp[1, :], len(FROCImList_np), btpsamp[2, :])

        fps_lists.append(fps)
        sens_lists.append(sens)
        thresholds_lists.append(thresholds)

    # compute statistic
    all_fps = np.linspace(FROC_minX, FROC_maxX, num=10000)

    # Then interpolate all FROC curves at this points
    interp_sens = np.zeros((numberOfBootstrapSamples, len(all_fps)), dtype='float32')
    for i in range(numberOfBootstrapSamples):
        interp_sens[i, :] = np.interp(all_fps, fps_lists[i], sens_lists[i])

    # compute mean and CI
    sens_mean, sens_lb, sens_up = compute_mean_ci(interp_sens, confidence=confidence)

    return all_fps, sens_mean, sens_lb, sens_up



def computeFROC(FROCGTList, FROCProbList, totalNumberOfImages, excludeList):

    FROCGTList_local = []
    FROCProbList_local = []

    for i in range(len(excludeList)):  # 无关结节不纳入计算
        if excludeList[i] == False:
            FROCGTList_local.append(FROCGTList[i])
            FROCProbList_local.append(FROCProbList[i])

    numberOfDetectedLesions = sum(FROCGTList_local)  # 检测正确的结节数
    totalNumberOfLesions = sum(FROCGTList)  # 实际注释的结节数
    totalNumberOfCandidates = len(FROCProbList_local)  # 候选结节的数目

    fpr, tpr, thresholds = skl_metrics.roc_curve(FROCGTList_local, FROCProbList_local)

    if sum(FROCGTList) == len(FROCGTList):  # 不存在假阳率的时候（检测正确 == 候选）
        print("WARNING, this system has no false positives..")
        fps = np.zeros(len(fpr))
    else:  # false positive / scans
        fps = fpr * (totalNumberOfCandidates - numberOfDetectedLesions) / totalNumberOfImages

    sens = (tpr * numberOfDetectedLesions) / totalNumberOfLesions

    return fps, sens, thresholds

def evaluateCAD(seriesUIDs, results_filename, outputDir, allNodules, CADSystemName, maxNumberOfCADMarks=-1,
                performBootstrapping=False, numberOfBootstrapSamples=1000, confidence=0.95):
    '''
    function to evaluate a CAD algorithm
    @param seriesUIDs: list of the seriesUIDs of the cases to be processed 病人id序列
    @param results_filename: file with results  检测结果
    @param outputDir: output directory  评估结果的输出目录
    @param allNodules: dictionary with all nodule annotations of all cases, keys of the dictionary are the seriesuids
    @param CADSystemName: name of the CAD system, to be used in filenames and on FROC curve 检测系统的名称
    '''

    nodOutputfile = open(os.path.join(outputDir, 'CADAnalysis.txt'), 'w')  # 写入CADAnalysis文件

    # 文件开始的说明
    nodOutputfile.write("\n")
    nodOutputfile.write((60 * "*") + "\n")
    nodOutputfile.write("CAD Analysis: %s\n" % CADSystemName)
    nodOutputfile.write((60 * "*") + "\n")
    nodOutputfile.write("\n")

    results = readCSV(results_filename)  # 读取检测结果的csv文件，id,x,y,z,p

    allCandsCAD = {}

    for seriesuid in seriesUIDs:  # 对每个病例读取相应的候选结节

        # collect candidates from result file
        nodules = {}
        header = results[0]

        i = 0
        for result in results[1:]:
            nodule_seriesuid = result[header.index(seriesuid_label)]  # 获取检测到的结节对应的病人id

            if seriesuid == nodule_seriesuid:
                nodule = getNodule(result, header)
                nodule.candidateID = i
                nodules[nodule.candidateID] = nodule
                i += 1

        if (maxNumberOfCADMarks > 0):
            # number of CAD marks, only keep must suspicous marks

            if len(nodules.keys()) > maxNumberOfCADMarks:
                # make a list of all probabilities
                probs = []
                for keytemp, noduletemp in nodules.iteritems():
                    probs.append(float(noduletemp.CADprobability))
                probs.sort(reverse=True)  # sort from large to small
                probThreshold = probs[maxNumberOfCADMarks]
                nodules2 = {}
                nrNodules2 = 0
                for keytemp, noduletemp in nodules.iteritems():
                    if nrNodules2 >= maxNumberOfCADMarks:
                        break
                    if float(noduletemp.CADprobability) > probThreshold:
                        nodules2[keytemp] = noduletemp
                        nrNodules2 += 1

                nodules = nodules2

        # print('adding candidates: ', seriesuid)
        allCandsCAD[seriesuid] = nodules  # 将病例与对应候选结节存入字典

    # open output files
    nodNoCandFile = open(os.path.join(outputDir, "nodulesWithoutCandidate_%s.txt" % CADSystemName), 'w')

    # --- iterate over all cases (seriesUIDs) and determine how
    # often a nodule annotation is not covered by a candidate

    # initialize some variables to be used in the loop
    candTPs = 0
    candFPs = 0
    candFNs = 0
    candTNs = 0

    totalNumberOfCands = 0  # 总候选结节数，也就是检测结果文件中所有结节的数量
    totalNumberOfNodules = 0  # 总标签结节数
    doubleCandidatesIgnored = 0
    irrelevantCandidates = 0  # 直径小于3的结节？
    minProbValue = -1000000000.0  # minimum value of a float

    FROCGTList = []
    FROCProbList = []
    FPDivisorList = []
    excludeList = []
    FROCtoNoduleMap = []
    ignoredCADMarksList = []

    # -- loop over the cases
    for seriesuid in seriesUIDs:  # 对与test的每一个病人
        # get the candidates for this case
        try:
            candidates = allCandsCAD[seriesuid]  # 获取当前病人的候选结节
        except KeyError:
            candidates = {}

        totalNumberOfCands += len(candidates.keys())  # 加进总候选结节中

        # make a copy in which items will be deleted
        candidates2 = candidates.copy()  # 对当前病人的候选结节复制一个副本

        # get the nodule annotations on this case
        try:
            noduleAnnots = allNodules[seriesuid]  # 获取当前病人所有的注释结节（真结节+无关结节）
        except KeyError:
            noduleAnnots = []

        # - loop over the nodule annotations
        for noduleAnnot in noduleAnnots:  # 对每一个注释的结节
            # increment the number of nodules
            if noduleAnnot.state == "Included":  # 加入用来评测的真结节
                totalNumberOfNodules += 1

            x = float(noduleAnnot.coordX)  # 获取一个注释结节的坐标
            y = float(noduleAnnot.coordY)
            z = float(noduleAnnot.coordZ)

            # 2. 查看一个注释结节是否被一个候选结节cover
            # A nodule is marked as detected when the center of mass of the candidate is within a distance R of
            # the center of the nodule. In order to ensure that the CAD mark is displayed within the nodule on the
            # CT scan, we set R to be the radius of the nodule size.
            diameter = float(noduleAnnot.diameter_mm)  # 获取注释结节的直径
            if diameter < 0.0:
                diameter = 10.0
            radiusSquared = pow((diameter / 2.0), 2.0)  # 半径的平方

            found = False
            noduleMatches = []
            for key, candidate in candidates.iteritems():  # 对于每一个候选结节，判断是否与真实结节相交
                x2 = float(candidate.coordX)
                y2 = float(candidate.coordY)
                z2 = float(candidate.coordZ)

                dist = math.pow(x - x2, 2.) + math.pow(y - y2, 2.) + math.pow(z - z2, 2.)  # 计算两个结节中心的距离的平方

                if dist < radiusSquared:  # 判断是否在半径距离内
                    if (noduleAnnot.state == "Included"):  # 如果是用来检测的结节，匹配成功
                        found = True
                        noduleMatches.append(candidate)

                        if key not in candidates2.keys():  # 把每个与注释结节相交的候选结节提取出来后，要删除副本中的id，以此检测是否有其他注释结节与该候选结节相交
                            print("This is strange: CAD mark %s detected two nodules! Check for overlapping nodule annotations, SeriesUID: %s, nodule Annot ID: %s" % (
                            str(candidate.id), seriesuid, str(noduleAnnot.id)))
                        else:
                            del candidates2[key]

                    elif (noduleAnnot.state == "Excluded"):  # 如果是无关结节

                        if bOtherNodulesAsIrrelevant:  # delete marks on excluded nodules so they don't count as false positives
                            if key in candidates2.keys():
                                irrelevantCandidates += 1
                                ignoredCADMarksList.append("%s,%s,%s,%s,%s,%s,%.9f" % (
                                seriesuid, -1, candidate.coordX, candidate.coordY, candidate.coordZ, str(candidate.id),
                                float(candidate.CADprobability)))
                                del candidates2[key]

            if len(noduleMatches) > 1:  # 如果一个标签结节对应多个候选结节，记下数目
                doubleCandidatesIgnored += (len(noduleMatches) - 1)

            if noduleAnnot.state == "Included":  # 若该注释结节是真结节
                # only include it for FROC analysis if it is included
                # otherwise, the candidate will not be counted as FP, but ignored in the
                # analysis since it has been deleted from the nodules2 vector of candidates
                if found == True:  # 找到了与之匹配的候选结节
                    # append the sample with the highest probability for the FROC analysis
                    maxProb = None
                    for idx in range(len(noduleMatches)):  # 有多个与之匹配的候选结节
                        candidate = noduleMatches[idx]
                        if (maxProb is None) or (float(candidate.CADprobability) > maxProb):
                            maxProb = float(candidate.CADprobability)  # 记录匹配的候选结节的概率，将最大概率存入maxProb

                    FROCGTList.append(1.0)  # 添加1
                    FROCProbList.append(float(maxProb))  # 添加最大概率
                    FPDivisorList.append(seriesuid)  # 添加病例id
                    excludeList.append(False)
                    FROCtoNoduleMap.append("%s,%s,%s,%s,%s,%.9f,%s,%.9f" % (
                    seriesuid, noduleAnnot.id, noduleAnnot.coordX, noduleAnnot.coordY, noduleAnnot.coordZ,
                    float(noduleAnnot.diameter_mm), str(candidate.id), float(candidate.CADprobability)))
                    candTPs += 1

                else:  # 没找到与之匹配的候选结节
                    candFNs += 1
                    # append a positive sample with the lowest probability, such that this is added in the FROC analysis

                    FROCGTList.append(1.0)
                    FROCProbList.append(minProbValue)
                    FPDivisorList.append(seriesuid)
                    excludeList.append(True)
                    FROCtoNoduleMap.append("%s,%s,%s,%s,%s,%.9f,%s,%s" % (
                    seriesuid, noduleAnnot.id, noduleAnnot.coordX, noduleAnnot.coordY, noduleAnnot.coordZ,
                    float(noduleAnnot.diameter_mm), int(-1), "NA"))
                    nodNoCandFile.write("%s,%s,%s,%s,%s,%.9f,%s\n" % (
                    seriesuid, noduleAnnot.id, noduleAnnot.coordX, noduleAnnot.coordY, noduleAnnot.coordZ,
                    float(noduleAnnot.diameter_mm), str(-1)))

        # add all false positives to the vectors
        for key, candidate3 in candidates2.iteritems():  # 此时candidates2中是无人领取的候选结节，都是false positive
            candFPs += 1
            FROCGTList.append(0.0)
            FROCProbList.append(float(candidate3.CADprobability))
            FPDivisorList.append(seriesuid)
            excludeList.append(False)
            FROCtoNoduleMap.append("%s,%s,%s,%s,%s,%s,%.9f" % (
            seriesuid, -1, candidate3.coordX, candidate3.coordY, candidate3.coordZ, str(candidate3.id),
            float(candidate3.CADprobability)))

    if not (len(FROCGTList) == len(FROCProbList) and len(FROCGTList) == len(FPDivisorList) and len(FROCGTList) == len(
            FROCtoNoduleMap) and len(FROCGTList) == len(excludeList)):
        nodOutputfile.write("Length of FROC vectors not the same, this should never happen! Aborting..\n")

    nodOutputfile.write("Candidate detection results:\n")
    nodOutputfile.write("    True positives: %d\n" % candTPs)
    nodOutputfile.write("    False positives: %d\n" % candFPs)
    nodOutputfile.write("    False negatives: %d\n" % candFNs)
    nodOutputfile.write("    True negatives: %d\n" % candTNs)
    nodOutputfile.write("    Total number of candidates: %d\n" % totalNumberOfCands)  # 总候选结节（对于目前test中的所有病人）
    nodOutputfile.write("    Total number of nodules: %d\n" % totalNumberOfNodules)  # 总标签结节

    nodOutputfile.write("    Ignored candidates on excluded nodules: %d\n" % irrelevantCandidates)
    nodOutputfile.write(
        "    Ignored candidates which were double detections on a nodule: %d\n" % doubleCandidatesIgnored)

    if int(totalNumberOfNodules) == 0:
        nodOutputfile.write("    Sensitivity: 0.0\n")
    else:
        nodOutputfile.write("    Sensitivity: %.9f\n" % (float(candTPs) / float(totalNumberOfNodules)))
    nodOutputfile.write(
        "    Average number of candidates per scan: %.9f\n" % (float(totalNumberOfCands) / float(len(seriesUIDs))))

    # 计算froc，返回的是召回率和假阳性率的列表
    fps, sens, thresholds = computeFROC(FROCGTList, FROCProbList, len(seriesUIDs), excludeList)

    if performBootstrapping:  # TODO ？
        fps_bs_itp, sens_bs_mean, sens_bs_lb, sens_bs_up = computeFROC_bootstrap(FROCGTList, FROCProbList,
                                                                                 FPDivisorList, seriesUIDs, excludeList,
                                                                                 numberOfBootstrapSamples=numberOfBootstrapSamples,
                                                                                 confidence=confidence)

#-----------------------------------------画图FROC--------------------------------------------
    with open(os.path.join(outputDir, "froc_%s.txt" % CADSystemName), 'w') as f:
        for i in range(len(sens)):
            f.write("%.9f,%.9f,%.9f\n" % (fps[i], sens[i], thresholds[i]))

    # Write FROC vectors to disk as well
    with open(os.path.join(outputDir, "froc_gt_prob_vectors_%s.csv" % CADSystemName), 'w') as f:
        for i in range(len(FROCGTList)):
            f.write("%d,%.9f\n" % (FROCGTList[i], FROCProbList[i]))

    fps_itp = np.linspace(FROC_minX, FROC_maxX, num=10001)

    sens_itp = np.interp(fps_itp, fps, sens)
    frvvlu = 0
    nxth = 0.125
    for fp, ss in zip(fps_itp, sens_itp):
        if abs(fp - nxth) < 3e-4:
            frvvlu += ss
            nxth *= 2
        if abs(nxth - 16) < 1e-5: break
    print(frvvlu / 7, nxth)
    print(sens_itp[fps_itp == 0.125] + sens_itp[fps_itp == 0.25] + sens_itp[fps_itp == 0.5] + sens_itp[fps_itp == 1] +
          sens_itp[fps_itp == 2] \
          + sens_itp[fps_itp == 4] + sens_itp[fps_itp == 8])
    if performBootstrapping:
        # Write mean, lower, and upper bound curves to disk
        with open(os.path.join(outputDir, "froc_%s_bootstrapping.csv" % CADSystemName), 'w') as f:
            f.write("FPrate,Sensivity[Mean],Sensivity[Lower bound],Sensivity[Upper bound]\n")
            for i in range(len(fps_bs_itp)):
                f.write("%.9f,%.9f,%.9f,%.9f\n" % (fps_bs_itp[i], sens_bs_mean[i], sens_bs_lb[i], sens_bs_up[i]))
    else:
        fps_bs_itp = None
        sens_bs_mean = None
        sens_bs_lb = None
        sens_bs_up = None

    # create FROC graphs
    if int(totalNumberOfNodules) > 0:
        graphTitle = str("")
        fig1 = plt.figure()
        ax = plt.gca()
        clr = 'b'
        plt.plot(fps_itp, sens_itp, color=clr, label="%s" % CADSystemName, lw=2)
        if performBootstrapping:
            plt.plot(fps_bs_itp, sens_bs_mean, color=clr, ls='--')
            plt.plot(fps_bs_itp, sens_bs_lb, color=clr, ls=':')  # , label = "lb")
            plt.plot(fps_bs_itp, sens_bs_up, color=clr, ls=':')  # , label = "ub")
            ax.fill_between(fps_bs_itp, sens_bs_lb, sens_bs_up, facecolor=clr, alpha=0.05)
        xmin = FROC_minX
        xmax = FROC_maxX
        plt.xlim(xmin, xmax)
        plt.ylim(0.5, 1)
        plt.xlabel('Average number of false positives per scan')
        plt.ylabel('Sensitivity')
        plt.legend(loc='lower right')
        plt.title('FROC performance - %s' % (CADSystemName))

        if bLogPlot:
            plt.xscale('log', basex=2)
            ax.xaxis.set_major_formatter(FixedFormatter([0.125, 0.25, 0.5, 1, 2, 4, 8]))

        # set your ticks manually
        ax.xaxis.set_ticks([0.125, 0.25, 0.5, 1, 2, 4, 8])
        ax.yaxis.set_ticks(np.arange(0.5, 1, 0.1))
        # ax.yaxis.set_ticks(np.arange(0, 1.1, 0.1))
        plt.grid(b=True, which='both')
        plt.tight_layout()

        plt.savefig(os.path.join(outputDir, "froc_%s.png" % CADSystemName), bbox_inches=0, dpi=300)

    return (fps, sens, thresholds, fps_bs_itp, sens_bs_mean, sens_bs_lb, sens_bs_up)


def getNodule(annotation, header, state=""):
    nodule = NoduleFinding()  # 实例化结节

    # 依次将x,y,z添加到nodule对象的属性中去
    nodule.coordX = annotation[header.index(coordX_label)]
    nodule.coordY = annotation[header.index(coordY_label)]
    nodule.coordZ = annotation[header.index(coordZ_label)]

    if diameter_mm_label in header:  # 检查有无直径标签，有则添加
        nodule.diameter_mm = annotation[header.index(diameter_mm_label)]

    if CADProbability_label in header:  # 检查有无概率标签，有则添加
        nodule.CADprobability = annotation[header.index(CADProbability_label)]

    if not state == "":
        nodule.state = state

    return nodule


def collectNoduleAnnotations(annotations, annotations_excluded, seriesUIDs):
    allNodules = {}  # 将所有结节存储在字典中
    noduleCount = 0
    noduleCountTotal = 0

    for seriesuid in seriesUIDs:  # 对于每一个病人
        # print('adding nodule annotations: ', seriesuid)

        nodules = []
        numberOfIncludedNodules = 0  # 对真正用来检测的结节计数

        # add included findings
        header = annotations[0]  # ID，x,y,z，d
        for annotation in annotations[1:]:
            nodule_seriesuid = annotation[header.index(seriesuid_label)]

            if seriesuid == nodule_seriesuid:  # 将结节所属的用户id与要建立字典索引的id比较，若相同，就获取它
                nodule = getNodule(annotation, header, state="Included")
                nodules.append(nodule)
                numberOfIncludedNodules += 1

        # add excluded findings
        header = annotations_excluded[0]
        for annotation in annotations_excluded[1:]:
            nodule_seriesuid = annotation[header.index(seriesuid_label)]

            if seriesuid == nodule_seriesuid:
                nodule = getNodule(annotation, header, state="Excluded")
                nodules.append(nodule)

        allNodules[seriesuid] = nodules  # 所有的结节
        noduleCount += numberOfIncludedNodules  # 所有应该包括进去的结节的个数
        noduleCountTotal += len(nodules)  # 所有结节的个数

    print('Total number of included nodule annotations: ' + str(noduleCount))
    print('Total number of nodule annotations: ' + str(noduleCountTotal))
    return allNodules


def collect(annotations_filename, annotations_excluded_filename, seriesuids_filename):
    annotations = readCSV(annotations_filename)
    annotations_excluded = readCSV(annotations_excluded_filename)
    seriesUIDs_csv = readCSV(seriesuids_filename)

    seriesUIDs = []  # 建立一个用户列表，将用户id添加进去
    for seriesUID in seriesUIDs_csv:
        seriesUIDs.append(seriesUID[0])  # seriesUID也是一个列表，只不过只有一个元素

    allNodules = collectNoduleAnnotations(annotations, annotations_excluded, seriesUIDs)

    return (allNodules, seriesUIDs)


def noduleCADEvaluation(annotations_filename, annotations_excluded_filename, seriesuids_filename, results_filename,
                        outputDir):
    '''
    function to load annotations and evaluate a CAD algorithm
    @param annotations_filename: list of annotations
    @param annotations_excluded_filename: list of annotations that are excluded from analysis
    @param seriesuids_filename: list of CT images in seriesuids
    @param results_filename: list of CAD marks with probabilities
    @param outputDir: output directory
    '''

    print(annotations_filename)

    # 根据标签和用户id，求出所有结节，所有用户
    (allNodules, seriesUIDs) = collect(annotations_filename, annotations_excluded_filename, seriesuids_filename)

    # 根据结节，用户，结果文件，输出froc值
    evaluateCAD(seriesUIDs, results_filename, outputDir, allNodules,
                os.path.splitext(os.path.basename(results_filename))[0],
                maxNumberOfCADMarks=100, performBootstrapping=bPerformBootstrapping,
                numberOfBootstrapSamples=bNumberOfBootstrapSamples, confidence=bConfidence)


if __name__ == '__main__':
    annotations_filename = './annotations/annotations.csv'
    annotations_excluded_filename = './annotations/annotations_excluded.csv'
    seriesuids_filename = './annotations/seriesuids.csv'
    results_filename = './annotations/3DRes18FasterR-CNN.csv'  # 3D Faster R-CNN - Res18.csv' #top5.csv'#
    noduleCADEvaluation(annotations_filename, annotations_excluded_filename, seriesuids_filename, results_filename,
                        './')
    print("Finished!")