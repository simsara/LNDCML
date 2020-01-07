import math
import os

import matplotlib

from eval.csv_label import uid_label, x_label, z_label, y_label, diameter_mm_label, probability_label
from eval.NoduleFinding import NoduleFinding
from utils.log import get_logger
from utils.tools import *

matplotlib.use('agg')

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedFormatter
import sklearn.metrics as skl_metrics
import numpy as np

log = get_logger(__name__)

# plot settings
FROC_minX = 0.125  # Mininum value of x-axis of FROC curve
FROC_maxX = 8  # Maximum value of x-axis of FROC curve


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
    log.info(interp_sens.shape)
    for i in range(interp_sens.shape[1]):
        # get sorted vector
        vec = interp_sens[:, i]
        vec.sort()

        sens_mean[i] = np.average(vec)
        sens_lb[i] = vec[int(math.floor(Pz * len(vec)))]
        sens_up[i] = vec[int(math.floor((1.0 - Pz) * len(vec)))]

    return sens_mean, sens_lb, sens_up


def compute_froc_bootstrap(gt_list, prob_list, froc_uid_list,
                           uid_list, exclude_list,
                           numberOfBootstrapSamples=1000, confidence=0.95):
    set1 = np.concatenate(([gt_list], [prob_list], [exclude_list]), axis=0)

    fps_lists = []
    sens_lists = []
    thresholds_lists = []
    uid_list_np = np.array(uid_list)

    # Make a dict with all candidates of all scans
    uid_candidate_dict = {}
    for i in range(len(froc_uid_list)):
        uid = froc_uid_list[i]
        candidate = np.copy(set1[:, i:i + 1])

        if uid not in uid_candidate_dict:
            uid_candidate_dict[uid] = candidate
        else:
            uid_candidate_dict[uid] = np.concatenate((uid_candidate_dict[uid], candidate), axis=1)

    for i in range(numberOfBootstrapSamples):
        # print 'computing FROC: bootstrap %d/%d' % (i,numberOfBootstrapSamples)
        # Generate a bootstrapped set
        btpsamp = generateBootstrapSet(uid_candidate_dict, uid_list_np)
        fps, sens, thresholds = computeFROC(btpsamp[0, :], btpsamp[1, :], len(uid_list), btpsamp[2, :])

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
            # log.info(1111111111111111111, FROCGTList[i])
            # log.info(2222222222222222222, FROCProbList[i])
            FROCGTList_local.append(FROCGTList[i])
            FROCProbList_local.append(FROCProbList[i])

    numberOfDetectedLesions = sum(FROCGTList_local)  # 检测正确的结节数
    totalNumberOfLesions = sum(FROCGTList)  # 实际注释的结节数
    totalNumberOfCandidates = len(FROCProbList_local)  # 候选结节的数目

    # log.info(1111111111111111111,FROCGTList_local)
    # log.info(2222222222222222222,FROCProbList_local)

    fpr, tpr, thresholds = skl_metrics.roc_curve(FROCGTList_local, FROCProbList_local)

    if sum(FROCGTList) == len(FROCGTList):  # 不存在假阳率的时候（检测正确 == 候选）
        log.info("WARNING, this system has no false positives..")
        fps = np.zeros(len(fpr))
    else:  # false positive / scans
        fps = fpr * (totalNumberOfCandidates - numberOfDetectedLesions) / totalNumberOfImages

    sens = (tpr * numberOfDetectedLesions) / totalNumberOfLesions

    return fps, sens, thresholds


def get_candidate_dict(uid_list, results_filename, maxNumberOfCADMarks):
    log.info('Result filename : %s' % results_filename)
    results = read_csv(results_filename)  # 读取检测结果的csv文件，id,x,y,z,p

    all_candidates = {}
    uid_with_cadidate = 0
    for uid in uid_list:  # 对每个病例读取相应的候选结节
        # collect candidates from result file
        nodules = {}
        header = results[0]  # ['seriesuid', 'coordX', 'coordY', 'coordZ', 'probability']
        candidate_count = 0
        for result in results[1:]:
            nodule_uid = result[header.index(uid_label)]  # 获取检测到的结节对应的病人id seriesuid_label = 'seriesuid'
            if uid == nodule_uid:
                nodule = get_nodule(result, header)
                nodule.candidateID = candidate_count
                nodules[nodule.candidateID] = nodule
                candidate_count += 1

        if candidate_count > 0:  # 看有多少病例找到了候选结节
            uid_with_cadidate += 1

        if maxNumberOfCADMarks > 0:
            # number of CAD marks, only keep must suspicous marks
            if len(nodules.keys()) > maxNumberOfCADMarks:
                # make a list of all probabilities
                probs = []
                for uid, nodule in nodules.items():
                    probs.append(float(nodule.CADprobability))
                probs.sort(reverse=True)  # sort from large to small
                prob_threshold = probs[maxNumberOfCADMarks]
                nodules2 = {}  # 找出最多 maxNumberOfCADMarks 个概率最高的结节
                nodules2_count = 0
                for uid, nodule in nodules.items():
                    if nodules2_count >= maxNumberOfCADMarks:
                        break
                    if float(nodule.CADprobability) > prob_threshold:
                        nodules2[uid] = nodule
                        nodules2_count += 1

                nodules = nodules2

        all_candidates[uid] = nodules  # 将病例与对应候选结节存入字典
        log.info('[%s] 候选结节个数: %d ' % (uid, candidate_count))  # 只会对做test的子文件里的病例挑选出候选结节，其他子文件夹内的病例候选结节为0
    log.info('有这么多个病例找到了候选结节: %d ' % uid_with_cadidate)
    return all_candidates


def get_froc_list(uid_list, output_dir, CADSystemName, all_nodules,
                  numberOfBootstrapSamples, confidence, all_candidates):
    # open output files
    nod_no_cand_file = open(os.path.join(output_dir, "nodulesWithoutCandidate_%s.txt" % CADSystemName), 'w')

    analysis_file = open(os.path.join(output_dir, 'CADAnalysis.txt'), 'w')  # 写入CADAnalysis文件
    # 文件开始的说明
    analysis_file.write("\n")
    analysis_file.write((60 * "*") + "\n")
    analysis_file.write("CAD Analysis: %s\n" % CADSystemName)
    analysis_file.write((60 * "*") + "\n")
    analysis_file.write("\n")

    # --- iterate over all cases (seriesUIDs) and determine how
    # often a nodule annotation is not covered by a candidate

    # initialize some variables to be used in the loop
    cand_tp = 0
    cand_fp = 0
    cand_fn = 0
    cand_tn = 0

    total_cand = 0  # 总候选结节数，也就是检测结果文件中所有结节的数量
    total_nodule = 0  # 总标签结节数
    multi_cand = 0
    irr_cand = 0  # 无关的候选结节 直径小于3
    min_prob = -1000000000.0  # minimum value of a float

    froc_gt_list = []
    froc_prob_list = []
    froc_uid_list = []
    excludeList = []
    froc_to_nodule_map = []
    ignored_cad_marks_list = []

    # -- loop over the cases
    for uid in uid_list:  # 对于test的每一个病人
        # get the candidates for this case
        try:
            candidates = all_candidates[uid]  # 获取当前病人的候选结节
        except KeyError:
            candidates = {}
        # log.info('[done] get the candidates for this case!')

        total_cand += len(candidates.keys())  # 加进总候选结节中

        # make a copy in which items will be deleted
        candidates2 = candidates.copy()  # 对当前病人的候选结节复制一个副本

        # get the nodule annotations on this case
        try:
            uid_nodules = all_nodules[uid]  # 获取当前病人所有的注释结节（真结节+无关结节）
        except KeyError:
            uid_nodules = []
        # log.info('[done] get the nodule annotations on this case!')
        # log.info('noduleAnnots ： ' + str(len(noduleAnnots)))

        # - loop over the nodule annotations
        for nodule in uid_nodules:  # 对每一个注释的结节
            # increment the number of nodules
            if nodule.included():  # 加入用来评测的真结节
                total_nodule += 1

            x = float(nodule.coordX)  # 获取一个注释结节的坐标
            y = float(nodule.coordY)
            z = float(nodule.coordZ)

            # 2. 查看一个注释结节是否被一个候选结节cover
            # A nodule is marked as detected when the center of mass of the candidate is within a distance R of
            # the center of the nodule. In order to ensure that the CAD mark is displayed within the nodule on the
            # CT scan, we set R to be the radius of the nodule size.
            diameter = float(nodule.diameter_mm)  # 获取注释结节的直径
            if diameter < 0.0:
                diameter = 10.0
            radius_squared = pow((diameter / 2.0), 2.0)  # 半径的平方

            found = False
            match_candidates = []
            for key, candidate in candidates.items():  # 对于每一个候选结节，判断是否与真实结节相交
                x2 = float(candidate.coordX)
                y2 = float(candidate.coordY)
                z2 = float(candidate.coordZ)

                dist = math.pow(x - x2, 2.) + math.pow(y - y2, 2.) + math.pow(z - z2, 2.)  # 计算两个结节中心的距离的平方

                if dist < radius_squared:  # 判断是否在半径距离内
                    if nodule.included():  # 如果是用来检测的结节，匹配成功
                        found = True
                        match_candidates.append(candidate)

                        if key not in candidates2.keys():  # 把每个与注释结节相交的候选结节提取出来后，要删除副本中的id，以此检测是否有其他注释结节与该候选结节相交
                            log.info(
                                "This is strange: CAD mark %s detected two nodules! Check for overlapping nodule annotations, SeriesUID: %s, nodule Annot ID: %s" % (
                                    str(candidate.id), uid, str(nodule.id)))
                        else:
                            del candidates2[key]
                    elif nodule.excluded():  # 如果是无关结节
                        if key in candidates2.keys():
                            irr_cand += 1
                            ignored_cad_marks_list.append("%s,%s,%s,%s,%s,%s,%.9f" % (
                                uid, -1, candidate.coordX, candidate.coordY, candidate.coordZ,
                                str(candidate.id), float(candidate.CADprobability)))
                            del candidates2[key]

            if len(match_candidates) > 1:  # 如果一个标签结节对应多个候选结节，记下数目
                multi_cand += (len(match_candidates) - 1)

            if nodule.included():  # 若该注释结节是真结节
                # only include it for FROC analysis if it is included
                # otherwise, the candidate will not be counted as FP, but ignored in the
                # analysis since it has been deleted from the nodules2 vector of candidates
                if found == True:  # 找到了与之匹配的候选结节
                    # append the sample with the highest probability for the FROC analysis
                    max_prob = None
                    for candidate in match_candidates:  # 有多个与之匹配的候选结节
                        if (max_prob is None) or (float(candidate.CADprobability) > max_prob):
                            max_prob = float(candidate.CADprobability)  # 记录匹配的候选结节的概率，将最大概率存入maxProb

                    froc_gt_list.append(1.0)  # 添加1
                    froc_prob_list.append(float(max_prob))  # 添加最大概率
                    froc_uid_list.append(uid)  # 添加病例id
                    excludeList.append(False)
                    froc_to_nodule_map.append("%s,%s,%s,%s,%s,%.9f,%s,%.9f" % (
                        uid, nodule.id, nodule.coordX, nodule.coordY, nodule.coordZ,
                        float(nodule.diameter_mm), str(candidate.id), float(candidate.CADprobability)))
                    cand_tp += 1

                else:  # 没找到与之匹配的候选结节
                    cand_fn += 1
                    # append a positive sample with the lowest probability, such that this is added in the FROC analysis
                    froc_gt_list.append(1.0)
                    froc_prob_list.append(min_prob)
                    froc_uid_list.append(uid)
                    excludeList.append(True)
                    froc_to_nodule_map.append("%s,%s,%s,%s,%s,%.9f,%s,%s" % (
                        uid, nodule.id, nodule.coordX, nodule.coordY, nodule.coordZ,
                        float(nodule.diameter_mm), int(-1), "NA"))
                    nod_no_cand_file.write("%s,%s,%s,%s,%s,%.9f,%s\n" % (
                        uid, nodule.id, nodule.coordX, nodule.coordY, nodule.coordZ,
                        float(nodule.diameter_mm), str(-1)))

        # add all false positives to the vectors
        for key, fp_candidate in candidates2.items():  # 此时candidates2中是无人领取的候选结节，都是false positive
            cand_fp += 1
            froc_gt_list.append(0.0)
            froc_prob_list.append(float(fp_candidate.CADprobability))
            froc_uid_list.append(uid)
            excludeList.append(False)
            froc_to_nodule_map.append("%s,%s,%s,%s,%s,%s,%.9f" % (
                uid, -1, fp_candidate.coordX, fp_candidate.coordY, fp_candidate.coordZ, str(fp_candidate.id),
                float(fp_candidate.CADprobability)))
    log.info('Total number of candidates : %d' % total_cand)

    analysis_file.write("Candidate detection results:\n")
    analysis_file.write("    True positives: %d\n" % cand_tp)
    analysis_file.write("    False positives: %d\n" % cand_fp)
    analysis_file.write("    False negatives: %d\n" % cand_fn)
    analysis_file.write("    True negatives: %d\n" % cand_tn)  # 没有统计 不需要
    analysis_file.write("    Total number of candidates: %d\n" % total_cand)  # 总候选结节（对于目前test中的所有病人）
    analysis_file.write("    Total number of nodules: %d\n" % total_nodule)  # 总标签结节

    analysis_file.write("    Ignored candidates on excluded nodules: %d\n" % irr_cand)
    analysis_file.write(
        "    Ignored candidates which were multiple detections on a nodule: %d\n" % multi_cand)

    if total_nodule == 0:
        analysis_file.write("    Sensitivity: 0.0\n")
    else:
        analysis_file.write("    Sensitivity: %.9f\n" % (float(cand_tp) / float(total_nodule)))
    analysis_file.write(
        "    Average number of candidates per scan: %.9f\n" % (float(total_cand) / float(len(uid_list))))

    if not (len(froc_gt_list) == len(froc_prob_list) and len(froc_gt_list) == len(froc_uid_list) and
            len(froc_gt_list) == len(froc_to_nodule_map) and len(froc_gt_list) == len(excludeList)):
        analysis_file.write("Length of FROC vectors not the same, this should never happen! Aborting..\n")
        raise ValueError('Length of FROC vectors not the same')

    # 计算froc，返回的是召回率和假阳性率的列表
    log.info('gt: %d. froc: %d. uid: %d. exclude: %d' % (len(froc_gt_list), len(froc_prob_list), len(uid_list), len(excludeList)))
    fps, sens, thresholds = computeFROC(froc_gt_list, froc_prob_list, len(uid_list), excludeList)

    fps_bs_itp, sens_bs_mean, sens_bs_lb, sens_bs_up = \
        compute_froc_bootstrap(froc_gt_list, froc_prob_list, froc_uid_list,
                               uid_list, excludeList,
                               numberOfBootstrapSamples=numberOfBootstrapSamples,
                               confidence=confidence)

    return sens, fps, thresholds, froc_gt_list, froc_prob_list, \
           fps_bs_itp, sens_bs_mean, sens_bs_lb, sens_bs_up, total_nodule


def evaluate_cad(uid_list, results_filename, output_dir, all_nodules, CADSystemName, maxNumberOfCADMarks=-1,
                 numberOfBootstrapSamples=1000, confidence=0.95):
    '''
    function to evaluate a CAD algorithm
    @param uid_list: list of the seriesUIDs of the cases to be processed 病人id序列
    @param results_filename: file with results  检测结果
    @param output_dir: output directory  评估结果的输出目录
    @param all_nodules: dictionary with all nodule annotations of all cases, keys of the dictionary are the seriesuids
    @param CADSystemName: name of the CAD system, to be used in filenames and on FROC curve 检测系统的名称
    '''
    all_candidates = get_candidate_dict(uid_list, results_filename, maxNumberOfCADMarks)  # uid -> list[nodule]
    sens, fps, thresholds, FROCGTList, FROCProbList, \
    fps_bs_itp, sens_bs_mean, sens_bs_lb, sens_bs_up, totalNumberOfNodules = \
        get_froc_list(uid_list, output_dir, CADSystemName, all_nodules, numberOfBootstrapSamples, confidence,
                      all_candidates)

    draw_froc(output_dir, CADSystemName,
              sens, fps, thresholds, FROCGTList, FROCProbList,
              fps_bs_itp, sens_bs_mean, sens_bs_lb, sens_bs_up, totalNumberOfNodules)

    return fps, sens, thresholds, fps_bs_itp, sens_bs_mean, sens_bs_lb, sens_bs_up


def draw_froc(output_dir, CADSystemName,
              sens, fps, thresholds, FROCGTList, FROCProbList,
              fps_bs_itp, sens_bs_mean, sens_bs_lb, sens_bs_up, totalNumberOfNodules):
    # -----------------------------------------画图FROC--------------------------------------------
    with open(os.path.join(output_dir, "froc_%s.txt" % CADSystemName), 'w') as f:
        for candidate_count in range(len(sens)):
            f.write("%.9f,%.9f,%.9f\n" % (fps[candidate_count], sens[candidate_count], thresholds[candidate_count]))

    # Write FROC vectors to disk as well
    with open(os.path.join(output_dir, "froc_gt_prob_vectors_%s.csv" % CADSystemName), 'w') as f:
        for candidate_count in range(len(FROCGTList)):
            f.write("%d,%.9f\n" % (FROCGTList[candidate_count], FROCProbList[candidate_count]))

    fps_itp = np.linspace(FROC_minX, FROC_maxX, num=10001)

    sens_itp = np.interp(fps_itp, fps, sens)
    frvvlu = 0
    nxth = 0.125
    for fp, ss in zip(fps_itp, sens_itp):
        if abs(fp - nxth) < 3e-4:
            frvvlu += ss
            nxth *= 2
        if abs(nxth - 16) < 1e-5:
            break

    # Write mean, lower, and upper bound curves to disk
    with open(os.path.join(output_dir, "froc_%s_bootstrapping.csv" % CADSystemName), 'w') as f:
        f.write("FPrate,Sensivity[Mean],Sensivity[Lower bound],Sensivity[Upper bound]\n")
        for candidate_count in range(len(fps_bs_itp)):
            f.write("%.9f,%.9f,%.9f,%.9f\n" % (
                fps_bs_itp[candidate_count], sens_bs_mean[candidate_count], sens_bs_lb[candidate_count],
                sens_bs_up[candidate_count]))

    # create FROC graphs
    if int(totalNumberOfNodules) > 0:
        graphTitle = str("")
        fig1 = plt.figure()
        ax = plt.gca()
        clr = 'b'
        plt.plot(fps_itp, sens_itp, color=clr, label="%s" % CADSystemName, lw=2)

        # bootstrap part
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
        plt.title('FROC performance - %s' % CADSystemName)

        plt.xscale('log', basex=2)
        ax.xaxis.set_major_formatter(FixedFormatter([0.125, 0.25, 0.5, 1, 2, 4, 8]))

        # set your ticks manually
        ax.xaxis.set_ticks([0.125, 0.25, 0.5, 1, 2, 4, 8])
        ax.yaxis.set_ticks(np.arange(0.5, 1, 0.1))
        # ax.yaxis.set_ticks(np.arange(0, 1.1, 0.1))
        plt.grid(b=True, which='both')
        plt.tight_layout()

        plt.savefig(os.path.join(output_dir, "froc_%s.png" % CADSystemName), bbox_inches=0, dpi=300)


def get_nodule(annotation, header, state=""):
    nodule = NoduleFinding()  # 实例化结节

    # 依次将x,y,z添加到nodule对象的属性中去
    nodule.coordX = annotation[header.index(x_label)]
    nodule.coordY = annotation[header.index(y_label)]
    nodule.coordZ = annotation[header.index(z_label)]

    if diameter_mm_label in header:  # 检查有无直径标签，有则添加
        nodule.diameter_mm = annotation[header.index(diameter_mm_label)]

    if probability_label in header:  # 检查有无概率标签，有则添加
        nodule.CADprobability = annotation[header.index(probability_label)]

    if not state == "":
        nodule.state = state

    return nodule


def collect_nodule_annotations(annotations, annotations_excluded, uid_list):
    all_nodules = {}  # 将所有结节存储在字典中
    nodule_count = 0
    nodule_count_total = 0

    for uid in uid_list:  # 对于每一个病人
        nodules = []
        include_count = 0  # 对真正用来检测的结节计数

        # add included findings
        header = annotations[0]  # csv 第一行
        for annotation in annotations[1:]:  # ID，x,y,z，d
            nodule_uid = annotation[header.index(uid_label)]
            if uid == nodule_uid:  # 将结节所属的用户id与要建立字典索引的id比较，若相同，就获取它
                nodule = get_nodule(annotation, header, state="Included")
                nodules.append(nodule)
                include_count += 1

        # add excluded findings
        header = annotations_excluded[0]
        for annotation in annotations_excluded[1:]:
            nodule_uid = annotation[header.index(uid_label)]

            if uid == nodule_uid:
                nodule = get_nodule(annotation, header, state="Excluded")
                nodules.append(nodule)

        all_nodules[uid] = nodules  # 所有的结节
        nodule_count += include_count  # 所有应该包括进去的结节的个数
        nodule_count_total += len(nodules)  # 所有结节的个数

    log.info('Nodule annotations. Total: %d. Included: %d.' % (nodule_count_total, nodule_count))
    return all_nodules


def collect(annotations_filename, annotations_excluded_filename, uid_list):
    annotations = read_csv(annotations_filename)
    annotations_excluded = read_csv(annotations_excluded_filename)
    log.info('Uid list: %s' % str(uid_list))

    all_nodule = collect_nodule_annotations(annotations, annotations_excluded, uid_list)

    return all_nodule


def nodule_cad_evaluation(annotations_filename, annotations_excluded_filename, uid_list,
                          results_filename, output_dir):
    '''
    function to load annotations and evaluate a CAD algorithm
    @param annotations_filename: list of annotations
    @param annotations_excluded_filename: list of annotations that are excluded from analysis
    @param seriesuids_filename: list of CT images in seriesuids
    @param results_filename: list of CAD marks with probabilities
    @param output_dir: output directory
    '''

    # 根据标签和用户id，求出所有结节，所有用户
    nodules = collect(annotations_filename, annotations_excluded_filename, uid_list)

    # 根据结节，用户，结果文件，输出froc值
    fps, sens, thresholds, fps_bs_itp, sens_bs_mean, sens_bs_lb, sens_bs_up = \
        evaluate_cad(uid_list, results_filename, output_dir, nodules,
                     os.path.splitext(os.path.basename(results_filename))[0],
                     maxNumberOfCADMarks=100)
    return fps, sens, thresholds, fps_bs_itp, sens_bs_mean, sens_bs_lb, sens_bs_up
