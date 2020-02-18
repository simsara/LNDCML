import os

import pandas as pd
import numpy as np

from utils import file
from utils.log import get_logger
from utils.tools import load_itk_image, world_to_voxel

log = get_logger(__name__)


# luna srs_id -> [pid, std_id]
def get_id_map():
    # read map file
    filename = os.path.join(file.get_cls_data_path(), 'LIDC-LUNA16-mapping.csv')
    sid_map = {}
    fid = open(filename, 'r')
    line = fid.readline()
    line = fid.readline()
    while line:
        pid_list = line.split(' ')
        pid = pid_list[0]
        std_id = pid_list[1]
        srs_id = pid_list[2]
        if srs_id not in sid_map:
            sid_map[srs_id] = [pid, std_id]
        else:
            assert sid_map[srs_id][0] == pid
            assert sid_map[srs_id][1] == std_id
        line = fid.readline()
    fid.close()
    return sid_map


# read luna16 annotation. map[srs_id] = [anno1, anno2]
def load_luna_dict():
    col_names = ['seriesuid', 'coordX', 'coordY', 'coordZ', 'diameter_mm']
    annotation = file.get_luna_csv_name('annotations.csv')
    luna_ant_frame = pd.read_csv(annotation, names=col_names)
    srslist = luna_ant_frame.seriesuid.tolist()[1:]
    cdxlist = luna_ant_frame.coordX.tolist()[1:]
    cdylist = luna_ant_frame.coordY.tolist()[1:]
    cdzlist = luna_ant_frame.coordZ.tolist()[1:]
    dimlist = luna_ant_frame.diameter_mm.tolist()[1:]
    luna_ant_dict = {}
    for idx in range(len(srslist)):
        vlu = [float(cdxlist[idx]), float(cdylist[idx]), float(cdzlist[idx]), float(dimlist[idx])]
        if srslist[idx] in luna_ant_dict:
            luna_ant_dict[srslist[idx]].append(vlu)
        else:
            luna_ant_dict[srslist[idx]] = [vlu]
    return luna_ant_dict


# srs_id -> [ant_list, voxel_list]
def generate_luna_ant_dict_lidc(luna_dict):
    data_dir = file.get_luna_data_path()
    luna_ant_dict_lidc = {}
    for fold in range(10):
        subset_dir = os.path.join(data_dir, 'subset%d' % fold)
        log.info("Reading %s" % subset_dir)

        file_map = {}  # mhd文件名 -> voxel坐标列表
        for fname in os.listdir(subset_dir):
            if fname.endswith('.mhd') and fname[:-4] in luna_dict:
                sliceim, origin, spacing, isflip = load_itk_image(os.path.join(subset_dir, fname))
                luna_ant_dict_lidc[fname[:-4]] = []
                voxcrdlist = []
                for luna_ant in luna_dict[fname[:-4]]:
                    voxcrd = world_to_voxel(luna_ant[:3][::-1], origin, spacing)
                    voxcrd[-1] = sliceim.shape[0] - voxcrd[0]
                    voxcrdlist.append(voxcrd)
                file_map[fname] = voxcrdlist

        for fname in os.listdir(subset_dir):
            if fname.endswith('.mhd') and fname[:-4] in luna_dict:
                luna_ant_dict_lidc[fname[:-4]] = []
                voxcrdlist = file_map[fname]
                for subidx, luna_ant in enumerate(luna_dict[fname[:-4]]):
                    luna_ant_dict_lidc[fname[:-4]].append([luna_ant, voxcrdlist[subidx]])
    to_save = os.path.join(file.get_cls_data_path(), 'luna_ant_dict_lidc.npy')
    np.save(to_save, luna_ant_dict_lidc)


# case -> [lidc_value_list]
def load_lidc_value_list_map():
    # read LIDC dataset
    lidc_csv_fname = os.path.join(file.get_cls_data_path(), 'list3.2.csv')
    col_names = ['case', 'scan', 'roi', 'volume', 'diam', 'x', 'y', 'slice', 'empty', 'nodIDs1', 'nodIDs2', 'nodIDs3',
                 'nodIDs4', 'nodIDs5', 'nodIDs6', 'nodIDs7']
    csv_data = pd.read_csv(lidc_csv_fname, names=col_names)
    case_l = csv_data.case.tolist()[1:]  # 0
    scan_l = csv_data.scan.tolist()[1:]  # 1
    roi_l = csv_data.roi.tolist()[1:]  # 2
    volume_l = csv_data.volume.tolist()[1:]  # 3
    diam_l = csv_data.diam.tolist()[1:]  # 4
    x_l = csv_data.x.tolist()[1:]  # 5
    y_l = csv_data.y.tolist()[1:]  # 6
    slice_l = csv_data.slice.tolist()[1:]  # 7

    nod1_l = csv_data.nodIDs1.tolist()[1:]  # 9
    nod2_l = csv_data.nodIDs2.tolist()[1:]  # 10
    nod3_l = csv_data.nodIDs3.tolist()[1:]  # 11
    nod4_l = csv_data.nodIDs4.tolist()[1:]  # 12
    nod5_l = csv_data.nodIDs5.tolist()[1:]  # 13
    nod6_l = csv_data.nodIDs6.tolist()[1:]  # 14
    nod7_l = csv_data.nodIDs7.tolist()[1:]  # 15
    nod_list = [nod1_l, nod2_l, nod3_l, nod4_l, nod5_l, nod6_l, nod7_l]

    ant_dict = {}
    for row in range(len(case_l)):
        value_list = [int(roi_l[row]), float(volume_l[row]), float(diam_l[row]), int(x_l[row]), int(y_l[row]), int(slice_l[row])]
        for col in range(7):
            val = nod_list[col][row]
            if val != '':
                try:
                    val = int(val)
                except ValueError:
                    pass
                value_list.append(val)
            key = '%s_%d' % (case_l[row], int(scan_l[row]))
            if key not in ant_dict:
                ant_dict[key] = [value_list]
            else:
                ant_dict[key].append(value_list)
    return ant_dict


def run():
    sid_map = get_id_map()
    luna_dict = load_luna_dict()
    generate_luna_ant_dict_lidc(luna_dict)
    lidc_value_list_map = load_lidc_value_list_map()
    


if __name__ == '__main__':
    load_lidc_value_list_map()