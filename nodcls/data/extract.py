import os

import pandas as pd
import numpy as np
import xlrd

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
                for lunaant in luna_dict[fname[:-4]]:
                    voxcrd = world_to_voxel(lunaant[:3][::-1], origin, spacing)
                    voxcrd[-1] = sliceim.shape[0] - voxcrd[0]
                    voxcrdlist.append(voxcrd)
                file_map[fname] = voxcrdlist

        for fname in os.listdir(subset_dir):
            if fname.endswith('.mhd') and fname[:-4] in luna_dict:
                luna_ant_dict_lidc[fname[:-4]] = []
                voxcrdlist = file_map[fname]
                for subidx, lunaant in enumerate(luna_dict[fname[:-4]]):
                    luna_ant_dict_lidc[fname[:-4]].append([lunaant, voxcrdlist[subidx]])
    to_save = os.path.join(file.get_cls_data_path(), 'luna_ant_dict_lidc.npy')
    np.save(to_save, luna_ant_dict_lidc)


def load_lidc():
    to_save = os.path.join(file.get_cls_data_path(), 'luna_ant_dict_lidc.npy')
    # read LIDC dataset
    lidc_csv_fname = os.path.join(file.get_cls_data_path(), 'list3.2.csv')
    antdict = {}
    wb = xlrd.open_workbook(os.path.join(lidccsvfname))
    for s in wb.sheets():
        if s.name == 'list3.2':
            for row in range(1, s.nrows):
                valuelist = [int(s.cell(row, 2).value), s.cell(row, 3).value, s.cell(row, 4).value, \
                             int(s.cell(row, 5).value), int(s.cell(row, 6).value), int(s.cell(row, 7).value)]
                assert abs(s.cell(row, 1).value - int(s.cell(row, 1).value)) < 1e-8
                assert abs(s.cell(row, 2).value - int(s.cell(row, 2).value)) < 1e-8
                assert abs(s.cell(row, 5).value - int(s.cell(row, 5).value)) < 1e-8
                assert abs(s.cell(row, 6).value - int(s.cell(row, 6).value)) < 1e-8
                assert abs(s.cell(row, 7).value - int(s.cell(row, 7).value)) < 1e-8
                for col in range(9, 16):
                    if s.cell(row, col).value != '':
                        if isinstance(s.cell(row, col).value, float):
                            valuelist.append(int(s.cell(row, col).value))
                            assert abs(s.cell(row, col).value - int(s.cell(row, col).value)) < 1e-8
                        else:
                            valuelist.append(s.cell(row, col).value)
                if s.cell(row, 0).value+'_'+str(int(s.cell(row, 1).value)) not in antdict:
                    antdict[s.cell(row, 0).value+'_'+str(int(s.cell(row, 1).value))] = [valuelist]
                else:
                    antdict[s.cell(row, 0).value+'_'+str(int(s.cell(row, 1).value))].append(valuelist)



def run():
    id_map = get_id_map()
    luna_dict = load_luna_dict()
    generate_luna_ant_dict_lidc(luna_dict)
