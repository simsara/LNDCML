import csv
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt

def writeCSV(filename, lines):
    with open(filename, "wb") as f:
        csvwriter = csv.writer(f)
        csvwriter.writerows(lines)

def readCSV(filename):
    lines = []
    with open(filename, "rb") as f:
        csvreader = csv.reader(f)
        for line in csvreader:
            lines.append(line)
    return lines

# 像素坐标转世界坐标
# voxel = (world-origin)/spacing -> world = voxel*spacing + origin
def VoxelToWorldCoord(voxelCoord, origin, spacing):
    strechedVocelCoord = voxelCoord * spacing
    worldCoord = strechedVocelCoord + origin
    return worldCoord

# 世界坐标转像素坐标
def world_to_voxel(world_coord, origin, spacing):
    stretched_voxel_coord = np.absolute(world_coord - origin)
    voxel_coord = stretched_voxel_coord / spacing
    return voxel_coord

def load_itk_image(filename):  # 与prepare/__init__.py的load_itk_img是一模一样的
    """
    加载原始数据
    :param filename: 以mhd结尾的文件路径字符串
    """
    with open(filename) as f:  # 打开一个mhd文件
        contents = f.readlines()
        # 找到图像矩阵是否翻转的标志，在现实中扫描时有人正卧有人仰卧，所以会出现图像翻转的现象
        line = [k for k in contents if k.startswith('TransformMatrix')][0]
        transformM = np.array(line.split(' = ')[1].split(' ')).astype('float')
        transformM = np.round(transformM)
        if np.any(transformM != np.array([1, 0, 0, 0, 1, 0, 0, 0, 1])):
            isflip = True
        else:
            isflip = False

    itkimage = sitk.ReadImage(filename)
    # 将其转换为ndarry形式，x轴与y轴发生了对调，(x,y,z)->(z,y,x)
    numpyImage = sitk.GetArrayFromImage(itkimage)

    numpyOrigin = np.array(list(reversed(itkimage.GetOrigin())))  # 世界坐标系的原点坐标
    numpySpacing = np.array(list(reversed(itkimage.GetSpacing())))  # 两个像素点之间的真实距离

    return numpyImage, numpyOrigin, numpySpacing, isflip

def show_image(numpy_image):
    image = np.squeeze(numpy_image[0, ...])
    plt.imshow(image, cmap='gray')
    plt.show()


# 计算两个bbox的iou
def iou(box0, box1):
    r0 = box0[3] / 2
    s0 = box0[:3] - r0
    e0 = box0[:3] + r0
    r1 = box1[3] / 2
    s1 = box1[:3] - r1
    e1 = box1[:3] + r1
    overlap = []
    for i in range(len(s0)):
        overlap.append(max(0, min(e0[i], e1[i]) - max(s0[i], s1[i])))
    intersection = overlap[0] * overlap[1] * overlap[2]
    union = box0[3] * box0[3] * box0[3] + box1[3] * box1[3] * box1[3] - intersection
    return intersection / union


# 非极大值抑制
def nms(output, nms_th):
    if len(output) == 0:
        return output
    output = output[np.argsort(-output[:, 0])]
    bboxes = [output[0]]
    for i in np.arange(1, len(output)):
        bbox = output[i]
        flag = 1
        for j in range(len(bboxes)):
            if iou(bbox[1:5], bboxes[j][1:5]) >= nms_th:
                flag = -1
                break
        if flag == 1:
            bboxes.append(bbox)
    bboxes = np.asarray(bboxes, np.float32)
    return bboxes