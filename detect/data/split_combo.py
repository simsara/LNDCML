import numpy as np

class SplitComb():  
    """
    输入数据的分割与合并，分割成小立方体再进行合并
    """
    def __init__(self,side_len,max_stride,stride,margin,pad_value):
        self.side_len = side_len  # 每一个小立方体的边长
        self.max_stride = max_stride  # 最长每隔多少切一个小立方体（最大值步长）
        self.stride = stride  # 每隔多少切一个小立方体
        self.margin = margin  # 增加的边缘值
        self.pad_value = pad_value  # 水的HU值
        
    def split(self, data, side_len = None, max_stride = None, margin = None):
        if side_len==None:  
            side_len = self.side_len
        if max_stride == None:
            max_stride = self.max_stride
        if margin == None:
            margin = self.margin
        
        assert(side_len > margin)  # 边长要大于增加的边缘值
        assert(side_len % max_stride == 0)  # 边长是最大步长的整数倍
        assert(margin % max_stride == 0)  # 增加的边缘是最大步长的整数倍

        splits = []
        _, z, h, w = data.shape # 数据的原始长宽高

        nz = int(np.ceil(float(z) / side_len)) # z轴方向上切多少份
        nh = int(np.ceil(float(h) / side_len)) # y轴方向上切多少份
        nw = int(np.ceil(float(w) / side_len)) # x轴方向上切多少份
        
        nzhw = [nz,nh,nw] # 三个维度构成的多少个立方体
        self.nzhw = nzhw
        
        pad = [ [0, 0],
                [margin, nz * side_len - z + margin],
                [margin, nh * side_len - h + margin],
                [margin, nw * side_len - w + margin]]
        # data是待填充数组，pad是填充的形状，（2,3）表示前面2个，后面3个，按边缘值进行填充
        data = np.pad(data, pad, 'edge')

        for iz in range(nz):
            for ih in range(nh):
                for iw in range(nw):
                    sz = iz * side_len  # sz 表示 start  ez 表示end
                    ez = (iz + 1) * side_len + 2 * margin
                    sh = ih * side_len
                    eh = (ih + 1) * side_len + 2 * margin
                    sw = iw * side_len
                    ew = (iw + 1) * side_len + 2 * margin

                    # np.newaxis增加了一维，使它为一个整体，split为切分出的一个小的立方体
                    split = data[np.newaxis, :, sz:ez, sh:eh, sw:ew]  # np.newaxis增加了一维，使它为一个整体
                    splits.append(split)

        splits = np.concatenate(splits, 0)
        return splits,nzhw  # 返回切分好的小立方体集合和每条边的切分个数

    def combine(self, output, nzhw = None, side_len=None, stride=None, margin=None):
        """
        将一个个小立方体还原成一个大的立方体
        """
        
        if side_len==None:
            side_len = self.side_len
        if stride == None:
            stride = self.stride
        if margin == None:
            margin = self.margin
        if nzhw==None:  # nxyz为坐标轴上的立方体个数
            nz = self.nz
            nh = self.nh
            nw = self.nw
        else:
            nz,nh,nw = nzhw

        assert(side_len % stride == 0)  # 边长是步长的整数倍
        assert(margin % stride == 0)  # 增加的边缘是步长的整数倍

        side_len /= stride
        margin /= stride

        splits = []
        for i in range(len(output)):
            splits.append(output[i])  # 将output看成是已经切分好的小立方体

        output = -1000000 * np.ones((
            nz * side_len,
            nh * side_len,
            nw * side_len,
            splits[0].shape[3],
            splits[0].shape[4]), np.float32)

        idx = 0
        for iz in range(nz):
            for ih in range(nh):
                for iw in range(nw):
                    sz = iz * side_len
                    ez = (iz + 1) * side_len
                    sh = ih * side_len
                    eh = (ih + 1) * side_len
                    sw = iw * side_len
                    ew = (iw + 1) * side_len

                    split = splits[idx][margin:margin + side_len, margin:margin + side_len, margin:margin + side_len]
                    output[sz:ez, sh:eh, sw:ew] = split   # 进行还原
                    idx += 1

        return output 