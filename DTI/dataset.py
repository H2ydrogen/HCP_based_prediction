import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torch.nn import functional as F
from DTI.utils import read_csv, export
import os
import numpy as np
import logging
import math

LOG = logging.getLogger('dataset')


# 统计列表的元素个数
def count_list(alist):
    data = alist
    data_dict = {}
    for key in data:
        data_dict[key] = data_dict.get(key, 0) + 1
    return data_dict


@export
def get_hcp_s1200(opt):
    root_dir = opt.data_path
    data = read_csv(os.path.join(root_dir, 'S1200_demographics_Restricted.csv'))
    data = [[line[0], line[1], line[8], line[10], line[12], line[22]] for line in data[1:]]
    # 0:subject, 1:Age, 8:Gender, 10:Race, 11:Ethnicity, 12:Handedness, 19:Height,  20:Weight, 22:BMICat

    # classify some of the data
    for i in range(len(data)):
        if int(data[i][1]) <= 21:
            data[i][1] = 0
        elif int(data[i][1]) <= 25:
            data[i][1] = 1
        elif int(data[i][1]) <= 30:
            data[i][1] = 2
        elif int(data[i][1]) <= 37:
            data[i][1] = 3
        if data[i][2] == 'M':
            data[i][2] = 0
        elif data[i][2] == 'F':
            data[i][2] = 1
        if data[i][3] == 'White':
            data[i][3] = 0
        elif data[i][3] == 'Black or African Am.':
            data[i][3] = 1
        else:
            data[i][3] = 2
        if int(data[i][4]) > 0:
            data[i][4] = 1
        elif int(data[i][4]) <= 0:
            data[i][4] = 0
        if data[i][5] == '':
            data[i][5] = 1
        data[i][5] = int(data[i][5])

    # 检查对应的数据文件是否存在，如果存在，把文件名加进去
    for i in reversed(range(len(data))):
        file1 = os.path.join(root_dir, 'UKF_2T_AtlasSpace', 'anatomical_tracts', str(data[i][0]) + '.csv')
        file2 = os.path.join(root_dir, 'UKF_2T_AtlasSpace', 'tracts_commissural', str(data[i][0]) + '.csv')
        file3 = os.path.join(root_dir, 'UKF_2T_AtlasSpace', 'tracts_left_hemisphere', str(data[i][0]) + '.csv')
        file4 = os.path.join(root_dir, 'UKF_2T_AtlasSpace', 'tracts_right_hemisphere', str(data[i][0]) + '.csv')
        if os.path.exists(file1) and os.path.exists(file2) and os.path.exists(file3) and os.path.exists(file4):
            data[i].append(file1)
            data[i].append(file2)
            data[i].append(file3)
            data[i].append(file4)
        else:
            data.pop(i)

    transform = transforms.Normalize(mean=0.05, std=0.5)  # mean 和 std是数据集的均值和方差，

    #  统计数据集
    # print(count_list([x[2] for x in data[1:]]))
    # print(count_list([x[7] for x in data[1:]]))

    return {
        'root_dir': root_dir,
        'data_list': data,
        'transform': transform
    }


# 装数据集的iterator的对象，可以不断next()出数据(x,y)
@export
class CreateDataset(Dataset):
    def __init__(self, opt, dataset, usage):
        self.root_dir = dataset['root_dir']
        self.data_list = dataset['data_list']
        self.transform = dataset['transform']
        self.fold_number = opt.FOLD_NUM
        self.opt = opt
        if usage == 'train':
            index = int(len(self.data_list) * (1.0 - 1.0 / self.fold_number))
            self.data_list = self.data_list[:index]
        elif usage == 'val':
            index = int(len(self.data_list) * (1.0 - 1.0 / self.fold_number))
            self.data_list = self.data_list[index:]

        self.data_list = self.data_list
        self.hemispheres = []
        if 'right-hemisphere' in self.opt.HEMISPHERES:
            self.hemispheres.append(-1)
        if 'left-hemisphere' in self.opt.HEMISPHERES:
            self.hemispheres.append(-2)
        if 'commissural' in self.opt.HEMISPHERES:
            self.hemispheres.append(-3)
        if 'anatomical' in self.opt.HEMISPHERES:
            self.hemispheres.append(-4)

        self.features = []
        if 'Num_Fibers' in self.opt.INPUT_FEATURES:
            self.features.append(1)
        if 'FA1-max' in self.opt.INPUT_FEATURES:  # row[9]
            self.features.append(8)
        if 'FA1-mean' in self.opt.INPUT_FEATURES: # row[10]
            self.features.append(9)
        if 'FA1-min' in self.opt.INPUT_FEATURES:  # row[12]
            self.features.append(11)
        if 'FA2-mean' in self.opt.INPUT_FEATURES:
            self.features.append(15)
        if 'Trace1-mean' in self.opt.INPUT_FEATURES:
            self.features.append(27)
        if 'Trace2-mean' in self.opt.INPUT_FEATURES:
            self.features.append(33)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # 读取idx对应文件，放入all_data
        all_data = np.zeros(1)
        for i in self.hemispheres:
            raw_data = read_csv(self.data_list[idx][i])  # (801,39)
            # 去除title，转换成np，维度标准化，data.shape=(38,800)
            data = np.array([row[1:] for row in raw_data[1:]]).astype(np.float).transpose()
            if i == -4:  # anatomical与其他不一样
                data = np.concatenate((data[0:26, :], data[32:44, :]))
            # 在dim=1连接所有数据
            if all_data.any():
                all_data = np.concatenate((all_data, data), axis=1)  # all_data.shape=(38,800*n)
            else:
                all_data = data

        # nan置0
        all_data[np.isnan(all_data)] = 0

        # 从all_data中取出想要的特征,归一化，然后放入x
        x = np.zeros(1)
        for i in self.features:
            feature = all_data[i, :][None]
            feature = (feature - feature.min()) / (feature.max() - feature.min())
            if x.any():
                x = np.concatenate((x, feature))  # x(n,num_cls)
                a = 0
            else:
                x = feature

        # 转换为二维
        if self.opt.MODEL == '2D-CNN' or self.opt.MODEL == 'Lenet':
            dim0, dim1 = x.shape
            size = math.ceil(dim1 ** 0.5)
            x = np.concatenate((x, np.zeros((dim0, size**2 - dim1))), axis=1)
            x = x.reshape((dim0, size, size))  # (n, size, size)

        # np->tensor
        x = torch.from_numpy(x).float()

        if self.opt.OUTPUT_FEATURES == 'sex':
            y = self.data_list[idx][2]
        elif self.opt.OUTPUT_FEATURES == 'race':
            y = self.data_list[idx][3]

        return {
            'x': x,  # size:(1,num_features)
            'y': torch.tensor(y)
        }





# 判断一个字符串是否为数字
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    return False