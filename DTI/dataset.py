import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torch.nn import functional as F
from DTI.utils import read_csv, export
import os
import numpy as np
import logging
from PIL import Image
import copy
from DTI import cli
import random
import collections

args = cli.create_parser().parse_args()
LOG = logging.getLogger('dataset')


# 统计列表的元素个数
def count_list(alist):
    data = alist
    data_dict = {}
    for key in data:
        data_dict[key] = data_dict.get(key, 0) + 1
    return data_dict


@export
def get_hcp_s1200():
    root_dir = args.data_path
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

    #  统计数据集
    # print(count_list([x[2] for x in data[1:]]))
    # print(count_list([x[7] for x in data[1:]]))

    return {
        'root_dir': root_dir,
        'data_list': data
    }


# 装数据集的iterator的对象，可以不断next()出数据(x,y)
@export
class CreateDataset(Dataset):
    def __init__(self, dataset, usage):
        self.root_dir = dataset['root_dir']
        self.data_list = dataset['data_list']
        self.fold_number = 10
        if usage == 'train':
            index = int(len(self.data_list) * (1.0 - 1.0 / self.fold_number))
            self.data_list = self.data_list[:index]
        elif usage == 'val':
            index = int(len(self.data_list) * (1.0 - 1.0 / self.fold_number))
            self.data_list = self.data_list[index:]

        self.data_list = self.data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        data = read_csv(self.data_list[idx][-1])
        data = np.array([row[1:] for row in data[1:]]).astype(np.float).transpose()  # np.size()=(38,800)
        x = np.zeros((1, 800))

        if 'Num_Fibers' in args.INPUT_FEATURES:
            x = np.concatenate((x, data[1, :][None]))
        if 'FA1-mean' in args.INPUT_FEATURES: # row[10]
            x = np.concatenate((x, data[9, :][None]))
        if 'FA2-mean' in args.INPUT_FEATURES:
            x = np.concatenate((x, data[15, :][None]))
        if 'Trace1-mean' in args.INPUT_FEATURES:
            x = np.concatenate((x, data[29, :][None]))
        if 'Trace2-mean' in args.INPUT_FEATURES:
            x = np.concatenate((x, data[33, :][None]))
        x = x[1:]
        x[~(x > -999999)] = 0

        if 'All' in args.INPUT_FEATURES:
            x = data

        if args.OUTPUT_FEATURES == 'sex':
            y = self.data_list[idx][2]
        elif args.OUTPUT_FEATURES == 'race':
            y = self.data_list[idx][3]

        return {
            'x': torch.from_numpy(x),  # size:
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