import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torch.nn import functional as F
from utils import read_csv, export
import os
import numpy as np
import logging
from PIL import Image
import copy
from DTI import cli
import random

args = cli.create_parser().parse_args()

#统计列表的元素个数
def count_list(alist):
    data = alist
    data_dict = {}
    for key in data:
        data_dict[key] = data_dict.get(key, 0) + 1
    return data_dict


@export
def HCP_S1200():
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

    # check if data exist
    for i in reversed(range(len(data))):
        if not os.path.exists(
                os.path.join(root_dir, 'UKF_2T_AtlasSpace', 'tracts_right_hemisphere', str(data[i][0])) + '.csv') or\
           not os.path.exists(
                os.path.join(root_dir, 'UKF_2T_AtlasSpace', 'tracts_left_hemisphere', str(data[i][0])) + '.csv') or\
           not os.path.exists(
                os.path.join(root_dir, 'UKF_2T_AtlasSpace', 'anatomical_tracts', str(data[i][0])) + '.csv') or\
           not os.path.exists(
                os.path.join(root_dir, 'UKF_2T_AtlasSpace', 'tracts_commissural', str(data[i][0])) + '.csv'):
            data.pop(i)

    # 统计数据集
    for i in range(len(data[0])):
        print(count_list([x[i] for x in data[1:]]))

    return {
        'data': data,
        'classes': ['COVID-19', 'Common', 'Normal'],
        'folds': ['01.txt', '02.txt', '03.txt', '04.txt'],
    }

dataset = HCP_S1200()


