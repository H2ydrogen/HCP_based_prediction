import logging
import time
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch
import torch.nn.functional as functional
import numpy as np
import matplotlib.pyplot as plt
from DTI import models, dataset, utils, analyse, cli
import os
import argparse

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def main():
    utils.setup_seed(9)
    start_time = time.time()

    # 数据集加载
    data_set = dataset.CreateDataset(dataset.get_hcp_s1200(), usage='')
    loader = DataLoader(data_set, batch_size=args.batch_size, drop_last=False, shuffle=True, pin_memory=True,
                        num_workers=args.num_workers)

    # 开始验证
    validation(None, loader)


def validation(model, val_loader):
    Mx = np.zeros((1))
    Fx = np.zeros((1))
    My = np.zeros((1))
    Fy = np.zeros((1))
    CLUSTER1 = 242
    CLUSTER2 = 91

    for batch_index, batch_samples in enumerate(val_loader):
        #
        x, y = batch_samples['x'].squeeze().detach().numpy(), batch_samples['y'].detach().numpy()
        for i in range(len(y)):
            array = x[i][739]
            if y[i] == 0:
                Mx = np.append(Mx, x[i][CLUSTER1-1][np.newaxis], axis=0)
                My = np.append(My, x[i][CLUSTER2-1][np.newaxis], axis=0)
            else:
                Fx = np.append(Fx, x[i][CLUSTER1-1][np.newaxis], axis=0)
                Fy = np.append(Fy, x[i][CLUSTER2-1][np.newaxis], axis=0)


    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    # matplotlib画图中中文显示会有问题，需要这两行设置默认字体

    plt.xlabel('cluster_{}'.format(CLUSTER1))
    plt.ylabel('cluster_{}'.format(CLUSTER2))
    plt.xlim(xmax=0.7, xmin=0.2)
    plt.ylim(ymax=0.9, ymin=0.5)
    # 画两条（0-9）的坐标轴并设置轴标签x，y


    colors1 = '#00CED1'  # 点的颜色
    colors2 = '#DC143C'
    area = np.pi * 4 ** 0.02  # 点面积
    # 画散点图
    plt.scatter(Mx[1:], My[1:], s=area, c=colors1, alpha=0.4, label='Male')
    plt.scatter(Fx[1:], Fy[1:], s=area, c=colors2, alpha=0.4, label='Female')
    plt.legend()
    plt.show()
    return


if __name__ == '__main__':
    LOG = logging.getLogger('Occ')
    logging.basicConfig(level=logging.INFO)
    args = cli.create_parser().parse_args()
    main()
