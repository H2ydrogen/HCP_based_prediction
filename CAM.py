# simple implementation of CAM in PyTorch for the networks such as ResNet, DenseNet, SqueezeNet, Inception
# last update by BZ, June 30, 2021
# modified by Hao, Jul 16, 2021

import logging
import time

from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch
import torch.nn.functional as functional
import numpy as np
from DTI import models, dataset, cli, utils, analyse
import os
import io
from PIL import Image
from torchvision import models, transforms
from torch.autograd import Variable
from torch.nn import functional as F
import cv2
import json
from DTI import models, cli
args = cli.create_parser().parse_args()
args.batch_size = 1
args.LOAD_PATH = '.\\LOG\\20210716-211828.pkl'

def main():
    # input image
    utils.setup_seed(14)




    # 数据集加载
    train_set = dataset.CreateDataset(dataset.get_hcp_s1200(), usage='train')
    val_set = dataset.CreateDataset(dataset.get_hcp_s1200(), usage='val')
    train_loader = DataLoader(train_set, batch_size=args.batch_size, drop_last=False, shuffle=True, pin_memory=True,
                              num_workers=args.num_workers)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, drop_last=True, shuffle=True, pin_memory=True,
                            num_workers=args.num_workers)

    # 网络加载
    # if args.MODEL == '1D-CNN':
    if args.INPUT_FEATURES != '4' and args.INPUT_FEATURES != 'all':
        c = 1
    else:
        c = 4

    if args.MODEL == '1D-CNN':
        net = models.HARmodel(c, args.NUM_CLASSES)
        finalconv_name = 'features'
    elif args.MODEL == 'CAM-CNN':
        net = models.CAM_CNN(c, args.NUM_CLASSES)
        finalconv_name = 'features'
    else:
        net = None

    net.load_state_dict(torch.load(args.LOAD_PATH))
    net.eval()

    # hook the feature extractor
    features_blobs = []
    def hook_feature(module, input, output):
        features_blobs.append(output.data.cpu().numpy())

    net._modules.get(finalconv_name).register_forward_hook(hook_feature)

    # get the softmax weight
    params = list(net.parameters())
    weight_softmax = np.squeeze(params[-2].data.numpy())


    def returnCAM(feature_conv, weight_softmax, class_idx):
        # generate the class activation maps
        for idx in class_idx:
            cam = weight_softmax[idx].dot(feature_conv)
            cam_normalize = cam - np.min(cam)
            cam_standard = cam / np.max(cam)
        return cam


    # load test image
        cam = torch.zeros(788)


    for batch_index, batch_samples in enumerate(train_loader):
        x, y = batch_samples['x'], batch_samples['y']
        if args.INPUT_FEATURES != '4' and args.INPUT_FEATURES != 'all':
            x = x.unsqueeze(1)
        else:
            x = x.transpose(1, 2)

        # forward
        logit = net(x)

        h_x = F.softmax(logit, dim=1).data.squeeze()
        probs, idx = h_x.sort(0, True)
        probs = probs.numpy()
        idx = idx.numpy()


    # generate class activation mapping for the top1 prediction
    CAMs = returnCAM(features_blobs[0], weight_softmax, [idx[0]])
    print(CAMs)

if __name__ == '__main__':
    main()