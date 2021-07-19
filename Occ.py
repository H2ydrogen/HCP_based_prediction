import logging
import time
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch
import torch.nn.functional as functional
import numpy as np
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

    # 网络加载
    if args.INPUT_FEATURES != '4' and args.INPUT_FEATURES != 'all':
        c = 1
    else:
        c = 4
    if args.MODEL == '1D-CNN':
        model = models.HARmodel(c, args.NUM_CLASSES).to(device)
    else:
        model = None
    # model.load_state_dict(torch.load(args.LOAD_PATH))
    model.eval()

    # 训练

    # 开始验证
    LOG.info("Args:{}".format(args))
    drop, idx = validation(model, loader)[1:].sort(0, True)
    drop = drop.numpy().tolist()
    idx = idx.numpy().tolist()

    # 结果存档
    f = open('.\\LOG\\rank\\Occ{}.csv'.format(time.strftime("%Y%m%d-%H%M%S", time.localtime())), 'a+')
    for i in range(len(idx)):
        f.writelines(str(idx[i]) + ',' + str(drop[i]) + '\n')
    f.close()
    LOG.info("--- Occ.py finish in %s seconds ---" % (time.time() - start_time))


def validation(model, val_loader):
    drop = torch.zeros(801)
    num_sample = 0
    with torch.no_grad():
        for batch_index, batch_samples in enumerate(val_loader):
            #
            x, y = batch_samples['x'].to('cuda'), batch_samples['y'].to('cuda')  # x.size = (bs,C, len); y.size = (bs)
            num_sample += len(y)
            # 2.forward
            output = model(x)  # output.size = (bs, 2)
            score = functional.softmax(output, dim=1)
            conf = torch.max(score, dim=1).values
            pred = output.argmax(dim=1, keepdim=True)

            confidence_drop = torch.zeros(1)
            for cluster in range(800):
                x_new = x.clone()
                for i in range(0, 1):
                    x_new[:, :, (cluster + i) % 800] = 0
                new_output = model(x_new)  # new_output.size = (bs,2)
                new_score = functional.softmax(new_output, dim=1)
                new_conf = torch.max(new_score, dim=1).values
                new_pred = new_output.argmax(dim=1, keepdim=True)
                confidence_drop = torch.cat((confidence_drop, torch.sum(conf - new_conf).unsqueeze(0).cpu()), 0)
            drop += confidence_drop
            if batch_index % args.display_batch == 0:
                LOG.info("--- training progress rate {}/{} ---".format(batch_index, len(val_loader)))

    drop_mean = drop / num_sample
    return drop_mean


if __name__ == '__main__':
    LOG = logging.getLogger('Occ')
    logging.basicConfig(level=logging.INFO)
    args = cli.create_parser().parse_args()
    main()
