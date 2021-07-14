import logging
import time

from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch
import torch.nn.functional as functional
import numpy as np
import copy
from DTI import models, dataset, cli, utils, analyse
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def main():
    utils.setup_seed(18)
    start_time = time.time()

    # 数据集加载
    train_set = dataset.CreateDataset(dataset.get_hcp_s1200(), usage='train')
    val_set = dataset.CreateDataset(dataset.get_hcp_s1200(), usage='val')
    train_loader = DataLoader(train_set, batch_size=args.batch_size, drop_last=False, shuffle=True, pin_memory=True,
                              num_workers=args.num_workers)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, drop_last=False, shuffle=True, pin_memory=True,
                            num_workers=args.num_workers)

    # 网络加载
    # model = models.CNN(800, 2).to(device)
    # if args.MODEL == '1D-CNN':
    model = models.HARmodel(1, 2).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.002)

    # 训练
    val_loss = []
    train_loss = []
    val_acc = []
    train_acc = []
    val_precision = []
    val_recall = []

    for epoch in range(args.epochs):
        train_results = train(train_loader, model, optimizer, epoch)
        val_results = validation(model, val_loader)

        # 训练结果记录
        train_loss.append(train_results['train_loss'])
        train_acc.append(train_results['train_acc'])
        val_loss.append(val_results['val_loss'])
        val_acc.append(val_results['val_acc'])
        val_precision.append(val_results['val_precision'])
        val_recall.append(val_results['val_recall'])

    # 训练结果存档
    f = open(args.RECORD_PATH, 'a+')
    f.writelines('args'+str(args)+'\n')
    f.writelines('train_loss'+str(train_loss)+'\n')
    f.writelines('train_acc' + str(train_acc)+'\n')
    f.writelines('val_loss' + str(val_loss)+'\n')
    f.writelines('val_acc' + str(val_acc)+'\n')
    f.writelines('val_precision' + str(val_precision)+'\n')
    f.writelines('val_recall' + str(val_recall)+'\n')
    f.close()

    LOG.info("--- main.py finish in %s seconds ---" % (time.time() - start_time))

def train(dataloader, model, optimizer, epoch):
    model.train()
    train_loss = 0
    train_correct = 0
    start_time = time.time()
    for batch_index, batch_samples in enumerate(dataloader):
        # 1.load data to CUDA
        x, y = batch_samples['x'].to(device), batch_samples['y'].to(device)
        x = x.unsqueeze(1)

        # 2.forward
        output = model(x)
        criteria = nn.CrossEntropyLoss()
        loss = criteria(output, y.long())

        # 3.backward
        optimizer.zero_grad()  # 把所有Variable的grad成员数值变为0
        loss.backward()  # 反向传播grad
        optimizer.step()  # 每个Variable的grad都被计算出来后，更新每个Variable的数值（优化更新）

        # 6.result
        pred = output.argmax(dim=1, keepdim=True)
        train_correct += pred.eq(y.long().view_as(pred)).sum().item()
        train_loss += loss
        if batch_index % args.display_batch == 0:
            LOG.info("--- training progress rate {}/{} ---".format(batch_index, len(dataloader)))

    LOG.info("--- training epoch {} finish in {} seconds ---".format(epoch, (time.time() - start_time)))
    LOG.info('\tLoss:{}\tCorrect:{}/{}({})'
             .format(train_loss, train_correct, len(dataloader.dataset),
                     train_correct / len(dataloader.dataset)))

    return {
        'train_loss': train_loss,
        'train_acc': train_correct / len(dataloader.dataset)
    }


def validation(model, val_loader):
    model.eval()
    test_loss = 0
    correct = 0
    start_time = time.time()
    with torch.no_grad():
        pred_list = []
        target_list = []
        for batch_index, batch_samples in enumerate(val_loader):
            #  1.load data to CUDA
            x, y = batch_samples['x'].to('cuda'), batch_samples['y'].to('cuda')
            x = x.unsqueeze(1)

            # 2.forward
            output = model(x)
            pred = output.argmax(dim=1, keepdim=True)

            #  3.result
            criteria = nn.CrossEntropyLoss()
            test_loss += criteria(output, y)
            correct += pred.eq(y.view_as(pred)).sum().item()

            y = y.cpu().numpy()
            pred_list = np.append(pred_list, pred.cpu().numpy())
            target_list = np.append(target_list, y)

            val_result = analyse.analyse_3class(target_list, pred_list)

    LOG.info("--- validation epoch finish in %s seconds ---" % (time.time() - start_time))
    return {
        'val_loss': test_loss,
        'val_acc': test_loss / len(val_loader.dataset),
        'val_precision': val_result['precision'],
        'val_recall': val_result['recall'],
    }


if __name__ == '__main__':
    LOG = logging.getLogger('main')
    logging.basicConfig(level=logging.INFO)
    args = cli.create_parser().parse_args()
    main()
