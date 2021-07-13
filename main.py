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
from DTI import models, dataset, cli, utils
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def main():
    utils.setup_seed(18)

    # 数据集加载
    train_set = dataset.CreateDataset(dataset.get_hcp_s1200(), usage='train')
    val_set = dataset.CreateDataset(dataset.get_hcp_s1200(), usage='val')
    train_loader = DataLoader(train_set, batch_size=args.batch_size, drop_last=False, shuffle=True, pin_memory=True,
                              num_workers=args.num_workers)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, drop_last=False, shuffle=True, pin_memory=True,
                            num_workers=args.num_workers)

    # 网络加载
    model = models.CNN(800, 5).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    # 训练
    for epoch in range(args.epochs):
        train(train_loader, model, optimizer, epoch)
        validation(model, val_loader)


def train(dataloader, model, optimizer, epoch):
    model.train()
    train_loss = 0
    train_correct = 0
    start_time = time.time()
    counter = 0
    for batch_index, batch_samples in enumerate(dataloader):
        # 1.load data to CUDA
        x = batch_samples['x']
        y = batch_samples['y']
        x, y = batch_samples['x'].to(device), batch_samples['y'].to(device)

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
             .format(train_loss / len(dataloader.dataset), train_correct, len(dataloader.dataset),
                     train_correct / len(dataloader.dataset)))

    return True


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
    LOG.info("--- validation epoch finish in %s seconds ---" % (time.time() - start_time))
    return {
        'target_list': target_list,
        'pred_list': pred_list
    }


if __name__ == '__main__':
    LOG = logging.getLogger('main')
    logging.basicConfig(level=logging.INFO)
    args = cli.create_parser().parse_args()
    main()
