import logging
import time

from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch
import torch.nn.functional as functional
import numpy as np
from DTI import models, dataset, cli, utils, analyse, visualizer
import operator
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def main():
    utils.setup_seed(123)
    start_time = time.time()
    vis = visualizer.Visualizer(args)

    # 数据集加载
    train_set = dataset.CreateDataset(args, dataset.get_hcp_s1200(args), usage='train')
    val_set = dataset.CreateDataset(args, dataset.get_hcp_s1200(args), usage='val')
    train_loader = DataLoader(train_set, batch_size=args.batch_size, drop_last=False, shuffle=True, pin_memory=True,
                              num_workers=args.num_workers)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, drop_last=True, shuffle=True, pin_memory=True,
                            num_workers=args.num_workers)

    # 网络加载

    if args.MODEL == '1D-CNN':
        model = models.HARmodel(args).to(device)
    elif args.MODEL == '2D-CNN':
        model = models.CNN_2D(args).to(device)
    elif args.MODEL == 'Lenet':
        model = models.Lenet(args).to(device)
    else:
        model = None

    # if os.path.exists(args.LOAD_PATH):
    #     model.load_state_dict(torch.load(args.LOAD_PATH))
    optimizer = torch.optim.SGD(model.parameters(), lr=args.LR, weight_decay=args.L2)

    # 训练
    val_loss = []
    train_loss = []
    val_acc = []
    train_acc = []
    val_precision = []
    val_recall = []

    # 打印训练信息
    LOG.info("Args:{}".format(args))

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

        # 打印最好成绩
        max_train_acc_index, max_train_acc = max(enumerate(train_acc), key=operator.itemgetter(1))
        max_val_acc_index, max_val_acc = max(enumerate(val_acc), key=operator.itemgetter(1))
        min_train_loss_index, min_train_loss = min(enumerate(train_loss), key=operator.itemgetter(1))
        print('best train_loss；{}({}epoch)---best t_acc:{}({}epoch)---best val_acc；{}({}epoch),'
              .format(min_train_loss, min_train_loss_index+1, max_train_acc, max_train_acc_index+1, max_val_acc, max_val_acc_index+1,))

        # 训练结果可视化
        vis.display_train_result(train_results['train_loss']/100, train_results['train_acc'], val_results['val_acc'], epoch)
    # 训练结果存档
    # torch.save(model.state_dict(), '.\\LOG\\{}.pkl'.format(args.RECORD_NAME))
    f = open('.\\LOG\\{}.txt'.format(args.RECORD_NAME), 'a+')
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
        length = len(dataloader)
        x, y = batch_samples['x'].to(device), batch_samples['y'].to(device)  # x.size = (bs,C, len); y.size = (bs)

        # 2.forward
        criteria = nn.CrossEntropyLoss()
        output = model(x)  # output.size = (bs, 3)
        # score = functional.softmax(output, dim=1)
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
        'train_loss': round(train_loss.tolist(), 4),
        'train_acc': round(train_correct / len(dataloader.dataset), 4)
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

    LOG.info("--- validation epoch finish in {} seconds --- Loss:{}\tCorrect:{}/{}({})"
             .format(time.time() - start_time, test_loss, correct, len(val_loader.dataset), correct / len(val_loader.dataset)))
    val_result = analyse.analyse_3class(target_list, pred_list)

    return {
        'val_loss': round(test_loss.cpu().numpy().tolist(), 4),
        'val_acc': round(correct / len(val_loader.dataset), 4),
        'val_precision': val_result['precision'],
        'val_recall': val_result['recall'],
    }


if __name__ == '__main__':
    LOG = logging.getLogger('main')
    logging.basicConfig(level=logging.INFO)
    args = cli.create_parser().parse_args()
    main()
