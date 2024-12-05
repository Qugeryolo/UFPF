# encoding: utf-8
# @Time    : 2024/3/16 下午6:41
# @Author  : Geng Qin
# @File    : train_seg_oscc.py
import os
import numpy as np
import torch
import sys
import argparse
import logging
import random
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import CrossEntropyLoss
import torch.optim as optim
from tensorboardX import SummaryWriter
from dataset.OSCC_dataset import HSISegmentation, get_train_loader
from getnetwork import get_network
from losses.loss import *
from utils.lr_policy import WarmUpPolyLR
from metric.segmentation_metrics import calculate_metrics_compose

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default=r'/media/datasets1/Quger/datasets/OSCC', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='OSCC/H-vmunet', help='experiment_name')
parser.add_argument('--snapshot_dir', type=str,
                    default='OSCC_save_checkpoint', help='save model weight checkpoint')
parser.add_argument('--model', type=str,
                    default='H-vmunet', help='model_name')
parser.add_argument('--nepochs', type=int,
                    default=150, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=4,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int, default=1,
                    help='whether use deterministic training')
parser.add_argument('--transforms', type=str, default='light_aug',
                    help='patch size of network input')
parser.add_argument('--seed', type=int, default=42, help='random seed')
parser.add_argument('--num_workers', type=int, default=0,
                    help='the number of workers')
parser.add_argument('--num_classes', type=int, default=2,
                    help='output channel of network')
# label and unlabel
parser.add_argument('--train_source', type=str, default='segmentation',
                    help='labeled train path')
parser.add_argument('--unsup_source', type=str, default='segmentation',
                    help='unlabeled train path')
parser.add_argument('--fold', type=int, default=4,
                    help='k fold in datasets')
# settings
parser.add_argument('--ema_decay', type=float, default=0.99, help='ema_decay')
parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
parser.add_argument('--base_lr', type=float, default=0.0001,
                    help='segmentation network learning rate')
parser.add_argument('--lr_power', type=float, default=0.9,
                    help='power of learning rate')
parser.add_argument('--warm_up_epoch', type=int, default=0,
                    help='warm_up_epoch')
parser.add_argument('--momentum', type=float, default=0.99,
                    help='momentum of SGD optimizer')
parser.add_argument('--weight_decay', type=float, default=5e-3,
                    help='weight decay of SGD optimizer')
args = parser.parse_args()


device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

train_dataset = HSISegmentation(root=args.root_path, txt_name='fold4_train.txt', transforms=args.transforms)

test_dataset = HSISegmentation(root=args.root_path, txt_name='fold4_test.txt', transforms=args.transforms)

train_loader = get_train_loader(train_dataset, batch_size=args.batch_size, num_workers=0)

test_loader = get_train_loader(test_dataset, batch_size=args.batch_size, num_workers=0)


def kaiming_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model


def xavier_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model


def train(args, snapshot_path):
    def create_model():
        # Network definition

        model = get_network(args.model, 2)
        # print(model)

        return model

    model = create_model().to(device)
    model = kaiming_normal_init_weight(model)
    # model = xavier_normal_init_weight(model)

    model.train()
    optimizer = optim.SGD(model.parameters(),
                           lr=args.base_lr,
                           momentum=args.momentum,
                           weight_decay=args.weight_decay)
    ce_loss = CrossEntropyLoss(reduction='mean')
    dice_loss = Dice_Loss(reduction='mean')
    logger = SummaryWriter(snapshot_path + '/log')
    lr_policy = WarmUpPolyLR(args.base_lr, args.lr_power, args.nepochs,
                             args.warm_up_epoch)

    iter_num = 0

    for epoch in range(args.nepochs):
        loss_total = 0
        sum_loss_dice = 0
        sum_loss_ce = 0

        for index, batch in enumerate(train_loader):
            images, true_masks = batch
            images = images.to(device=device)
            true_masks = true_masks.long().to(device=device)
            pred_sup = model(images)
            # pred_sup = pred_sup[3]  # emcad

            celoss = ce_loss(pred_sup, true_masks)
            diceloss = dice_loss(pred_sup, true_masks)
            loss = celoss + diceloss
            loss_total += loss.item()
            sum_loss_dice += celoss.item()
            sum_loss_ce += diceloss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            current_idx = epoch
            lr_ = lr_policy.get_lr(current_idx)

            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

        logging.info(
            'Epoch: %03d/%03d || total_Loss: %.4f loss_dice: %.4f loss_ce: %.4f'
            % (epoch + 1, args.nepochs, loss_total / len(train_loader),
               sum_loss_dice / len(train_loader),
               sum_loss_ce / len(train_loader)))

        if (epoch % 5 == 0) and (epoch >= 130) or (epoch == args.nepochs - 1):
            model.eval()
            model_l_savedir = os.path.join(args.snapshot_dir, f'model_{args.model}_fold{args.fold}', )
            if not os.path.exists(model_l_savedir):
                os.makedirs(model_l_savedir)
            torch.save(model.state_dict(), os.path.join(model_l_savedir, f'fold{args.fold}_epoch{epoch}_{args.model}.pth'))
            model.train()

            """
            Args:
                    model test dataloader 
            :return
                    metrics: 
                    {
                'TPR': tpr,
                'TNR': tnr,
                'DICE': Dsc,
                'IOU': Jaccard,
                'ACC': acc,
                'Precision': Prec,
                'HD': hous_dis,
                'HD95': hous_dis95,
                'Recall': Recall,
                'sen': sen,
                'asd': asd
            }
            """

            metrics_sum = [0.0 for _ in range(9)]
            model.eval()
            for step, batch in enumerate(test_loader):
                images, labels = batch
                images = images.to(device)
                labels = labels.long().to(device)

                with torch.no_grad():
                    outs = model(images)
                output = F.interpolate(
                    outs, labels.shape[1:], mode="bilinear", align_corners=True
                )
                output = output.max(1)[1].cpu().numpy()
                label = labels.cpu().numpy()
                dict = calculate_metrics_compose(output, label)
                """=============calculate metrics=============="""

                metrics_sum = [sum_val + dict[metric] for sum_val, metric in zip(metrics_sum,
                                                                                 ['DICE', 'IOU', 'ACC',
                                                                                  'Precision', 'HD', 'HD95',
                                                                                  'Recall', 'sen', 'asd'])]

            metrics_avg = [sum_val / len(test_loader) for sum_val in metrics_sum]

            logging.info(
                'Epoch: %03d/%03d || model1---DSC: %.4f IoU: %.4f '
                'ACC: %.4f Pre: %.4f HD: %.4f HD95: %.4f  Recall: %.4f Sen: %.4f ASD: %.4f'
                % (epoch + 1, args.nepochs, metrics_avg[0], metrics_avg[1],
                   metrics_avg[2], metrics_avg[3], metrics_avg[4], metrics_avg[5], metrics_avg[6], metrics_avg[7], metrics_avg[8]))
            model.train()

    logger.close()
    print('all done')


if __name__ == "__main__":

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    snapshot_path = "./{}_{}/{}".format(
        args.exp, args.fold, args.model)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    '''
    Python logging  
        %(asc time)s表示日志记录的时间,
        %(mse cs)03d表示时间戳的毫秒部分,
        %(message)s表示日志消息,
        date fmt='%H:%M:%S'定义了时间的格式。这里使用24小时制的时间格式,
        将时间按照小时、分钟和秒的顺序显示
    '''
    logging.basicConfig(filename=snapshot_path + "/log_{}.txt".format(args.model), level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)
