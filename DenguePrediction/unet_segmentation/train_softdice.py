  
#!/usr/bin/env python3

from __future__ import print_function

import sys
sys.path.append("../")

import os
import json
from time import time
from argparse import ArgumentParser
import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from model import UNet
from dataset import DengueDataset
from loss import SoftDiceLoss
from metrics import RunningScore


def parser():
    arg_parser = ArgumentParser()

    # training settings
    arg_parser.add_argument('--batch_size', type=int, default=8,
                            help='batch size for each iteration')
    arg_parser.add_argument('--epochs', type=int, default=2000,
                            help='the number of epochs to train')
    arg_parser.add_argument('--logdir', type=str, default=None,
                            help='directory to save tensorboradX log')
    arg_parser.add_argument('--lr', type=float, default=0.01,
                            help='learning rate')

    # dataset settings
    arg_parser.add_argument('--data_dir', type=str, required=True,
                            help='path to data directory')
    arg_parser.add_argument('--use_dengue', action='store_true',
                            help='if use dengue data as input')
    arg_parser.add_argument('--sst_type', choices=['day', 'night', 'both', 'none'],
                            default='day',
                            help='sea surface data type')
    arg_parser.add_argument('--use_precipitation', action='store_true',
                            help='if use precipitation data')
    arg_parser.add_argument('--use_solar', action='store_true',
                            help='if use solar radiation data')
    arg_parser.add_argument('--delay', type=int, default=1,
                            help='time delay for label')
    arg_parser.add_argument('--n_worker', type=int, default=8,
                            help='the number of workder for data loader')

    # GPU settings
    arg_parser.add_argument('--gpu_id', type=str, default='0',
                            help='id(s) for CUDA_VISIBLE_DEVICES')

    args = arg_parser.parse_args()
    return args


def main():
    args = parser()

    # GPU (device) settings ###############################
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    use_cuda = torch.cuda.is_available()

    # dataset #############################################
    kwargs = {'num_workers': args.n_worker, 'pin_memory': False} if use_cuda else {}
    # train
    train_dataset = DengueDataset(data_dir=args.data_dir, mode='train', delay=args.delay,
                                  binary_label=True, use_dengue=args.use_dengue,
                                  sst_type=args.sst_type, use_precipitation=args.use_precipitation,
                                  use_solar=args.use_solar)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)

    # val
    val_dataset = DengueDataset(data_dir=args.data_dir, mode='val', delay=args.delay,
                                  binary_label=True, use_dengue=args.use_dengue,
                                  sst_type=args.sst_type, use_precipitation=args.use_precipitation,
                                  use_solar=args.use_solar)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, **kwargs)

    # test (unused)
    test_dataset = DengueDataset(data_dir=args.data_dir, mode='test', delay=args.delay,
                                  binary_label=True, use_dengue=args.use_dengue,
                                  sst_type=args.sst_type, use_precipitation=args.use_precipitation,
                                  use_solar=args.use_solar)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, **kwargs)

    # network model & optimizer ###########################
    model = UNet(in_channels=train_dataset.n_channels, n_class=2)

    # optimizer ###########################################
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # loss function
    criterion = SoftDiceLoss()

    # CPU or GPU
    if use_cuda:
        model = nn.DataParallel(model).cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    # tensorboardX ######################################
    writer = SummaryWriter(log_dir=args.logdir)
    log_dir = writer.file_writer.get_logdir()
    with open(os.path.join(log_dir, 'args.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=4)

    #######################################################
    # the beginning of train loop
    #######################################################
    iteration = 1
    best_score = 0.0
    loss_sum = 0.0
    _start_time = time()
    for epoch in range(1, args.epochs + 1):
        print("epoch:", epoch)

        # train #######################
        model.train()
        for data, label, _ in train_loader:
            if use_cuda:
                data = data.cuda()
                label = label.cuda()
            
            output = model(data)
            loss = criterion(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()

            if iteration % 100 == 0:
                print("iteration: %06d loss: %0.8f elapsed time: %0.1f" % (iteration,
                                                                           loss_sum / 100.0,
                                                                           time() - _start_time))
                writer.add_scalar('loss/train', loss_sum / 100.0, iteration)
                loss_sum = 0.0

            iteration += 1

        # evaluation ##################
        if epoch % 50 == 0:
            print("\nEvaluation ...")
            print("   Validation data ...")
            val_score, val_class_iou = evaluation(model, val_loader, use_cuda)
            writer.add_scalar('val_accuracy/overall_acc', val_score["OverallAcc"], epoch)
            writer.add_scalar('val_accuracy/mean_acc', val_score["MeanAcc"], epoch)
            writer.add_scalar('val_accuracy/freqw_acc', val_score["FreqWAcc"], epoch)
            writer.add_scalar('val_accuracy/mean_iou', val_score["MeanIoU"], epoch)
            writer.add_scalar('val_accuracy/dice', val_score["Dice"], epoch)
            writer.add_scalar('val_class_iou/others', val_class_iou[0], epoch)
            writer.add_scalar('val_class_iou/dengue', val_class_iou[1], epoch)

            if best_score < val_score["Dice"]:
                print("save best model...")
                best_score = val_score["Dice"]
                torch.save(model.state_dict(), os.path.join(log_dir, "snapshot-best.pt"))
        ############################################

        # save model ##################
        if epoch % 500 == 0:
            print("\nsave model...\n")
            torch.save(model.state_dict(), os.path.join(log_dir, "snapshot-%04d.pt" % epoch))

        print("epoch:", epoch, "; done.\n")

    #######################################################
    # the end of train loop
    #######################################################

    # save final model & close writer
    torch.save(model.state_dict(), os.path.join(log_dir, "snapshot-final.pt"))
    writer.close()

    print("done.")


# evaluation for classificaiton problem
def evaluation(model, data_loader, use_cuda):
    eval_metrics = RunningScore(n_classes=2)
    model.eval()
    with torch.no_grad():
        for data, label, _ in data_loader:
            if use_cuda:
                data = data.cuda()

            output = model(data)
            
            if use_cuda:
                pred = output.data.max(1)[1].cpu().numpy()
            else:
                pred = output.data.max(1)[1].numpy()

            eval_metrics.update(label.numpy(), pred)

    score, class_iou = eval_metrics.get_scores()

    print("Overall score:\n", score)
    print("Class IoU:\n", class_iou, "\n")

    return score, class_iou


if __name__ == "__main__":
    main()
