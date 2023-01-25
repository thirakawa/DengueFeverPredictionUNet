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


def save_args(filename, args):
    with open(filename, 'w') as f:
        json.dump(args.__dict__, f, indent=4)


def load_args(filename):
    import argparse
    with open(filename, 'r') as f:
        ns = json.load(f)
    return argparse.Namespace(**ns)


def parser():
    arg_parser = ArgumentParser()

    # basic settings
    arg_parser.add_argument('--data_dir', type=str, required=True,
                            help='directory stored demo data')
    arg_parser.add_argument('--logdir', type=str, required=True,
                            help='directory stored trained model and settings')
    arg_parser.add_argument('--arg_file', type=str, default='args.json',
                            help='json file name saved training settings')
    arg_parser.add_argument('--resume', type=str, default='snapshot-best.pt',
                            help='trained model file')

    # GPU settings
    arg_parser.add_argument('--gpu_id', type=str, default='0',
                            help='id(s) for CUDA_VISIBLE_DEVICES')

    args = arg_parser.parse_args()
    return args


def main():
    args = parser()

    # check train dir. and load args.json #################
    if not os.path.exists(args.logdir):
        print('ERROR: %s no such file or directory.' % args.logdir)
        sys.exit(-1)
    with open(os.path.join(args.logdir, args.arg_file)) as f:
        train_args = json.load(f)
    result_dir = os.path.join(args.logdir, "demo_result")
    os.makedirs(result_dir, exist_ok=True)

    if 'use_dengue' not in train_args.keys():
        train_args['use_dengue'] = True

    # GPU (device) settings ###############################
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    use_cuda = torch.cuda.is_available()

    # dataset #############################################
    kwargs = {'num_workers': 2, 'pin_memory': False} if use_cuda else {}
    # val (unused)
    val_dataset = DengueDataset(data_dir=args.data_dir, mode='val', delay=train_args['delay'],
                                  binary_label=True, use_dengue=train_args['use_dengue'],
                                  sst_type=train_args['sst_type'], use_precipitation=train_args['use_precipitation'],
                                  use_solar=train_args['use_solar'])
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, **kwargs)

    # test
    test_dataset = DengueDataset(data_dir=args.data_dir, mode='test', delay=train_args['delay'],
                                  binary_label=True, use_dengue=train_args['use_dengue'],
                                  sst_type=train_args['sst_type'], use_precipitation=train_args['use_precipitation'],
                                  use_solar=train_args['use_solar'])
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, **kwargs)

    # network model & optimizer ###########################
    model = UNet(in_channels=val_dataset.n_channels, n_class=2)

    # CPU or GPU
    if use_cuda:
        model = nn.DataParallel(model).cuda()
        cudnn.benchmark = True

    # load trained params
    model.load_state_dict(torch.load(os.path.join(args.logdir, args.resume)))

    eval_metrics = RunningScore(n_classes=2)

    model.eval()
    with torch.no_grad():
        for data, label, bname in test_loader:
            print("process:", bname)
            if use_cuda:
                data = data.cuda()

            output = model(data)

            if use_cuda:
                predicted_label = output.data.max(1)[1].cpu().numpy()
            else:
                predicted_label = output.data.max(1)[1].numpy()

            eval_metrics.update(label.numpy(), predicted_label)

            if use_cuda:
                pred = output.data.cpu().numpy()
            else:
                pred = output.data.numpy()

            visualized_result = np.argmax(pred, axis=1).squeeze().astype(np.uint8)
            visualized_result *= 255
            cv2.imwrite(os.path.join(result_dir, bname[0] + ".png"), visualized_result)

    score, class_iou = eval_metrics.get_scores()
    print("Overall score:\n", score)
    print("Class IoU:\n", class_iou, "\n")

    with open(os.path.join(result_dir, "score.txt"), 'w') as f_score:
        f_score.write('overall_acc: %f\n' % score["OverallAcc"])
        f_score.write('mean_acc: %f\n' % score["MeanAcc"])
        f_score.write('freqw_acc: %f\n' % score["FreqWAcc"])
        f_score.write('mean_iou: %f\n' % score["MeanIoU"])
        f_score.write('dice: %f\n' % score["Dice"])
        f_score.write('class_iou/others: %f\n' % class_iou[0])
        f_score.write('class_iou/dengue: %f\n' % class_iou[1])

    print("done.")


if __name__ == "__main__":
    main()
