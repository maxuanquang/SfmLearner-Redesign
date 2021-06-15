# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import argparse
from typing import TYPE_CHECKING

file_dir = os.path.dirname(__file__)  # the directory that options.py resides in


class SfmLearnerConfig:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="SfmLearner configuration")

        # CONFIG DATALOADER
        self.parser.add_argument('--data',
                                type=str,
                                default='/content/resulting_formatted_data_full_sfmlearner_pytorch',
                                help='path to dataset')
        self.parser.add_argument('--dataset-format', 
                                type=str,
                                default='sequential',
                                help='dataset format, stacked: stacked frames (from original TensorFlow code) sequential: sequential folders (easier to convert to with a non KITTI/Cityscape dataset')
        self.parser.add_argument('--sequence-length', 
                                type=int, 
                                default=3,                                 
                                help='sequence length for training')
        self.parser.add_argument('--rotation-mode', 
                                type=str, 
                                choices=['euler', 'quat'], 
                                default='euler',
                                help='rotation mode for PoseExpnet : euler (yaw,pitch,roll) or quaternion (last 3 coefficients)')
        self.parser.add_argument('--padding-mode', 
                                type=str, 
                                choices=['zeros', 'border'], 
                                default='zeros',
                                help='padding mode for image warping : this is important for photometric differenciation when going outside target image, zeros will null gradients outside target image, border will only null gradients of the coordinate outside (x or y)')
        self.parser.add_argument('-j', '--workers', 
                                default=4, 
                                type=int, 
                                metavar='N',
                                help='number of data loading workers')
        self.parser.add_argument('-b', '--batch-size', 
                                default=4, 
                                type=int,
                                metavar='N', 
                                help='mini-batch size')

        # CONFIG OPTIMIZER  
        self.parser.add_argument('--lr', '--learning-rate', 
                                default=2e-4, 
                                type=float,
                                metavar='LR', 
                                help='initial learning rate')
        self.parser.add_argument('--momentum', 
                                default=0.9, 
                                type=float, 
                                metavar='M',
                                help='momentum for sgd, alpha parameter for adam')
        self.parser.add_argument('--beta', 
                                default=0.999, 
                                type=float, 
                                metavar='M',
                                help='beta parameters for adam')
        self.parser.add_argument('--weight-decay', '--wd', 
                                default=0, 
                                type=float,
                                metavar='W', 
                                help='weight decay')

        # CONFIG LOSS FUNCTION
        self.parser.add_argument('-p', '--photo-loss-weight', 
                                type=float, 
                                help='weight for photometric loss', 
                                metavar='W', 
                                default=1)
        self.parser.add_argument('-m', '--mask-loss-weight', 
                                type=float, 
                                help='weight for explainabilty mask loss', 
                                metavar='W', 
                                default=0.2)
        self.parser.add_argument('-s', '--smooth-loss-weight', 
                                type=float, 
                                help='weight for disparity smoothness loss', 
                                metavar='W', 
                                default=0.1)
                              
        # CONFIG DISPNET
        self.parser.add_argument('--dispnet',
                                type=str,
                                default='DispNetS',
                                help='disparity network architecture')
        self.parser.add_argument('--pretrained-disp', 
                                dest='pretrained_disp', 
                                default=None, 
                                metavar='PATH',
                                help='path to pre-trained dispnet model')
        self.parser.add_argument('--with-gt',
                                type=self.str2bool, 
                                default=True, 
                                help='use depth ground truth for validation, You need to store it in npy 2D arrays see data/kitti_raw_loader.py for an example')

        # CONFIG POSE EXP NET
        self.parser.add_argument('--poseexpnet',
                                type=str,
                                default='PoseExpNet',
                                help='disparity network architecture')
        self.parser.add_argument('--pretrained-exppose', 
                                dest='pretrained_exp_pose', 
                                default=None, 
                                metavar='PATH',
                                help='path to pre-trained Exp Pose net model')
        self.parser.add_argument('--with-pose', 
                                type=self.str2bool,
                                default=False, 
                                help='use pose ground truth for validation, You need to store it in text files of 12 columns see data/kitti_raw_loader.py for an example, Note that for kitti, it is recommend to use odometry train set to test pose')

        # CONFIG OTHER
        self.parser.add_argument('--train',
                                action='store_true',
                                help='train model')
        self.parser.add_argument('--test',
                                action='store_true',
                                help='test model on benchmarks')
        self.parser.add_argument('--infer',
                                action='store_true',
                                help='model inference')
        self.parser.add_argument('--epochs', 
                                default=2, 
                                type=int, 
                                metavar='N',
                                help='number of total epochs to run')
        self.parser.add_argument('--epoch-size', 
                                default=200, 
                                type=int, 
                                metavar='N',
                                help='manual epoch size (will match dataset size if not set)')
        self.parser.add_argument('--print-freq', 
                                default=10, 
                                type=int,
                                metavar='N', 
                                help='print frequency')
        self.parser.add_argument('-e', '--evaluate', 
                                dest='evaluate', 
                                action='store_true',
                                help='evaluate model on validation set')
        self.parser.add_argument('--seed', 
                                default=0, 
                                type=int, 
                                help='seed for random functions, and network initialization')
        self.parser.add_argument('--log-summary', 
                                default='progress_log_summary.csv', 
                                metavar='PATH',
                                help='csv where to save per-epoch train and valid stats')
        self.parser.add_argument('--log-full', 
                                default='progress_log_full.csv', 
                                metavar='PATH',
                                help='csv where to save per-gradient descent train stats')
        self.parser.add_argument('--log-output', 
                                action='store_true', 
                                help='will log dispnet outputs and warped imgs at validation step')
        self.parser.add_argument('-f', '--training-output-freq', 
                                type=int,
                                help='frequence for outputting dispnet outputs and warped imgs at training for all scales if 0, will not output',
                                metavar='N', 
                                default=1000)
        self.parser.add_argument('--name', 
                                dest='name', 
                                type=str, 
                                default='demo',
                                help='name of the experiment, checpoints are stored in checpoints/name')

    def str2bool(self, v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    def parse(self):
        self.config = self.parser.parse_args()
        return self.config
