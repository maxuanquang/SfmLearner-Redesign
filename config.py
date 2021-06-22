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

#region TRAINING
        #region CONFIG DATALOADER
        self.parser.add_argument('--dataset-train',
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
        #endregion

        #region CONFIG OPTIMIZER  
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
        #endregion

        #region CONFIG LOSS FUNCTION

        #region Config loss components
        # self.parser.add_argument('--photometric-reconstruction-loss',
        #                         type=self.str2bool, 
        #                         default=True, 
        #                         help='use photometric reconstruction loss')
        # self.parser.add_argument('--mask-loss',
        #                         type=self.str2bool, 
        #                         default=True, 
        #                         help='use mask loss')
        # self.parser.add_argument('--smooth-loss',
        #                         type=self.str2bool, 
        #                         default=True, 
        #                         help='use smoothness loss')
        # self.parser.add_argument('--photometric-flow-loss',
        #                         type=self.str2bool, 
        #                         default=False, 
        #                         help='use photometric flow loss')
        # self.parser.add_argument('--concensus-depth-flow-loss',
        #                         type=self.str2bool, 
        #                         default=False, 
        #                         help='use consensus depth flow loss')

        self.parser.add_argument('-p', '--photo-loss-weight', 
                                type=float,
                                default=1,
                                help='weight for photometric reconstruction loss')
        self.parser.add_argument('-m', '--mask-loss-weight', 
                                type=float, 
                                default=0.2,
                                help='weight for explainabilty mask loss')
        self.parser.add_argument('-s', '--smooth-loss-weight', 
                                type=float, 
                                default=0.1,
                                help='weight for smoothness loss')
        self.parser.add_argument('--photometric-flow-loss-weight', 
                                type=float, 
                                default=0.0,
                                help='weight for photometric flow loss')
        self.parser.add_argument('--consensus-depth-flow-loss-weight', 
                                type=float, 
                                default=0.0,
                                help='weight for consensus depth flow loss')
        #endregion

        #region Config photometric reconstruction and photometric flow loss
        self.parser.add_argument('--L1-photometric',
                                type=self.str2bool, 
                                default=True, 
                                help='use L1 for photometric reconstruction and photometric flow loss')
        self.parser.add_argument('--robust-L1-photometric',
                                type=self.str2bool, 
                                default=False, 
                                help='use robust L1 photometric reconstruction and photometric flow loss')
        self.parser.add_argument('--L2-photometric',
                                type=self.str2bool, 
                                default=False, 
                                help='use L2 photometric reconstruction and photometric flow loss')
        self.parser.add_argument('--ssim-photometric',
                                type=self.str2bool, 
                                default=False, 
                                help='use ssim photometric reconstruction and photometric flow loss')

        self.parser.add_argument('--L1-photometric-weight', 
                                type=float, 
                                default=1,
                                help='L1 weight in photometric reconstruction and photometric flow loss')
        self.parser.add_argument('--robust-L1-photometric-weight', 
                                type=float, 
                                default=0,
                                help='robust-L1 weight in photometric reconstruction and photometric flow loss')
        self.parser.add_argument('--L2-photometric-weight', 
                                type=float, 
                                default=0,
                                help='L2 weight in photometric reconstruction and photometric flow loss')
        self.parser.add_argument('--ssim-photometric-weight', 
                                type=float, 
                                default=0.8,
                                help='SSIM weight in photometric reconstruction and photometric flow loss')

        self.parser.add_argument('--min-photometric',
                                type=self.str2bool, 
                                default=False, 
                                help='calculate min for photometric reconstruction and photometric flow loss')
        self.parser.add_argument('--mean-photometric',
                                type=self.str2bool, 
                                default=True, 
                                help='calculate mean for photometric reconstruction and photometric flow loss')

        self.parser.add_argument('--use-mask-for-photometric',
                                type=self.str2bool, 
                                default=False, 
                                help='mutiply explainability mask into photometric reconstruction and photometric flow loss')
        #endregion

        #region Config smooth loss
        self.parser.add_argument('--use-disp-smooth',
                                type=self.str2bool, 
                                default=False, 
                                help='use disp for calculate smoothness loss')
        self.parser.add_argument('--use-depth-smooth',
                                type=self.str2bool, 
                                default=True, 
                                help='use depth for calculate smoothness loss')
        self.parser.add_argument('--use-flow-smooth',
                                type=self.str2bool, 
                                default=False, 
                                help='use flow for calculate smoothness loss')
        self.parser.add_argument('--use-mask-smooth',
                                type=self.str2bool, 
                                default=False, 
                                help='use mask for calculate smoothness loss')

        self.parser.add_argument('--disp-smooth-weight', 
                                type=float, 
                                default=1,
                                help='disp-smooth weight in smoothness loss')
        self.parser.add_argument('--depth-smooth-weight', 
                                type=float, 
                                default=1,
                                help='depth-smooth weight in smoothness loss')
        self.parser.add_argument('--flow-smooth-weight', 
                                type=float, 
                                default=1,
                                help='flow-smooth weight in smoothness loss')
        self.parser.add_argument('--mask-smooth-weight', 
                                type=float, 
                                default=1,
                                help='mask-smooth weight in smoothness loss')

        self.parser.add_argument('--use-first-derivative',
                                type=self.str2bool, 
                                default=False, 
                                help='use first-order gradients for calculate smoothness loss')
        self.parser.add_argument('--use-second-derivative',
                                type=self.str2bool, 
                                default=True, 
                                help='use second-order gradients for calculate smoothness loss')
        self.parser.add_argument('--use-L1-smooth',
                                type=self.str2bool, 
                                default=True, 
                                help='use L1 for calculate smoothness loss')
        self.parser.add_argument('--use-L2-smooth',
                                type=self.str2bool, 
                                default=False, 
                                help='use L2 for calculate smoothness loss')
        #endregion

        #region Config consensus depth flow loss
        self.parser.add_argument('--lambda-c', 
                                type=float, 
                                default=0.001,
                                help='lambda_C constant')
        #endregion

        #endregion
                              
        #region CONFIG DISPNET
        self.parser.add_argument('--dispnet-architecture',
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
        self.parser.add_argument('--dispnet-decisive-error',
                                type=str,
                                choices=['abs_diff', 'abs_rel', 'sq_rel', 'a1', 'a2', 'a3'],
                                default='abs_rel',
                                help='decisive error for choosing best dispnet model')
        #endregion

        #region CONFIG POSE - POSEEXP NET
        # If poseexpnet_architecture is set, then posenet_architecture have to be None and vice versa
        self.parser.add_argument('--poseexpnet-architecture',
                                default='PoseExpNet',
                                help='disparity network architecture')
        self.parser.add_argument('--pretrained-poseexp', 
                                dest='pretrained_pose_exp', 
                                default=None, 
                                metavar='PATH',
                                help='path to pre-trained Exp Pose net model')
        self.parser.add_argument('--posenet-architecture',
                                default=None,
                                help='pose estimation network architecture')
        self.parser.add_argument('--pretrained-pose', 
                                dest='pretrained_pose', 
                                default=None, 
                                metavar='PATH',
                                help='path to pre-trained Pose net model')
        self.parser.add_argument('--with-pose', 
                                type=self.str2bool,
                                default=False, 
                                help='use pose ground truth for validation, You need to store it in text files of 12 columns see data/kitti_raw_loader.py for an example, Note that for kitti, it is recommend to use odometry train set to test pose')
        #endregion

        #region CONFIG FLOWNET
        self.parser.add_argument('--flownet-architecture',
                                default=None,
                                help='disparity network architecture')
        self.parser.add_argument('--pretrained-flow', 
                                dest='pretrained_flow', 
                                default=None, 
                                metavar='PATH',
                                help='path to pre-trained flownet model')
        self.parser.add_argument('--with-flow-gt',
                                type=self.str2bool, 
                                default=False, 
                                help='use depth ground truth for validation, You need to store it in npy 2D arrays see data/kitti_raw_loader.py for an example')
        self.parser.add_argument('--flownet-decisive-error',
                                type=str,
                                choices=['epe_total', 'epe_rigid', 'epe_non_rigid', 'outliers', 'epe_total_with_gt_mask', 'epe_rigid_with_gt_mask', 'epe_non_rigid_with_gt_mask', 'outliers_gt_mask'],
                                default='epe_non_rigid_with_gt_mask',
                                help='decisive error for choosing best flownet model')
        #endregion

        #region CONFIG MASKNET
        self.parser.add_argument('--masknet-architecture',
                                default=None,
                                help='disparity network architecture')
        self.parser.add_argument('--pretrained-mask', 
                                dest='pretrained_disp',
                                default=None, 
                                metavar='PATH',
                                help='path to pre-trained dispnet model')
        self.parser.add_argument('--with-mask-gt',
                                type=self.str2bool, 
                                default=False, 
                                help='use depth ground truth for validation, You need to store it in npy 2D arrays see data/kitti_raw_loader.py for an example')
        self.parser.add_argument('--masknet-decisive-error',
                                type=str,
                                choices=['epe_total', 'epe_rigid', 'epe_non_rigid', 'outliers', 'epe_total_with_gt_mask', 'epe_rigid_with_gt_mask', 'epe_non_rigid_with_gt_mask', 'outliers_gt_mask'],
                                default='outliers',
                                help='decisive error for choosing best masknet model')
        #endregion

        #region CONFIG OTHER
        self.parser.add_argument('--train',
                                action='store_true',
                                help='train model')
        self.parser.add_argument('--evaluate',
                                action='store_true',
                                help='evaluate model performance on benchmarks')
        self.parser.add_argument('--infer',
                                action='store_true',
                                help='model inference')

        self.parser.add_argument('--eval-dispnet',
                                action='store_true',
                                help='evaluate dispnet performance on benchmarks')
        self.parser.add_argument('--eval-poseexpnet',
                                action='store_true',
                                help='evaluate pose_exp_net performance on benchmarks')
        self.parser.add_argument('--eval-posenet',
                                action='store_true',
                                help='evaluate posenet performance on benchmarks')

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
        # self.parser.add_argument('-e', '--evaluate', 
        #                         dest='evaluate', 
        #                         action='store_true',
        #                         help='evaluate model on validation set')
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
        self.parser.add_argument('--nlevels', 
                                default=4, 
                                type=int, 
                                help='seed for random functions, and network initialization')
        self.parser.add_argument('--checkpoint-folder', 
                                type=str, 
                                default='/content/drive/MyDrive/VinAI/Motion segmentation/checkpoints_sfmlearner',
                                help='path to checkpoint folder')
        self.parser.add_argument('--name', 
                                dest='name', 
                                type=str, 
                                default='demo',
                                help='name of the experiment, checkpoints are stored in checkpoints/name')
        self.parser.add_argument('--resume',
                                action='store_true',
                                help='resume from checkpoint')
        #endregion
#endregion

#region EVALUATING
        # self.parser.add_argument("--pretrained-dispnet", required=True, type=str, help="pretrained DispNet path")
        # self.parser.add_argument("--pretrained-posenet", default=None, type=str, help="pretrained PoseNet path (for scale factor)")
        self.parser.add_argument("--img-height", default=128, type=int, help="Image height")
        self.parser.add_argument("--img-width", default=416, type=int, help="Image width")
        self.parser.add_argument("--no-resize", action='store_true', help="no resizing is done")
        self.parser.add_argument("--min-depth", default=1e-3)
        self.parser.add_argument("--max-depth", default=80)

        self.parser.add_argument("--dataset-dir", default='/content/eigen_test_split', type=str, help="Dataset directory")
        self.parser.add_argument("--dataset-list", default=None, type=str, help="Dataset list file")
        self.parser.add_argument("--output-dir", default=None, type=str, help="Output directory for saving predictions in a big 3D numpy file")

        self.parser.add_argument("--gt-type", default='KITTI', type=str, help="GroundTruth data type", choices=['npy', 'png', 'KITTI', 'stillbox'])
        self.parser.add_argument("--gps", '-g', action='store_true',
                            help='if selected, will get displacement from GPS for KITTI. Otherwise, will integrate speed')
        self.parser.add_argument("--img-exts", default=['png', 'jpg', 'bmp'], nargs='*', type=str, help="images extensions to glob")
#endregion

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
