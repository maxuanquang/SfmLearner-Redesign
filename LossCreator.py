import argparse
import time
import csv
from path import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import custom_transforms
import models
from utils import tensor2array, save_checkpoint, save_path_formatter, log_output_tensorboard

from loss_functions import photometric_reconstruction_loss, explainability_loss, smooth_loss
from loss_functions import compute_depth_errors, compute_pose_errors
from inverse_warp import pose_vec2mat

class LossCreator():
    def __init__(self, args):
        self.args = args
    def create(self):
        

        w1, w2, w3 = self.args.photo_loss_weight, self.args.mask_loss_weight, self.args.smooth_loss_weight
        def loss_function(tgt_img, ref_imgs, intrinsics, 
            depth, explainability_mask, pose, 
            rotation_mode, padding_mode):

            loss_1, warped, diff = photometric_reconstruction_loss(tgt_img, ref_imgs, intrinsics, 
                                                            depth, explainability_mask, pose,
                                                            rotation_mode, padding_mode)

            if w2 > 0:
                loss_2 = explainability_loss(explainability_mask)
            else:
                loss_2 = 0
                
            loss_3 = smooth_loss(depth)

            loss = w1*loss_1 + w2*loss_2 + w3*loss_3

            return loss, loss_1, warped, diff, loss_2, loss_3
        
        return loss_function