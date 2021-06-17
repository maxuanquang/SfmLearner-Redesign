import argparse
from operator import mod
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

from loss_functions import photometric_reconstruction_loss, explainability_loss, photometric_reconstruction_results, smooth_loss, photometric_flow_loss, consensus_depth_flow_mask
from loss_functions import compute_depth_errors, compute_pose_errors
from inverse_warp import pose_vec2mat

class LossCreator():
    def __init__(self, args):
        self.args = args
    def create(self, function='loss_function'):

        def loss_function(tgt_img, ref_imgs, intrinsics, 
            depth, explainability_mask, pose, 
            rotation_mode, padding_mode, args):

            w1 = self.args.photo_loss_weight
            w2 = self.args.mask_loss_weight
            w3 = self.args.smooth_loss_weight
            w4 = self.args.photometric_flow_loss_weight
            w5 = self.args.consensus_depth_flow_loss_weight

            if w1 > 0:
                loss_1 = photometric_reconstruction_loss(tgt_img, ref_imgs, intrinsics, 
                                                        depth, explainability_mask, pose,
                                                        args,
                                                        rotation_mode, padding_mode)
            else:
                loss_1 = 0

            if w2 > 0:
                loss_2 = explainability_loss(explainability_mask)
            else:
                loss_2 = 0
                
            if w3 > 0:
                loss_3 = smooth_loss(depth)
            else:
                loss_3 = 0

            if w4 > 0:
                loss_4 = 0
                # loss_4 = photometric_flow_loss()
            else:
                loss_4 = 0

            if w5 > 0:
                loss_5 = 0
                # loss_5 = consensus_depth_flow_mask()
            else:
                loss_5 = 0
            
            loss = w1*loss_1 + w2*loss_2 + w3*loss_3 + w4*loss_4 + w5*loss_5

            losses_dict = {
                'total_loss': loss,
                'photometric_reconstruction_loss': loss_1,
                'explainability_loss': loss_2,
                'smooth_loss': loss_3,
                'photometric_flow_loss': loss_4,
                'consensus_depth_flow_loss': loss_5
            }

            return losses_dict
        

        def calculate_intermediate_results(tgt_img, ref_imgs, intrinsics, 
                                            depth, explainability_mask, pose,
                                            rotation_mode, padding_mode, args):

            w1 = self.args.photo_loss_weight
            w2 = self.args.mask_loss_weight
            w3 = self.args.smooth_loss_weight
            w4 = self.args.photometric_flow_loss_weight
            w5 = self.args.consensus_depth_flow_loss_weight

            if w1 > 0:
                warped_results, diff_results = photometric_reconstruction_results(tgt_img, ref_imgs, intrinsics, 
                                                        depth, explainability_mask, pose,
                                                        args,
                                                        rotation_mode, padding_mode)
                photometric_reconstruction_results_dict = {}
                photometric_reconstruction_results_dict['photometric_reconstruction_warped'] = warped_results
                photometric_reconstruction_results_dict['photometric_reconstruction_diff'] = diff_results
                
            else:
                photometric_reconstruction_results_dict = {}

            if w4 > 0:
                photometric_flow_results_dict = {}
            else:
                photometric_flow_results_dict = {}

            if w5 > 0:
                consensus_results_dict = {}
            else:
                consensus_results_dict = {}
            
            results_dict = {}
            results_dict.update(photometric_reconstruction_results_dict)
            results_dict.update(photometric_flow_results_dict)
            results_dict.update(consensus_results_dict)

            return results_dict

        if function == 'loss_function':
            return loss_function
        if function == 'calculate_intermediate_results':
            return calculate_intermediate_results