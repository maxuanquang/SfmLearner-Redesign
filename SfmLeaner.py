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
from logger import TermLogger, AverageMeter

from ModelCreator import ModelCreator
from DataLoaderCreator import DataLoaderCreator
from OptimizerCreator import OptimizerCreator
from LossCreator import LossCreator
from Reporter import Reporter

from tensorboardX import SummaryWriter
import glob

best_error = -1
n_iter = 0
device = torch.device("cuda")

class SfmLearner():
    def __init__(self, args):
        self.args = args

    def train(self):
        # create main objects
        global best_error, n_iter, device
        torch.manual_seed(self.args.seed)

        model_creator = ModelCreator(self.args)
        optimizer_creator = OptimizerCreator(self.args)
        dataloader_creator = DataLoaderCreator(self.args)
        loss_creator = LossCreator(self.args)

        self.disp_net = model_creator.create(model='dispnet')
        if self.args.poseexpnet_architecture:
            self.pose_exp_net = model_creator.create(model='poseexpnet')
            self.optimizer = optimizer_creator.create(self.disp_net, self.pose_exp_net)
        else:
            self.args.mask_loss_weight = 0 # because posenet does not output explainability mask
            self.pose_net = model_creator.create(model='posenet')
            self.optimizer = optimizer_creator.create(self.disp_net, self.pose_net)

        self.train_loader, self.val_loader = dataloader_creator.create()
        self.loss_function = loss_creator.create()

        # objects serve for training
        self.reporter = Reporter(self.args)
        self.tb_writer = SummaryWriter(self.args.save_path)

        if self.args.epoch_size == 0:
            self.args.epoch_size = len(self.train_loader)
        train_size=min(len(self.train_loader), self.args.epoch_size)

        self.logger = TermLogger(n_epochs=self.args.epochs, train_size=train_size, valid_size=len(self.val_loader))
        self.logger.epoch_bar.start()

        for epoch in range(self.args.epochs):
            self.logger.epoch_bar.update(epoch)

            # train for one epoch
            self.logger.reset_train_bar()
            train_loss = self.train_one_epoch()
            self.logger.train_writer.write(' * Avg Loss : {:.3f}'.format(train_loss))

            # evaluate on validation set
            self.logger.reset_valid_bar()
            if self.args.with_gt and self.args.with_pose:
                errors, error_names = self.validate_with_gt_pose(epoch=epoch)
            elif self.args.with_gt:
                errors, error_names = self.validate_with_gt(epoch=epoch)

            error_string = ', '.join('{} : {:.3f}'.format(name, error) for name, error in zip(error_names, errors))
            self.logger.valid_writer.write(' * Avg {}'.format(error_string))

            for error, name in zip(errors, error_names):
                self.tb_writer.add_scalar(name, error, epoch)

            # Decisive error to measure your model's performance specified in config
            decisive_error = 0
            for i, name in enumerate(error_names):
                if self.args.dispnet_decisive_error == name:
                    decisive_error = errors[i]
                break
            if best_error < 0:
                best_error = decisive_error

            # remember lowest error and save checkpoint
            is_best = decisive_error < best_error
            best_error = min(best_error, decisive_error)
            save_checkpoint(
                self.args.save_path, {
                    'epoch': epoch + 1,
                    'state_dict': self.disp_net.module.state_dict()
                }, {
                    'epoch': epoch + 1,
                    'state_dict': self.pose_exp_net.module.state_dict()
                },
                is_best)
            self.reporter.update_log_summary(train_loss, decisive_error)

        self.logger.epoch_bar.finish()
        self.reporter.create_report()

        return 0

    def test(self):
        return 0

    def infer(self):
        return 0

    def prepare_dataset(self):
        return 0


    @torch.no_grad()
    def validate_with_gt_pose(self, epoch, sample_nb_to_log=3):
        global device
        batch_time = AverageMeter()
        depth_error_names = ['abs_diff', 'abs_rel', 'sq_rel', 'a1', 'a2', 'a3']
        depth_errors = AverageMeter(i=len(depth_error_names), precision=4)
        pose_error_names = ['ATE', 'RTE']
        pose_errors = AverageMeter(i=2, precision=4)
        log_outputs = sample_nb_to_log > 0
        # Output the logs throughout the whole dataset
        batches_to_log = list(np.linspace(0, len(self.val_loader), sample_nb_to_log).astype(int))
        poses_values = np.zeros(((len(self.val_loader)-1) * self.args.batch_size * (self.args.sequence_length-1), 6))
        disp_values = np.zeros(((len(self.val_loader)-1) * self.args.batch_size * 3))

        # switch to evaluate mode
        self.disp_net.eval()
        self.pose_exp_net.eval()

        end = time.time()
        self.logger.valid_bar.update(0)
        for i, (tgt_img, ref_imgs, gt_depth, gt_poses) in enumerate(self.val_loader):
            tgt_img = tgt_img.to(device)
            gt_depth = gt_depth.to(device)
            gt_poses = gt_poses.to(device)
            ref_imgs = [img.to(device) for img in ref_imgs]
            b = tgt_img.shape[0]

            # compute output
            output_disp = self.disp_net(tgt_img)
            output_depth = 1/output_disp
            explainability_mask, output_poses = self.pose_exp_net(tgt_img, ref_imgs)

            reordered_output_poses = torch.cat([output_poses[:, :gt_poses.shape[1]//2],
                                                torch.zeros(b, 1, 6).to(output_poses),
                                                output_poses[:, gt_poses.shape[1]//2:]], dim=1)

            # pose_vec2mat only takes B, 6 tensors, so we simulate a batch dimension of B * seq_length
            unravelled_poses = reordered_output_poses.reshape(-1, 6)
            unravelled_matrices = pose_vec2mat(unravelled_poses, rotation_mode=self.args.rotation_mode)
            inv_transform_matrices = unravelled_matrices.reshape(b, -1, 3, 4)

            rot_matrices = inv_transform_matrices[..., :3].transpose(-2, -1)
            tr_vectors = -rot_matrices @ inv_transform_matrices[..., -1:]

            transform_matrices = torch.cat([rot_matrices, tr_vectors], axis=-1)

            first_inv_transform = inv_transform_matrices.reshape(b, -1, 3, 4)[:, :1]
            final_poses = first_inv_transform[..., :3] @ transform_matrices
            final_poses[..., -1:] += first_inv_transform[..., -1:]
            final_poses = final_poses.reshape(b, -1, 3, 4)

            if log_outputs and i in batches_to_log:  # log first output of wanted batches
                index = batches_to_log.index(i)
                if epoch == 0:
                    for j, ref in enumerate(ref_imgs):
                        self.tb_writer.add_image('val Input {}/{}'.format(j, index), tensor2array(tgt_img[0]), 0)
                        self.tb_writer.add_image('val Input {}/{}'.format(j, index), tensor2array(ref[0]), 1)

                log_output_tensorboard(self.tb_writer, 'val', index, '', epoch, output_depth, output_disp, None, None, explainability_mask)

            if log_outputs and i < len(self.val_loader)-1:
                step = self.args.batch_size*(self.args.sequence_length-1)
                poses_values[i * step:(i+1) * step] = output_poses.cpu().view(-1, 6).numpy()
                step = self.args.batch_size * 3
                disp_unraveled = output_disp.cpu().view(self.args.batch_size, -1)
                disp_values[i * step:(i+1) * step] = torch.cat([disp_unraveled.min(-1)[0],
                                                                disp_unraveled.median(-1)[0],
                                                                disp_unraveled.max(-1)[0]]).numpy()

            depth_errors.update(compute_depth_errors(gt_depth, output_depth[:, 0]))
            pose_errors.update(compute_pose_errors(gt_poses, final_poses))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            self.logger.valid_bar.update(i+1)
            if i % self.args.print_freq == 0:
                self.logger.valid_writer.write(
                    'valid: Time {} Abs Error {:.4f} ({:.4f}), ATE {:.4f} ({:.4f})'.format(batch_time,
                                                                                        depth_errors.val[0],
                                                                                        depth_errors.avg[0],
                                                                                        pose_errors.val[0],
                                                                                        pose_errors.avg[0]))
        if log_outputs:
            prefix = 'valid poses'
            coeffs_names = ['tx', 'ty', 'tz']
            if self.args.rotation_mode == 'euler':
                coeffs_names.extend(['rx', 'ry', 'rz'])
            elif self.args.rotation_mode == 'quat':
                coeffs_names.extend(['qx', 'qy', 'qz'])
            for i in range(poses_values.shape[1]):
                self.tb_writer.add_histogram('{} {}'.format(prefix, coeffs_names[i]), poses_values[:, i], epoch)
            self.tb_writer.add_histogram('disp_values', disp_values, epoch)
        self.logger.valid_bar.update(len(self.val_loader))
        return depth_errors.avg + pose_errors.avg, depth_error_names + pose_error_names


    @torch.no_grad()
    def validate_with_gt(self, epoch, sample_nb_to_log=3):
        global device
        batch_time = AverageMeter()
        error_names = ['abs_diff', 'abs_rel', 'sq_rel', 'a1', 'a2', 'a3']
        errors = AverageMeter(i=len(error_names))
        log_outputs = sample_nb_to_log > 0
        # Output the logs throughout the whole dataset
        batches_to_log = list(np.linspace(0, len(self.val_loader)-1, sample_nb_to_log).astype(int))

        # switch to evaluate mode
        self.disp_net.eval()

        end = time.time()
        self.logger.valid_bar.update(0)
        for i, (tgt_img, depth) in enumerate(self.val_loader):
            tgt_img = tgt_img.to(device)
            depth = depth.to(device)

            # compute output
            output_disp = self.disp_net(tgt_img)
            output_depth = 1/output_disp[:, 0]

            if log_outputs and i in batches_to_log:
                index = batches_to_log.index(i)
                if epoch == 0:
                    self.tb_writer.add_image('val Input/{}'.format(index), tensor2array(tgt_img[0]), 0)
                    depth_to_show = depth[0]
                    self.tb_writer.add_image('val target Depth Normalized/{}'.format(index),
                                        tensor2array(depth_to_show, max_value=None),
                                        epoch)
                    depth_to_show[depth_to_show == 0] = 1000
                    disp_to_show = (1/depth_to_show).clamp(0, 10)
                    self.tb_writer.add_image('val target Disparity Normalized/{}'.format(index),
                                        tensor2array(disp_to_show, max_value=None, colormap='magma'),
                                        epoch)

                self.tb_writer.add_image('val Dispnet Output Normalized/{}'.format(index),
                                    tensor2array(output_disp[0], max_value=None, colormap='magma'),
                                    epoch)
                self.tb_writer.add_image('val Depth Output Normalized/{}'.format(index),
                                    tensor2array(output_depth[0], max_value=None),
                                    epoch)
            errors.update(compute_depth_errors(depth, output_depth))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            self.logger.valid_bar.update(i+1)
            if i % self.args.print_freq == 0:
                self.logger.valid_writer.write('valid: Time {} Abs Error {:.4f} ({:.4f})'.format(batch_time, errors.val[0], errors.avg[0]))
        self.logger.valid_bar.update(len(self.val_loader))
        return errors.avg, error_names


    def train_one_epoch(self):
        global n_iter, device
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter(precision=4)
        w1, w2, w3 = self.args.photo_loss_weight, self.args.mask_loss_weight, self.args.smooth_loss_weight

        # switch to train mode
        self.disp_net.train()
        self.pose_exp_net.train()

        end = time.time()
        self.logger.train_bar.update(0)

        for i, (tgt_img, ref_imgs, intrinsics, intrinsics_inv) in enumerate(self.train_loader):
            log_losses = i > 0 and n_iter % self.args.print_freq == 0
            log_output = self.args.training_output_freq > 0 and n_iter % self.args.training_output_freq == 0

            # measure data loading time
            data_time.update(time.time() - end)
            tgt_img = tgt_img.to(device)
            ref_imgs = [img.to(device) for img in ref_imgs]
            intrinsics = intrinsics.to(device)

            # compute output
            disparities = self.disp_net(tgt_img)
            depth = [1/disp for disp in disparities]
            if self.args.poseexpnet_architecture:
                explainability_mask, pose = self.pose_exp_net(tgt_img, ref_imgs)
            else:
                pose = self.pose_net(tgt_img, ref_imgs)
                explainability_mask = []
                for _ in range(self.args.nlevels):
                    explainability_mask.append(None)

            loss, loss_1, warped, diff, loss_2, loss_3 = self.loss_function(tgt_img, ref_imgs, intrinsics,
                                                                            depth, explainability_mask, pose,
                                                                            self.args.rotation_mode, self.args.padding_mode, self.args)

            if log_losses:
                self.tb_writer.add_scalar('photometric_error', loss_1.item(), n_iter)
                if w2 > 0:
                    self.tb_writer.add_scalar('explanability_loss', loss_2.item(), n_iter)
                self.tb_writer.add_scalar('disparity_smoothness_loss', loss_3.item(), n_iter)
                self.tb_writer.add_scalar('total_loss', loss.item(), n_iter)

            if log_output:
                self.tb_writer.add_image('train Input', tensor2array(tgt_img[0]), n_iter)
                for k, scaled_maps in enumerate(zip(depth, disparities, warped, diff, explainability_mask)):
                    log_output_tensorboard(self.tb_writer, "train", 0, " {}".format(k), n_iter, *scaled_maps)

            # record loss and EPE
            losses.update(loss.item(), self.args.batch_size)

            # compute gradient and do Adam step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # with open(args.save_path/args.log_full, 'a') as csvfile:
            #     writer = csv.writer(csvfile, delimiter='\t')
            #     writer.writerow([loss.item(), loss_1.item(), loss_2.item() if w2 > 0 else 0, loss_3.item()])
            self.logger.train_bar.update(i+1)
            if i % self.args.print_freq == 0:
                self.logger.train_writer.write('Train: Time {} Data {} Loss {}'.format(batch_time, data_time, losses))
            if i >= self.args.epoch_size - 1:
                break

            n_iter += 1

        return losses.avg[0]
