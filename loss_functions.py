from __future__ import division
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from inverse_warp import inverse_warp
from ssim import ssim
epsilon = 1e-8

def l2(x):
    x = torch.pow(x, 2)
    x = x.mean()
    return x

def l2_per_pix(x):
    x = torch.pow(x, 2)
    return x

def l1(x):
    x = torch.abs(x)
    x = x.mean()
    return x

def l1_per_pix(x):
    x = torch.abs(x)
    return x

def robust_l1(x, q=0.5, eps=1e-2):
    x = torch.pow((x.pow(2) + eps), q)
    x = x.mean()
    return x

def robust_l1_per_pix(x, q=0.5, eps=1e-2):
    x = torch.pow((x.pow(2) + eps), q)
    return x

def photometric_reconstruction_loss(tgt_img, ref_imgs, intrinsics,
                                    depth, explainability_mask, pose,
                                    args,
                                    rotation_mode='euler', padding_mode='zeros'):
    def one_scale(depth, explainability_mask):
        assert(explainability_mask is None or depth.size()[2:] == explainability_mask.size()[2:])
        assert(pose.size(1) == len(ref_imgs))

        reconstruction_loss = 0
        b, _, h, w = depth.size()
        downscale = tgt_img.size(2)/h

        tgt_img_scaled = F.interpolate(tgt_img, (h, w), mode='area')
        ref_imgs_scaled = [F.interpolate(ref_img, (h, w), mode='area') for ref_img in ref_imgs]
        intrinsics_scaled = torch.cat((intrinsics[:, 0:2]/downscale, intrinsics[:, 2:]), dim=1)

        warped_imgs = []
        diff_maps = []
        loss_list = []

        for i, ref_img in enumerate(ref_imgs_scaled):
            current_pose = pose[:, i]

            ref_img_warped, valid_points = inverse_warp(ref_img, depth[:,0], current_pose,
                                                        intrinsics_scaled,
                                                        rotation_mode, padding_mode)
            diff = (tgt_img_scaled - ref_img_warped) * valid_points.unsqueeze(1).float()

            if args.L1_photometric:
                diff = l1_per_pix(diff)*args.L1_photometric_weight
            elif args.robust_L1_photometric:
                diff = robust_l1_per_pix(diff)*args.robust_L1_photometric_weight
            elif args.L2_photometric:
                diff = l2_per_pix(diff)*args.L2_photometric_weight

            if args.ssim_photometric:
                ssim_loss = 1 - ssim(tgt_img_scaled, ref_img_warped)
                ssim_loss = ssim_loss * args.ssim_photometric_weight
            else:
                ssim_loss = 0

            current_loss = diff + ssim_loss

            if explainability_mask is not None:
                current_loss = current_loss * explainability_mask[:,i:i+1].expand_as(current_loss)

            if args.mean_photometric:
                reconstruction_loss += current_loss.mean()
                assert((reconstruction_loss == reconstruction_loss).item() == 1)
            elif args.min_photometric:
                current_loss = current_loss.mean(1)
                loss_list.append(current_loss)

            warped_imgs.append(ref_img_warped[0])
            diff_maps.append(diff[0])
        
        if args.min_photometric:
            loss_list = torch.stack(loss_list)
            loss_list = loss_list.min(0)[0]
            reconstruction_loss = loss_list.mean()

        return reconstruction_loss, warped_imgs, diff_maps

    warped_results, diff_results = [], []
    if type(explainability_mask) not in [tuple, list]:
        explainability_mask = [explainability_mask]
    if type(depth) not in [list, tuple]:
        depth = [depth]

    total_loss = 0
    for d, mask in zip(depth, explainability_mask):
        loss, warped, diff = one_scale(d, mask)
        total_loss += loss
        warped_results.append(warped)
        diff_results.append(diff)
    return total_loss, warped_results, diff_results


def explainability_loss(mask):
    if type(mask) not in [tuple, list]:
        mask = [mask]
    loss = 0
    for mask_scaled in mask:
        ones_var = torch.ones_like(mask_scaled)
        loss += nn.functional.binary_cross_entropy(mask_scaled, ones_var)
    return loss


def smooth_loss(pred_map):
    def gradient(pred):
        D_dy = pred[:, :, 1:] - pred[:, :, :-1]
        D_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        return D_dx, D_dy

    if type(pred_map) not in [tuple, list]:
        pred_map = [pred_map]

    loss = 0
    weight = 1.

    for scaled_map in pred_map:
        dx, dy = gradient(scaled_map)
        dx2, dxdy = gradient(dx)
        dydx, dy2 = gradient(dy)
        loss += (dx2.abs().mean() + dxdy.abs().mean() + dydx.abs().mean() + dy2.abs().mean())*weight
        weight /= 2.3  # don't ask me why it works better
    return loss


def photometric_flow_loss(tgt_img, ref_imgs, flows, explainability_mask, lambda_oob=0, qch=0.5, wssim=0.5, use_occ_mask_at_scale=False):
    def one_scale(explainability_mask, occ_masks, flows):
        assert(explainability_mask is None or flows[0].size()[2:] == explainability_mask.size()[2:])
        assert(len(flows) == len(ref_imgs))

        reconstruction_loss = 0
        b, _, h, w = flows[0].size()
        downscale = tgt_img.size(2)/h

        tgt_img_scaled = nn.functional.adaptive_avg_pool2d(tgt_img, (h, w))
        ref_imgs_scaled = [nn.functional.adaptive_avg_pool2d(ref_img, (h, w)) for ref_img in ref_imgs]

        weight = 1.

        for i, ref_img in enumerate(ref_imgs_scaled):
            current_flow = flows[i]

            ref_img_warped = flow_warp(ref_img, current_flow)
            valid_pixels = 1 - (ref_img_warped == 0).prod(1, keepdim=True).type_as(ref_img_warped)
            diff = (tgt_img_scaled - ref_img_warped) * valid_pixels
            ssim_loss = 1 - ssim(tgt_img_scaled, ref_img_warped) * valid_pixels
            oob_normalization_const = valid_pixels.nelement()/valid_pixels.sum()

            if explainability_mask is not None:
                diff = diff * explainability_mask[:,i:i+1].expand_as(diff)
                ssim_loss = ssim_loss * explainability_mask[:,i:i+1].expand_as(ssim_loss)

            if occ_masks is not None:
                diff = diff *(1-occ_masks[:,i:i+1]).expand_as(diff)
                ssim_loss = ssim_loss*(1-occ_masks[:,i:i+1]).expand_as(ssim_loss)

            reconstruction_loss += (1- wssim)*weight*oob_normalization_const*(robust_l1(diff, q=qch) + wssim*ssim_loss.mean()) + lambda_oob*robust_l1(1 - valid_pixels, q=qch)
            #weight /= 2.83
            assert((reconstruction_loss == reconstruction_loss).item() == 1)

        return reconstruction_loss

    if type(flows[0]) not in [tuple, list]:
        if explainability_mask is not None:
            explainability_mask = [explainability_mask]
        flows = [[uv] for uv in flows]

    loss = 0
    for i in range(len(flows[0])):
        flow_at_scale = [uv[i] for uv in flows]
        occ_mask_at_scale_bw, occ_mask_at_scale_fw  = occlusion_masks(flow_at_scale[0], flow_at_scale[1])
        occ_mask_at_scale = torch.stack((occ_mask_at_scale_bw, occ_mask_at_scale_fw), dim=1)
        if use_occ_mask_at_scale == False:
            occ_mask_at_scale = None
        loss += one_scale(explainability_mask[i], occ_mask_at_scale, flow_at_scale)

    return loss


def consensus_depth_flow_mask(explainability_mask, census_mask_bwd, census_mask_fwd, exp_masks_bwd_target, exp_masks_fwd_target, THRESH, wbce):
    # Loop over each scale
    assert(len(explainability_mask)==len(census_mask_bwd))
    assert(len(explainability_mask)==len(census_mask_fwd))
    loss = 0.
    for i in range(len(explainability_mask)):
        exp_mask_one_scale = explainability_mask[i]
        census_mask_fwd_one_scale = (census_mask_fwd[i] < THRESH).type_as(exp_mask_one_scale).prod(dim=1, keepdim=True)
        census_mask_bwd_one_scale = (census_mask_bwd[i] < THRESH).type_as(exp_mask_one_scale).prod(dim=1, keepdim=True)

        #Using the pixelwise consensus term
        exp_fwd_target_one_scale = exp_masks_fwd_target[i]
        exp_bwd_target_one_scale = exp_masks_bwd_target[i]
        census_mask_fwd_one_scale = logical_or(census_mask_fwd_one_scale, exp_fwd_target_one_scale)
        census_mask_bwd_one_scale = logical_or(census_mask_bwd_one_scale, exp_bwd_target_one_scale)

        # OR gate for constraining only rigid pixels
        # exp_mask_fwd_one_scale = (exp_mask_one_scale[:,2].unsqueeze(1) > 0.5).type_as(exp_mask_one_scale)
        # exp_mask_bwd_one_scale = (exp_mask_one_scale[:,1].unsqueeze(1) > 0.5).type_as(exp_mask_one_scale)
        # census_mask_fwd_one_scale = 1- (1-census_mask_fwd_one_scale)*(1-exp_mask_fwd_one_scale)
        # census_mask_bwd_one_scale = 1- (1-census_mask_bwd_one_scale)*(1-exp_mask_bwd_one_scale)

        census_mask_fwd_one_scale = Variable(census_mask_fwd_one_scale.data, requires_grad=False)
        census_mask_bwd_one_scale = Variable(census_mask_bwd_one_scale.data, requires_grad=False)

        rigidity_mask_combined = torch.cat((census_mask_bwd_one_scale, census_mask_bwd_one_scale,
                        census_mask_fwd_one_scale, census_mask_fwd_one_scale), dim=1)
        loss += weighted_binary_cross_entropy(exp_mask_one_scale, rigidity_mask_combined.type_as(exp_mask_one_scale), [wbce, 1-wbce])

    return loss



@torch.no_grad()
def compute_depth_errors(gt, pred, crop=True):
    abs_diff, abs_rel, sq_rel, a1, a2, a3 = 0,0,0,0,0,0
    batch_size = gt.size(0)

    '''
    crop used by Garg ECCV16 to reprocude Eigen NIPS14 results
    construct a mask of False values, with the same size as target
    and then set to True values inside the crop
    '''
    if crop:
        crop_mask = gt[0] != gt[0]
        y1,y2 = int(0.40810811 * gt.size(1)), int(0.99189189 * gt.size(1))
        x1,x2 = int(0.03594771 * gt.size(2)), int(0.96405229 * gt.size(2))
        crop_mask[y1:y2,x1:x2] = 1
    skipped = 0
    for current_gt, current_pred in zip(gt, pred):
        valid = (current_gt > 0) & (current_gt < 80)
        if crop:
            valid = valid & crop_mask
        if valid.sum() == 0:
            continue

        valid_gt = current_gt[valid]
        valid_pred = current_pred[valid].clamp(1e-3, 80)

        valid_pred = valid_pred * torch.median(valid_gt)/torch.median(valid_pred)

        thresh = torch.max((valid_gt / valid_pred), (valid_pred / valid_gt))
        a1 += (thresh < 1.25).float().mean()
        a2 += (thresh < 1.25 ** 2).float().mean()
        a3 += (thresh < 1.25 ** 3).float().mean()

        abs_diff += torch.mean(torch.abs(valid_gt - valid_pred))
        abs_rel += torch.mean(torch.abs(valid_gt - valid_pred) / valid_gt)

        sq_rel += torch.mean(((valid_gt - valid_pred)**2) / valid_gt)
    if skipped == batch_size:
        return None

    return [metric.item() / (batch_size - skipped) for metric in [abs_diff, abs_rel, sq_rel, a1, a2, a3]]


@torch.no_grad()
def compute_pose_errors(gt, pred):
    RE = 0
    for (current_gt, current_pred) in zip(gt, pred):
        snippet_length = current_gt.shape[0]
        scale_factor = torch.sum(current_gt[..., -1] * current_pred[..., -1]) / torch.sum(current_pred[..., -1] ** 2)
        ATE = torch.norm((current_gt[..., -1] - scale_factor * current_pred[..., -1]).reshape(-1)).cpu().numpy()
        R = current_gt[..., :3] @ current_pred[..., :3].transpose(-2, -1)
        for gt_pose, pred_pose in zip(current_gt, current_pred):
            # Residual matrix to which we compute angle's sin and cos
            R = (gt_pose[:, :3] @ torch.inverse(pred_pose[:, :3])).cpu().numpy()
            s = np.linalg.norm([R[0, 1]-R[1, 0],
                                R[1, 2]-R[2, 1],
                                R[0, 2]-R[2, 0]])
            c = np.trace(R) - 1
            # Note: we actually compute double of cos and sin, but arctan2 is invariant to scale
            RE += np.arctan2(s, c)

    return [ATE/snippet_length, RE/snippet_length]
