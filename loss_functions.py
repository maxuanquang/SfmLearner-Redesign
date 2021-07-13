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

def one_scale_reconstruction(tgt_img, ref_imgs, intrinsics, depth, explainability_mask, pose, args):
    assert(explainability_mask is None or depth.size()[2:] == explainability_mask.size()[2:])
    assert(pose.size(1) == len(ref_imgs))

    reconstruction_loss = 0
    scale_diff_loss = 0
    scale_ssim_loss = 0
    b, _, h, w = depth.size()
    downscale = tgt_img.size(2)/h

    tgt_img_scaled = F.interpolate(tgt_img, (h, w), mode='area')
    ref_imgs_scaled = [F.interpolate(ref_img, (h, w), mode='area') for ref_img in ref_imgs]
    intrinsics_scaled = torch.cat((intrinsics[:, 0:2]/downscale, intrinsics[:, 2:]), dim=1)

    warped_imgs = []
    diff_maps = []

    for i, ref_img in enumerate(ref_imgs_scaled):
        current_pose = pose[:, i]

        ref_img_warped, valid_points = inverse_warp(ref_img, depth[:,0], current_pose,
                                                    intrinsics_scaled,
                                                    args.rotation_mode, args.padding_mode)
        valid_pixels = valid_points.unsqueeze(1).float()
        diff = (tgt_img_scaled - ref_img_warped) * valid_pixels
        ssim_loss = 1 - ssim(tgt_img_scaled, ref_img_warped) * valid_pixels
        oob_normalization_const = valid_pixels.nelement()/valid_pixels.sum()

        if explainability_mask is not None:
            diff = diff * explainability_mask[:,i:i+1].expand_as(diff)
            ssim_loss = ssim_loss * explainability_mask[:,i:i+1].expand_as(ssim_loss)

        reconstruction_loss += args.L1_photometric_weight*oob_normalization_const*(l1(diff) + args.ssim_photometric_weight*ssim_loss.mean())
        scale_diff_loss += l1(diff)
        scale_ssim_loss += ssim_loss.mean()
        #weight /= 2.83
        assert((reconstruction_loss == reconstruction_loss).item() == 1)                

        warped_imgs.append(ref_img_warped[0])
        diff_maps.append(diff[0])

    return reconstruction_loss, warped_imgs, diff_maps, scale_diff_loss, scale_ssim_loss

def photometric_reconstruction_loss(tgt_img, ref_imgs, intrinsics, depth, explainability_mask, pose, args):

    if type(explainability_mask) not in [tuple, list]:
        explainability_mask = [explainability_mask]
    if type(depth) not in [list, tuple]:
        depth = [depth]

    total_loss = 0
    total_diff_loss = 0
    total_ssim_loss = 0
    for d, mask in zip(depth, explainability_mask):
        loss, _, _, diff_loss, ssim_loss = one_scale_reconstruction(tgt_img, ref_imgs, intrinsics, d, mask, pose, args)
        total_loss += loss
        total_diff_loss += diff_loss
        total_ssim_loss += ssim_loss
    return total_loss, total_diff_loss, total_ssim_loss

def photometric_reconstruction_results(tgt_img, ref_imgs, intrinsics, depth, explainability_mask, pose, args):

    warped_results, diff_results = [], []
    if type(explainability_mask) not in [tuple, list]:
        explainability_mask = [explainability_mask]
    if type(depth) not in [list, tuple]:
        depth = [depth]

    for d, mask in zip(depth, explainability_mask):
        _, warped, diff, _, _ = one_scale_reconstruction(tgt_img, ref_imgs, intrinsics, d, mask, pose, args)
        warped_results.append(warped)
        diff_results.append(diff)
    return warped_results, diff_results


def explainability_loss(mask):
    if type(mask) not in [tuple, list]:
        mask = [mask]
    loss = 0
    for mask_scaled in mask:
        ones_var = torch.ones_like(mask_scaled)
        loss += nn.functional.binary_cross_entropy(mask_scaled, ones_var)
    return loss


def smooth_loss(pred_map, args):
    def gradient(pred):
        D_dy = pred[:, :, 1:] - pred[:, :, :-1]
        D_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        return D_dx, D_dy

    if type(pred_map) not in [tuple, list]:
        pred_map = [pred_map]

    loss = 0
    weight = 1.

    if args.use_second_derivative and args.use_L1_smooth:
        for scaled_map in pred_map:
            dx, dy = gradient(scaled_map)
            dx2, dxdy = gradient(dx)
            dydx, dy2 = gradient(dy)
            loss += (dx2.abs().mean() + dxdy.abs().mean() + dydx.abs().mean() + dy2.abs().mean())*weight
            weight /= 2.3  # don't ask me why it works better
        return loss
    elif args.use_second_derivative and args.use_L2_smooth:
        for scaled_map in pred_map:
            dx, dy = gradient(scaled_map)
            dx2, dxdy = gradient(dx)
            dydx, dy2 = gradient(dy)
            loss += (l2(dx2) + l2(dxdy) + l2(dydx) + l2(dy2))*weight
            weight /= 2.3  # don't ask me why it works better
        return loss
    elif args.use_first_derivative and args.use_L1_smooth:
        for scaled_map in pred_map:
            dx, dy = gradient(scaled_map)
            loss += (dx.abs().mean() + dy.abs().mean())*weight
            weight /= 2.3  # don't ask me why it works better
        return loss
    elif args.use_first_derivative and args.use_L2_smooth:
        for scaled_map in pred_map:
            dx, dy = gradient(scaled_map)
            loss += (l2(dx) + l2(dy))*weight
            weight /= 2.3  # don't ask me why it works better
        return loss

def edge_aware_smoothness_loss(img, pred_disp):
    def gradient_x(img):
        gx = img[:,:,:-1,:] - img[:,:,1:,:]
        return gx

    def gradient_y(img):
        gy = img[:,:,:,:-1] - img[:,:,:,1:]
        return gy

    def get_edge_smoothness(img, pred):
        pred_gradients_x = gradient_x(pred)
        pred_gradients_y = gradient_y(pred)

        image_gradients_x = gradient_x(img)
        image_gradients_y = gradient_y(img)

        weights_x = torch.exp(-torch.mean(torch.abs(image_gradients_x), 1, keepdim=True))
        weights_y = torch.exp(-torch.mean(torch.abs(image_gradients_y), 1, keepdim=True))

        smoothness_x = torch.abs(pred_gradients_x) * weights_x
        smoothness_y = torch.abs(pred_gradients_y) * weights_y
        return torch.mean(smoothness_x) + torch.mean(smoothness_y)

    loss = 0
    weight = 1.

    for scaled_disp in pred_disp:
        b, _, h, w = scaled_disp.size()
        scaled_img = nn.functional.adaptive_avg_pool2d(img, (h, w))
        loss += get_edge_smoothness(scaled_img, scaled_disp)
        weight /= 2.3   # 2sqrt(2)

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
