from loss_functions import photometric_reconstruction_loss, explainability_loss, photometric_reconstruction_results, smooth_loss#, photometric_flow_loss, consensus_depth_flow_mask
from loss_functions import edge_aware_smoothness_loss
class SfmLearnerLoss():
    def __init__(self, args):
        self.args = args

    def calculate_loss(self, tgt_img, ref_imgs, intrinsics, depth, explainability_mask, pose):

        w1 = self.args.photo_loss_weight
        w2 = self.args.mask_loss_weight
        w3 = self.args.smooth_loss_weight
        # w4 = self.args.photometric_flow_loss_weight
        # w5 = self.args.consensus_depth_flow_loss_weight

        if w1 > 0:
            loss_1 = photometric_reconstruction_loss(tgt_img, ref_imgs, intrinsics, depth, explainability_mask, pose, self.args)
        else:
            loss_1 = 0

        if w2 > 0: # When explainability_masks are None -> then w2 is already equal 0 (in init of SfmLearner) -> loss_2 = 0
            loss_2 = explainability_loss(explainability_mask)
        else:
            loss_2 = 0
            
        if w3 > 0:
            loss_3 = 0

            loss_3 = edge_aware_smoothness_loss(tgt_img, depth)
            # if self.args.depth_smooth_weight > 0:
            #     loss_3 += smooth_loss(depth, self.args)*self.args.depth_smooth_weight
            # if self.args.disp_smooth_weight > 0:
            #     disparities = [1/d for d in depth]
            #     loss_3 += smooth_loss(disparities, self.args)*self.args.disp_smooth_weight
            # # if self.args.use_flow_smooth:
            # #     loss_3 += smooth_loss(flow)*self.args.flow_smooth_weight
            # if self.args.mask_smooth_weight > 0:
            #     loss_3 += smooth_loss(explainability_mask)*self.args.mask_smooth_weight
        else:
            loss_3 = 0

        # if w4 > 0:
        #     loss_4 = 0
        #     # loss_4 = photometric_flow_loss()
        # else:
        #     loss_4 = 0

        # if w5 > 0:
        #     loss_5 = 0
        #     # loss_5 = consensus_depth_flow_mask()
        # else:
        #     loss_5 = 0
        
        loss = w1*loss_1 + w2*loss_2 + w3*loss_3# + w4*loss_4 + w5*loss_5

        losses_dict = {
            'total_loss': loss,
            'photometric_reconstruction_loss': loss_1,
            'explainability_loss': loss_2,
            'smooth_loss': loss_3,
            # 'photometric_flow_loss': loss_4,
            # 'consensus_depth_flow_loss': loss_5
        }

        return losses_dict
        

    def calculate_intermediate_results(self, tgt_img, ref_imgs, intrinsics, depth, explainability_mask, pose):

        w1 = self.args.photo_loss_weight
        w2 = self.args.mask_loss_weight
        w3 = self.args.smooth_loss_weight
        # w4 = self.args.photometric_flow_loss_weight
        # w5 = self.args.consensus_depth_flow_loss_weight

        if w1 > 0:
            warped_results, diff_results = photometric_reconstruction_results(tgt_img, ref_imgs, intrinsics, depth, explainability_mask, pose, self.args)
            photometric_reconstruction_results_dict = {}
            photometric_reconstruction_results_dict['photometric_reconstruction_warped'] = warped_results
            photometric_reconstruction_results_dict['photometric_reconstruction_diff'] = diff_results
            
        else:
            photometric_reconstruction_results_dict = {}

        # if w4 > 0:
        #     photometric_flow_results_dict = {}
        # else:
        #     photometric_flow_results_dict = {}

        # if w5 > 0:
        #     consensus_results_dict = {}
        # else:
        #     consensus_results_dict = {}
        
        results_dict = {}
        results_dict.update(photometric_reconstruction_results_dict)
        # results_dict.update(photometric_flow_results_dict)
        # results_dict.update(consensus_results_dict)

        return results_dict