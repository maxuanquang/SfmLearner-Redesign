from operator import pos
import models
import torch
import torch.backends.cudnn as cudnn

class ModelCreator():
    def __init__(self, args):
        self.args = args
    def create(self):
        # create model
        print("=> creating model")
        device = torch.device("cuda")

        disp_net = models.DispNetS().to(torch.device("cuda"))
        output_exp = self.args.mask_loss_weight > 0
        if not output_exp:
            print("=> no mask loss, PoseExpnet will only output pose")
        pose_exp_net = models.PoseExpNet(nb_ref_imgs=self.args.sequence_length - 1, output_exp=self.args.mask_loss_weight > 0).to(device)

        if self.args.pretrained_exp_pose != "None":
            print("=> using pre-trained weights for explainabilty and pose net")
            weights = torch.load(self.args.pretrained_exp_pose)
            pose_exp_net.load_state_dict(weights['state_dict'], strict=False)
        else:
            pose_exp_net.init_weights()

        if self.args.pretrained_disp != "None":
            print("=> using pre-trained weights for Dispnet")
            weights = torch.load(self.args.pretrained_disp)
            disp_net.load_state_dict(weights['state_dict'])
        else:
            disp_net.init_weights()

        cudnn.benchmark = True
        disp_net = torch.nn.DataParallel(disp_net)
        pose_exp_net = torch.nn.DataParallel(pose_exp_net)

        return disp_net, pose_exp_net