from operator import pos
import models
import torch
import torch.backends.cudnn as cudnn

class ModelCreator():
    def __init__(self, args):
        self.args = args
    def create(self, model='dispnet'):
        # create model
        if model == 'dispnet':
            print("=> creating disparity network")

            if self.args.dispnet == 'DispNetS':
                disp_net = models.DispNetS().to(torch.device("cuda"))
            elif self.args.dispnet == 'DispResNet4':
                disp_net = models.DispResNet4().to(torch.device("cuda"))
            elif self.args.dispnet == 'DispResNetS4':
                disp_net = models.DispResNetS4().to(torch.device("cuda"))
            elif self.args.dispnet == 'DispNetS6':
                disp_net = models.DispNetS6().to(torch.device("cuda"))
            elif self.args.dispnet == 'DispResNet6':
                disp_net = models.DispResNet6().to(torch.device("cuda"))
            elif self.args.dispnet == 'DispResNetS6':
                disp_net = models.DispResNetS6().to(torch.device("cuda"))

            if self.args.pretrained_disp != "None":
                print("=> using pre-trained weights for Dispnet")
                weights = torch.load(self.args.pretrained_disp)
                disp_net.load_state_dict(weights['state_dict'])
            else:
                disp_net.init_weights()

            cudnn.benchmark = True
            disp_net = torch.nn.DataParallel(disp_net)

            return disp_net

        # create model
        if model == 'poseexpnet':
            print("=> creating explainability pose network")

            output_exp = self.args.mask_loss_weight > 0
            if not output_exp:
                print("=> no mask loss, PoseExpnet will only output pose")
            pose_exp_net = models.PoseExpNet(nb_ref_imgs=self.args.sequence_length - 1, output_exp=self.args.mask_loss_weight > 0).to(torch.device("cuda"))

            if self.args.pretrained_exp_pose != "None":
                print("=> using pre-trained weights for explainabilty and pose net")
                weights = torch.load(self.args.pretrained_exp_pose)
                pose_exp_net.load_state_dict(weights['state_dict'], strict=False)
            else:
                pose_exp_net.init_weights()

            cudnn.benchmark = True
            pose_exp_net = torch.nn.DataParallel(pose_exp_net)

            return pose_exp_net