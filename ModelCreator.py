import models
import torch
import torch.backends.cudnn as cudnn

class ModelCreator():
    def __init__(self, args):
        self.args = args
    def create(self, model='dispnet'):
        # create disp network
        if model == 'dispnet':
            print("=> creating disparity network")

            if self.args.dispnet_architecture == 'DispNetS':
                disp_net = models.DispNetS().to(torch.device("cuda"))
            elif self.args.dispnet_architecture == 'DispResNet4':
                disp_net = models.DispResNet4().to(torch.device("cuda"))
            elif self.args.dispnet_architecture == 'DispResNetS4':
                disp_net = models.DispResNetS4().to(torch.device("cuda"))
            elif self.args.dispnet_architecture == 'DispNetS6':
                disp_net = models.DispNetS6().to(torch.device("cuda"))
            elif self.args.dispnet_architecture == 'DispResNet6':
                disp_net = models.DispResNet6().to(torch.device("cuda"))
            elif self.args.dispnet_architecture == 'DispResNetS6':
                disp_net = models.DispResNetS6().to(torch.device("cuda"))

            if self.args.pretrained_dispnet and not self.args.resume:
                print("=> using pre-trained weights for Dispnet")
                weights = torch.load(self.args.pretrained_dispnet)
                disp_net.load_state_dict(weights['state_dict'])
            elif self.args.resume:
                print("=> resuming Dispnet from checkpoint")
                weights = torch.load(self.args.save_path/'dispnet_checkpoint.pth.tar')
                disp_net.load_state_dict(weights['state_dict'])
            else:
                print("=> intializing Dispnet weights")
                disp_net.init_weights()

            cudnn.benchmark = True
            disp_net = torch.nn.DataParallel(disp_net)

            return disp_net

        # create poseexp model
        elif model == 'poseexpnet':
            print("=> creating explainability pose network")

            output_exp = self.args.mask_loss_weight > 0
            if not output_exp:
                print("=> no mask loss, PoseExpnet will only output pose")
            pose_exp_net = models.PoseExpNet(nb_ref_imgs=self.args.sequence_length - 1, output_exp=self.args.mask_loss_weight > 0).to(torch.device("cuda"))

            if self.args.pretrained_pose_exp and not self.args.resume:
                print("=> using pre-trained weights for explainabilty and pose net")
                weights = torch.load(self.args.pretrained_pose_exp)
                pose_exp_net.load_state_dict(weights['state_dict'], strict=False)
            elif self.args.resume:
                print("=> resuming Exp_pose_net from checkpoint")
                weights = torch.load(self.args.save_path/'pose_exp_checkpoint.pth.tar')
                pose_exp_net.load_state_dict(weights['state_dict'])
            else:
                print('=> initializing Pose-exp-net weights')
                pose_exp_net.init_weights()

            cudnn.benchmark = True
            pose_exp_net = torch.nn.DataParallel(pose_exp_net)

            return pose_exp_net

        # create pose network
        elif model == 'posenet':
            print("=> creating pose network")

            if self.args.posenet_architecture == 'PoseNet6':
                pose_net = models.PoseNet6(nb_ref_imgs=self.args.sequence_length - 1).to(torch.device("cuda"))
            elif self.args.posenet_architecture == 'PoseNetB6':
                pose_net = models.PoseNetB6(nb_ref_imgs=self.args.sequence_length - 1).to(torch.device("cuda"))

            if self.args.pretrained_posenet and not self.args.resume:
                print("=> using pre-trained weights for explainabilty and pose net")
                weights = torch.load(self.args.pretrained_posenet)
                pose_net.load_state_dict(weights['state_dict'], strict=False)
            elif self.args.resume:
                print("=> resuming posenet from checkpoint")
                weights = torch.load(self.args.save_path/'posenet_checkpoint.pth.tar')
                pose_net.load_state_dict(weights['state_dict'])
            else:
                print('=> initializing Posenet weights')
                pose_net.init_weights()

            cudnn.benchmark = True
            pose_net = torch.nn.DataParallel(pose_net)

            return pose_net

        # create mask network
        elif model == 'masknet':
            pass

        # create flow network
        elif model == 'flownet':
            pass