import torch
import csv


class OptimizerCreator():
    def __init__(self, args):
        self.args = args

    def create(self, disp_net, pose_exp_net):
        print('=> setting adam solver')
        print(type(self.args.lr))
        optim_params = [
            {'params': disp_net.parameters(), 'lr': self.args.lr},
            {'params': pose_exp_net.parameters(), 'lr': self.args.lr}
        ]
        optimizer = torch.optim.Adam(optim_params,
                                    betas=(self.args.momentum, self.args.beta),
                                    weight_decay=self.args.weight_decay)

        return optimizer