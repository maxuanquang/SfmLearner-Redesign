import torch
from path import Path

class OptimizerCreator():
    def __init__(self, args):
        self.args = args

    def create(self, *networks):
        print('=> setting adam solver')

        optim_params = []
        for item in networks:
            optim_params.append({'params': item.parameters(), 'lr': self.args.lr})

        optimizer = torch.optim.Adam(optim_params,
                                    betas=(self.args.momentum, self.args.beta),
                                    weight_decay=self.args.weight_decay)

        if self.args.resume and (self.args.save_path/'optimizer_checkpoint.pth.tar').exists():
            print("=> resuming optimizer from checkpoint")
            optimizer_weights = torch.load(self.args.save_path/'optimizer_checkpoint.pth.tar')
            optimizer.load_state_dict(optimizer_weights['state_dict'])
        elif self.args.pretrained_optimizer and not self.args.resume:
            print("=> using pre-trained weights for Optimizer")
            optimizer_weights = torch.load(self.args.pretrained_optimizer)
            optimizer.load_state_dict(optimizer_weights['state_dict'])

        return optimizer