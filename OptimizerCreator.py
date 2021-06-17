import torch

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

        return optimizer