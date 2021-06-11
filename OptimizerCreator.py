import torch
import csv


class OptimizerCreator():
    def __init__(self, args):
        self.args = args

    def create(self, disp_net, pose_exp_net):
        print('=> setting adam solver')

        optim_params = [
            {'params': disp_net.parameters(), 'lr': self.args.lr},
            {'params': pose_exp_net.parameters(), 'lr': self.args.lr}
        ]
        optimizer = torch.optim.Adam(optim_params,
                                    betas=(self.args.momentum, self.args.beta),
                                    weight_decay=self.args.weight_decay)

        with open(self.args.save_path/self.args.log_summary, 'w') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t')
            writer.writerow(['train_loss', 'validation_loss'])

        with open(self.args.save_path/self.args.log_full, 'w') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t')
            writer.writerow(['train_loss', 'photo_loss', 'explainability_loss', 'smooth_loss'])

        return optimizer