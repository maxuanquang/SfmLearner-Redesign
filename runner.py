import argparse
from .SfmLeaner import SfmLearner


parser = argparse.ArgumentParser(description='Structure from Motion Learner training on KITTI and CityScapes Dataset',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--train', action='store_true')
parser.add_argument('--test', action='store_true')
parser.add_argument('--infer', action='store_true')

parser.add_argument('--config-path', dest='config_path', type=str, default='config', required=True)
parser.add_argument('--epochs', type=int, default=1, required=True)
parser.add_argument('--epoch-size', type=int, default=100, required=True)
parser.add_argument('--name', dest='name', type=str, default='demo', required=False)

def main():
    global best_error, n_iter, device
    args = parser.parse_args()
    sfmlearner = SfmLearner(args)

    if args.train:
        sfmlearner.train()
    elif args.test:
        sfmlearner.test()
    elif args.infer:
        sfmlearner.infer()
    elif args.prepare_dataset:
        sfmlearner.prepare_dataset()

if __name__ == '__main__':
    main()