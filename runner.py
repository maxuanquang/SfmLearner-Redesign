# This repo is inspired from Monodepth2 and SfmLearner repositories

from __future__ import absolute_import, division, print_function

from SfmLeaner import SfmLearner
from config import SfmLearnerConfig

config = SfmLearnerConfig()
args = config.parse()

if __name__ == "__main__":
    sfmlearner = SfmLearner(args)
    if args.train:
        sfmlearner.train()
    elif args.test:
        sfmlearner.test()
    elif args.infer:
        sfmlearner.infer()
