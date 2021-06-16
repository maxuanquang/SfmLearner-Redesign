# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

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
