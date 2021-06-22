from numpy.core.numeric import normalize_axis_tuple
import torch
import custom_transforms
from path import Path

class DataLoaderCreator():
    def __init__(self, args):
        self.args = args
    def create(self, mode):
        if self.args.dataset_format == 'stacked':
            from datasets.stacked_sequence_folders import SequenceFolder
        elif self.args.dataset_format == 'sequential':
            from datasets.sequence_folders import SequenceFolder

        if mode == 'train':
            # Data loading code
            train_transform = custom_transforms.Compose([
                custom_transforms.RandomHorizontalFlip(),
                custom_transforms.RandomScaleCrop(),
                custom_transforms.ArrayToTensor(),
                custom_transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                            std=[0.5, 0.5, 0.5])
            ])

            print("=> fetching scenes in '{}'".format(self.args.data))
            train_set = SequenceFolder(
                self.args.data,
                transform=train_transform,
                seed=self.args.seed,
                train=True,
                sequence_length=self.args.sequence_length
            )

            print('{} samples found in {} train scenes'.format(len(train_set), len(train_set.scenes)))

            train_loader = torch.utils.data.DataLoader(
                train_set, batch_size=self.args.batch_size, shuffle=True,
                num_workers=self.args.workers, pin_memory=True)

            return train_loader

        elif mode == "val":
            # Data loading code
            normalize = custom_transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        
            valid_transform = custom_transforms.Compose([custom_transforms.ArrayToTensor(), normalize])

            # if no Groundtruth is avalaible, Validation set is the same type as training set to measure photometric loss from warping
            if self.args.with_gt:
                if self.args.with_pose:
                    from datasets.validation_folders import ValidationSetWithPose
                    val_set = ValidationSetWithPose(
                        self.args.data,
                        sequence_length=self.args.sequence_length,
                        transform=valid_transform)
                else:
                    from datasets.validation_folders import ValidationSet
                    val_set = ValidationSet(
                        self.args.data,
                        transform=valid_transform
                    )
            else:
                val_set = SequenceFolder(
                    self.args.data,
                    transform=valid_transform,
                    seed=self.args.seed,
                    train=False,
                    sequence_length=self.args.sequence_length,
                )

            print('{} samples found in {} valid scenes'.format(len(val_set), len(val_set.scenes)))

            val_loader = torch.utils.data.DataLoader(
                val_set, batch_size=self.args.batch_size, shuffle=False,
                num_workers=self.args.workers, pin_memory=True)

            return val_loader

        elif mode == 'test_eigen':
            if self.args.gt_type == 'KITTI':
                from kitti_eval.depth_evaluation_utils import test_framework_KITTI as test_framework
            elif self.args.gt_type == 'stillbox':
                from stillbox_eval.depth_evaluation_utils import test_framework_stillbox as test_framework
            
            dataset_dir = Path(self.args.dataset_dir)

            test_files = [file.relpathto(dataset_dir) for file in sum([dataset_dir.files('*.{}'.format(ext)) for ext in self.args.img_exts], [])]

            framework = test_framework(dataset_dir, test_files, 1,
                            self.args.min_depth, self.args.max_depth)

            return framework