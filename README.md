# SfMLearner Pytorch Software Architecture Redesign version
This codebase implements the system described in the paper: Unsupervised Learning of Depth and Ego-Motion from Video

Original Author : Tinghui Zhou (tinghuiz@berkeley.edu)
Pytorch implementation : Clément Pinard (clement.pinard@ensta-paristech.fr)

![sample_results](misc/cityscapes_sample_results.gif)

## Prerequisite

```bash
pip3 install -r requirements.txt
```
### What has been done

* Train disparity network, pose network easily with different architectures, different losses and different strategies.

## Preparing training data
Preparation is roughly the same command as in the original code.

For [KITTI](http://www.cvlibs.net/datasets/kitti/raw_data.php), first download the dataset using this [script](http://www.cvlibs.net/download.php?file=raw_data_downloader.zip) provided on the official website, and then run the following command. The `--with-depth` option will save resized copies of groundtruth to help you setting hyper parameters. The `--with-pose` will dump the sequence pose in the same format as Odometry dataset (see pose evaluation)
```bash
python3 data/prepare_train_data.py /path/to/raw/kitti/dataset/ \
--dataset-format 'kitti' \
--dump-root /path/to/resulting/formatted/data/ \
--width 416 --height 128 \
--num-threads 4 \
--static-frames /path/to/static_frames.txt \
--with-depth --with-pose
```

For [Cityscapes](https://www.cityscapes-dataset.com/), download the following packages: 1) `leftImg8bit_sequence_trainvaltest.zip`, 2) `camera_trainvaltest.zip`. You will probably need to contact the administrators to be able to get it. Then run the following command
```bash
python3 data/prepare_train_data.py /path/to/cityscapes/dataset/ \
--dataset-format 'cityscapes' \
--dump-root /path/to/resulting/formatted/data/ \
--width 416 --height 171 --num-threads 4
```
Notice that for Cityscapes the `img_height` is set to 171 because we crop out the bottom part of the image that contains the car logo, and the resulting image will have height 128.

## Training
Once the data are formatted following the above instructions, you should be able to train the model by running the following command
```bash
python runner.py --train \
--dispnet-architecture DispNetS --nlevels 4 \
--posenet-architecture PoseExpNet \
--dataset-dir /path/to/the/formatted/data/ \
-b 4 -m 0.2 -s 0.1 \
--epoch-size 0 --epochs 10 \
--name demo
```
You can train the models with pretrained models by running the following command
```bash
python runner.py --train \
--dispnet-architecture DispNetS --nlevels 4 \
--posenet-architecture PoseExpNet \
--pretrained-dispnet /path/to/pretrained/dispnet \
--pretrained-posenet /path/to/pretrained/posenet \
--dataset-dir /path/to/the/formatted/data/ \
-b 4 -m 0.2 -s 0.1 \
--epoch-size 0 --epochs 10 \
--name demo
```
You can kick off training from checkpoint by running the following command
```bash
python runner.py --train \
--dispnet-architecture DispNetS --nlevels 4 \
--posenet-architecture PoseExpNet \
--dataset-dir /path/to/the/formatted/data/ \
-b 4 -m 0.2 -s 0.1 \
--epoch-size 0 --epochs 10 \
--name demo --resume
```
You can then start a `tensorboard` session in this folder by
```bash
tensorboard --logdir=checkpoints/
```
and visualize the training progress by opening [https://localhost:6006](https://localhost:6006) on your browser. If everything is set up properly, you should start seeing reasonable depth prediction after ~30K iterations when training on KITTI.

## Evaluation

Disparity evaluation is avalaible
```bash
python runner.py --eval-dispnet \
--dispnet-architecture DispNetS \
--pretrained-dispnet /path/to/dispnet \
--dataset-dir /path/to/KITTI_raw \
--dataset-list /path/to/test_files_list
```
Odometry evaluation is avalaible

```bash
python3 runner.py --eval-posenet \
--posenet-architecture PoseExpNet \
--pretrained-posenet /path/to/posenet \
--dataset-dir /path/to/KITIT_odometry \
--sequences 09
```

**ATE** (*Absolute Trajectory Error*) is computed as long as **RE** for rotation (*Rotation Error*). **RE** between `R1` and `R2` is defined as the angle of `R1*R2^-1` when converted to axis/angle. It corresponds to `RE = arccos( (trace(R1 @ R2^-1) - 1) / 2)`.
While **ATE** is often said to be enough to trajectory estimation, **RE** seems important here as sequences are only `seq_length` frames long.

## Inference

Disparity map (or depth map) generation can be done with `runner.py`
```bash
python runner.py --infer \
--dispnet-architecture DispNetS \
--pretrained-dispnet /path/to/dispnet \
--dataset-dir /path/to/pictures/dir \
--output-dir /path/to/output/dir \
--output-disp \
--output-depth 
```
Will run inference on all pictures inside `dataset-dir` and save a jpg of disparity (or depth) to `output-dir`.

## Pretrained Nets

[Avalaible here](https://drive.google.com/drive/folders/1wVTJTP7OlBoEUqk24u_esYDcA2KXvVfW?usp=sharing)

Arguments used :

```bash
!python runner.py --train \
--dispnet-architecture DispResNet6 --nlevels 4 \
--posenet-architecture PoseExpNet \
--dataset-dir '/content/resulting_formatted_data_full' \
-b 4 -p 1.0 -m 0.0 \
--L1-photometric-weight 1.0 --ssim-photometric-weight 0.12 -s 0.21 \
--smoothness-type edgeaware \
--lr 1e-4 --epoch-size 3000 -f 500 --epochs 40 \
--name SSIM_0.36_L1_0.63_Smooth_0.01
```

### Depth Results

| Abs Rel | Acc.1 |
|---------|-------|
| 0.176   | 0.768 |

### Pose Results

5-frames snippets used

|    | Seq. 09|
|----|--------|
|ATE | 0.0128 |
|RE  | 0.0021 |

