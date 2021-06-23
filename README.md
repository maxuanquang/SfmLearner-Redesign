# SfMLearner Pytorch Software Architecture Redesign version
This codebase implements the system described in the paper:

Unsupervised Learning of Depth and Ego-Motion from Video

[Tinghui Zhou](https://people.eecs.berkeley.edu/~tinghuiz/), [Matthew Brown](http://matthewalunbrown.com/research/research.html), [Noah Snavely](http://www.cs.cornell.edu/~snavely/), [David G. Lowe](http://www.cs.ubc.ca/~lowe/home.html)

In CVPR 2017 (**Oral**).

See the [project webpage](https://people.eecs.berkeley.edu/~tinghuiz/projects/SfMLearner/) for more details. 

Original Author : Tinghui Zhou (tinghuiz@berkeley.edu)
Pytorch implementation : Clément Pinard (clement.pinard@ensta-paristech.fr)

![sample_results](misc/cityscapes_sample_results.gif)

## Preamble
This codebase was developed and tested with Pytorch 1.0.1, CUDA 10 and Ubuntu 16.04. Original code was developped in tensorflow, you can access it [here](https://github.com/tinghuiz/SfMLearner)

## Prerequisite

```bash
pip3 install -r requirements.txt
```

or install manually the following packages :

```
pytorch >= 1.4.0
pebble
matplotlib
imageio
scipy==1.1.0
argparse
tensorboardX
blessings
progressbar2
path.py
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
--dispnet-architecture PoseExpNet \
--pretrained-dispnet /path/to/dispnet \
--dataset-dir /path/to/KITTI_raw \
--dataset-list /path/to/test_files_list
```

Test file list is available in kitti eval folder. To get fair comparison with [Original paper evaluation code](https://github.com/tinghuiz/SfMLearner/blob/master/kitti_eval/eval_depth.py), don't specify a posenet.

Pose evaluation is also available on [Odometry dataset](http://www.cvlibs.net/datasets/kitti/eval_odometry.php). Be sure to download both color images and pose !

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
--pretrained-dispnet /path/to/dispnet \
--dataset-dir /path/to/pictures/dir \
--output-dir /path/to/output/dir \
--output-disp \
--output-depth 
```
Will run inference on all pictures inside `dataset-dir` and save a jpg of disparity (or depth) to `output-dir` for each one see script help (`-h`) for more options.

## Pretrained Nets

[Avalaible here](https://drive.google.com/drive/folders/1H1AFqSS8wr_YzwG2xWwAQHTfXN5Moxmx)

Arguments used :

```bash
python3 runner.py --train \
--dispnet-architecture DispNetS \
--posenet-architecture PoseExpNet \
--dataset-dir /path/to/the/formatted/data/ \
-b4 -m0 -s2.0 \
--epoch-size 1000 epochs 200 \
--sequence-length 5 \
--log-output \
--with-gt
```

### Depth Results

| Abs Rel | Sq Rel | RMSE  | RMSE(log) | Acc.1 | Acc.2 | Acc.3 |
|---------|--------|-------|-----------|-------|-------|-------|
| 0.181   | 1.341  | 6.236 | 0.262     | 0.733 | 0.901 | 0.964 | 

### Pose Results

5-frames snippets used

|    | Seq. 09              | Seq. 10              |
|----|----------------------|----------------------|
|ATE | 0.0179 (std. 0.0110) | 0.0141 (std. 0.0115) |
|RE  | 0.0018 (std. 0.0009) | 0.0018 (std. 0.0011) | 


## Other Implementations

[TensorFlow](https://github.com/tinghuiz/SfMLearner) by tinghuiz (original code, and paper author)
