# TF2DeepFloorplan [![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)]
This repo contains a basic procedure to train and deploy the DNN model suggested by the paper ['Deep Floor Plan Recognition using a Multi-task Network with Room-boundary-Guided Attention'](https://arxiv.org/abs/1908.11025). It rewrites the original codes from [zlzeng/DeepFloorplan](https://github.com/zlzeng/DeepFloorplan) into newer versions of Tensorflow and Python.
<be>

![Furniture Predictions](https://github.com/eth-siplab-students/t-mt-2023-FloorplanReconstruction-AdrianaMohap/blob/master/assets/multi_task_generated_vs_non_generated.png)

#
## Requirements
Depends on different applications, the following installation methods can

|OS|Hardware|Application|Command|
|---|---|---|---|
|Ubuntu|CPU|Model Development|`pip install -e .[tfcpu,dev,testing,linting]`|
|Ubuntu|GPU|Model Development|`pip install -e .[tfgpu,dev,testing,linting]`|

## How to run?
1. Install packages.
```
# Option 1
python -m venv venv
source venv/bin/activate
pip install --upgrade pip setuptools wheel

# Option 2 (Preferred)
conda create -n venv python=3.8 cudatoolkit=10.1 cudnn=7.6.5
conda activate venv

# common install
pip install -e .[tfgpu,api,dev,testing,linting]
```
2. Create a Structured3D dataset and transform it to tfrecords `tf2deep.tfrecords`. 
3. Run the `train_furn.py` file  to initiate the training, model checkpoint is stored as `log/store/G` and weight is in `model/store`,
```
python -m dfp.train_furn [--batchsize 1][--lr 1e-4][--epochs 100]
[--logdir 'log/store'][--modeldir 'model/store']
[--save-tensor-interval 10][--save-model-interval 20]
[--tfmodel 'subclass'/'func'][--feature-channels 256 128 64 32]
[--backbone 'vgg16'/'mobilenetv1'/'mobilenetv2'/'resnet50']
[--feature-names block1_pool block2_pool block3_pool block4_pool block5_pool]
```
- for example,
```
python -m dfp.train_furn --batchsize=1 --lr=5e-4 --epochs=100
--logdir=log/store --modeldir=model/store
```
4. Run Tensorboard to view the progress of loss and images via,
```
tensorboard --logdir=log/store
```

