# TF2DeepFloorplan [![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)]
This repo contains a basic procedure to train and deploy the DNN model suggested by the paper ['Deep Floor Plan Recognition using a Multi-task Network with Room-boundary-Guided Attention'](https://arxiv.org/abs/1908.11025). It rewrites the original codes from [zlzeng/DeepFloorplan](https://github.com/zlzeng/DeepFloorplan) into newer versions of Tensorflow and Python.
<br>


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
2. According to the original repo, please download r3d dataset and transform it to tfrecords `r3d.tfrecords`. 
3. Run the `train.py` file  to initiate the training, model checkpoint is stored as `log/store/G` and weight is in `model/store`,
```
python -m dfp.train [--batchsize 2][--lr 1e-4][--epochs 1000]
[--logdir 'log/store'][--modeldir 'model/store']
[--save-tensor-interval 10][--save-model-interval 20]
[--tfmodel 'subclass'/'func'][--feature-channels 256 128 64 32]
[--backbone 'vgg16'/'mobilenetv1'/'mobilenetv2'/'resnet50']
[--feature-names block1_pool block2_pool block3_pool block4_pool block5_pool]
```
- for example,
```
python -m dfp.train --batchsize=4 --lr=5e-4 --epochs=100
--logdir=log/store --modeldir=model/store
```
4. Run Tensorboard to view the progress of loss and images via,
```
tensorboard --logdir=log/store
```

```
7. Deploy the model via `deploy.py`, please be aware that load method parameter should match with weight input.
```
python -m dfp.deploy [--image 'path/to/image']
[--postprocess][--colorize][--save 'path/to/output_image']
[--loadmethod 'log'/'pb'/'tflite']
[--weight 'log/store/G'/'model/store'/'model/store/model.tflite']
[--tfmodel 'subclass'/'func']
[--feature-channels 256 128 64 32]
[--backbone 'vgg16'/'mobilenetv1'/'mobilenetv2'/'resnet50']
[--feature-names block1_pool block2_pool block3_pool block4_pool block5_pool]
```
- for example,
```
python -m dfp.deploy --image floorplan.jpg --weight log/store/G
--postprocess --colorize --save output.jpg --loadmethod log
```

## Results
- From `train.py` and `tensorboard`.

|Compare Ground Truth (top)<br> against Outputs (bottom)|Total Loss|
|:-------------------------:|:-------------------------:|
|<img src="resources/epoch60.png" width="400">|<img src="resources/Loss.png" width="400">|
|Boundary Loss|Room Loss|
|<img src="resources/LossB.png" width="400">|<img src="resources/LossR.png" width="400">|

- From `deploy.py` and `utils/legend.py`.

|Input|Legend|Output|
|:-------------------------:|:-------------------------:|:-------------------------:|
|<img src="resources/30939153.jpg" width="250">|<img src="resources/legend.png" width="180">|<img src="resources/output.jpg" width="250">|
|`--colorize`|`--postprocess`|`--colorize`<br>`--postprocess`|
|<img src="resources/color.jpg" width="250">|<img src="resources/post.jpg" width="250">|<img src="resources/postcolor.jpg" width="250">|

## Optimization
- Backbone Comparison in Size

|Backbone|log|pb|tflite|toml|
|---|---|---|---|---|
|VGG16|130.5Mb|119Mb|45.3Mb|[link](docs/experiments/vgg16/exp1)|
|MobileNetV1|102.1Mb|86.7Mb|50.2Mb|[link](docs/experiments/mobilenetv1/exp1)|
|MobileNetV2|129.3Mb|94.4Mb|57.9Mb|[link](docs/experiments/mobilenetv2/exp1)|
|ResNet50|214Mb|216Mb|107.2Mb|[link](docs/experiments/resnet50/exp1)|

- Feature Selection Comparison in Size

|Backbone|Feature Names|log|pb|tflite|toml|
|---|---|---|---|---|---|
|MobileNetV1|"conv_pw_1_relu", <br>"conv_pw_3_relu", <br>"conv_pw_5_relu", <br>"conv_pw_7_relu", <br>"conv_pw_13_relu"|102.1Mb|86.7Mb|50.2Mb|[link](docs/experiments/mobilenetv1/exp1)|
|MobileNetV1|"conv_pw_1_relu", <br>"conv_pw_3_relu", <br>"conv_pw_5_relu", <br>"conv_pw_7_relu", <br>"conv_pw_12_relu"|84.5Mb|82.3Mb|49.2Mb|[link](docs/experiments/mobilenetv1/exp2)|

- Feature Channels Comparison in Size

|Backbone|Channels|log|pb|tflite|toml|
|---|---|---|---|---|---|
|VGG16|[256,128,64,32]|130.5Mb|119Mb|45.3Mb|[link](docs/experiments/vgg16/exp1)|
|VGG16|[128,64,32,16]|82.4Mb|81.6Mb|27.3Mb||
|VGG16|[32,32,32,32]|73.2Mb|67.5Mb|18.1Mb|[link](docs/experiments/vgg16/exp2)|

- tfmot
  - Pruning (not working)
  - Clustering (not working)
  - Post training Quantization (work the best)
  - Training aware Quantization (not supported by the version)
