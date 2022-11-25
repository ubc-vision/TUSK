# TUSK: Task-Agnostic Unsupervised Keypoints
### Yuhe Jin, Weiwei Sun, Jan Hosang, Eduard Trulls, Kwang Moo Yi
This repository contains training and inference code for [TUSK: Task-Agnostic Unsupervised Keypoints](https://arxiv.org/abs/2206.08460).

![alt text](https://github.com/ubc-vision/TUSK/blob/main/images/mnist_results.png)
## Installation
This code is implemented based on PyTorch. A conda environment is provided with all the dependencies:
```
conda env create -f system/tusk.yml
```
## Pretrained models and datasets
We provide pretrained model for [MNIST-hard](https://www.cs.ubc.ca/research/kmyi_data/files/2021/mist/mnist_hard.zip), [CLEVR](https://github.com/deepmind/multi_object_datasets#clevr-with-masks), [Tetrominoes](https://github.com/deepmind/multi_object_datasets#tetrominoes), [CelebA-in-the-wild](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html), [Human3.6M](http://vision.imar.ro/human3.6m/description.php). We also included two pretrained linear projection to evaluate on [MAFL (a subset of CelebA-in-the-wild)](https://github.com/zhzhanp/TCDCN-face-alignment/tree/master/MAFL) and [Human3.6M](http://vision.imar.ro/human3.6m/description.php).

Download Link for pretrained model: 
`https://drive.google.com/file/d/1hplP1zr64sKXZXLvdHtsujZlPkc8oC2i/view?usp=sharing`
## Training
Config files are included in the `./configs` folder.

Following command will train model on MNIST dataset
```
python train.py --yaml_path=configs/mnist.yaml
```

Training on CelebA-in-the-wild dataset requires pretrained VGG16 weights, run `./utils/download_vgg.py` to download the weights.

## Inference and Evaluation
Following commands will run pretrained model on test set and training set. Visualization can be found in `./results`
```
python inference_celeba_h36m_mnist.py --yaml_path=./configs/h36m.yaml
python inference_multi_objects.py --mode=test --yaml_path=./configs/clevr.yaml
```

Evaluation metrics for CLEVR, and Tetrominoes will be shown in the terminal when running the inference code. For MAFL and Human3.6M, pretrained linear projection need to be downloaded and stored as `./results/H36M/h36m/LinearProjection_model` and `./results/MAFL/celeba/LinearProjection_model`, then run the following command
```
python evaluate_celeba_h36m.py --mode=test --yaml_path=./configs/celeba.yaml
python evaluate_celeba_h36m.py --mode=test --yaml_path=./configs/h36m.yaml
```
For MNIST-hard, run following command to evaluate the result
```
python evaluate_mnist.py --mode=test --yaml_path=./configs/mnist.yaml
```
## Citation
```
@InProceedings{jin2022tusk,
title = {TUSK: Task-Agnostic Unsupervised Keypoints},
author = {Yuhe Jin, Weiwei Sun, Jan Hosang, Eduard Trulls, Kwang Moo Yi},
booktitle = {Neural Information Processing Systems},
year = {2022}}
```
