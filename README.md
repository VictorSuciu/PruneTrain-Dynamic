# About PruneTrain

This repository contains the source codes and scripts for training a set of CNNs using PruneTrian, continuous structured model pruning and network architecture reconfiguration for faster neural network training (https://arxiv.org/pdf/1901.09290.pdf). We provide the code base to train ResNet and VGG models with different number of layers for CIFAR10, CIFAR100, and ResNet50 model for ImageNet.

We flatten CNN architectures (each layer module definition) to support easy network architecture reconfiguration and generation into python files, where we use metadata programming. After pruning the CNN model using group lasso regularization, each layer has different channel dimensions. To store layers each with different channel dimensions, it is rather convenient to flatten the network layer structure than building with nested loops. It is possible to modify the module status without generating as a file. However, we do this for simpler pruning history track with intermediate checkpoints (model file and network file).

# Training Results

We also provide the pruned models and network architecture file.

| Dataset        | Model           | Removed training FLOPs | Removed inference FLOPs  | Top1 error (fine-tuning) | Model | Network |
|----------------|:---------------:|:----------------------:|:------------------------:|:------------:|:-----:|:--------------------:|
| CIFAR10        | ResNet32        | 53%                    |   66%                    | 91.8%        |       |                      |
| CIFAR10        | ResNet50        | 50%                    |   70%                    | 93.1%        |       |                      |
| CIFAR100       | ResNet32        | 32%                    |   46%                    | 69.5%        |       |                      |
| CIFAR100       | ResNet50        | 53%                    |   69%                    | 72.4%        |       |                      |
| ImageNet       | ResNet50        | 37%                    |   53%                    | 74.4% (74.6%) |       |                      |
| ImageNet       | ResNet50        | 29%                    |   43%                    | 74.7% (75.2%) |       |                      |


# Training Examples

* Training ResNet32 on CIFAR10 with 1 GPU
```
python run-script.py --data-path /path/to/dataset --dataset cifar10 --model resnet32 --num-gpus 1
```

* Training VGG11 on CIFAR100 with 2 GPU
```
python run-script.py --data-path /path/to/dataset --dataset cifar100 --model vgg11 --num-gpus 2
```

* Training ResNet50 on ImageNet with 4 GPU
```
python run-script.py --data-path /path/to/dataset --dataset imagenet --model resnet50 --num-gpus 4
```

* Training ResNet50 on ImageNet with 4 GPU and regularization penalty ratio of 0.3
```
python run-script.py --data-path /path/to/dataset --dataset imagenet --model resnet50 --num-gpus 4 --penalty-ratio 0.3
```