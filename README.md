# About PruneTrain

This repository contains the source codes and scripts for training a set of CNNs using PruneTrian, continuous structured model pruning and network architecture reconfiguration for faster neural network training (https://arxiv.org/pdf/1901.09290.pdf). We provide the code base to train ResNet and VGG models with different number of layers for CIFAR10, CIFAR100, and ResNet50 model for ImageNet.

We flatten CNN architectures (each layer module definition) to support easy network architecture reconfiguration and generation into python files, where we use metadata programming. After pruning the CNN model using group lasso regularization, each layer has different channel dimensions. To store layers each with different channel dimensions, it is rather convenient to flatten the network layer structure than building with nested loops. It is possible to modify the module status without generating as a file. However, we do this for simpler pruning history track with intermediate checkpoints (model file and network file).

# Training Results

We also provide the pruned models and network architecture file.

| Dataset        | Model           | Removed Training FLOPs | Removed Inference FLOPs  | Top1 Error | Model | Network architecture |
|:--------------:|:---------------:|:----------------------:|:------------------------:|:----------:|:-----:|:--------------------:|
| /2. CIFAR10        | ResNet32        | XX%                    |   XX%                    |            |       |                      |
                 | ResNet50        | XX%                    |   XX%                    |            |       |                      |
| ^              | ResNet50        | XX%                    |   XX%                    |            |       |                      |
| CIFAR100       | ResNet32        | XX%                    |   XX%                    |            |       |                      |
| ^              | ResNet50        | XX%                    |   XX%                    |            |       |                      |
| ^              | ResNet50        | XX%                    |   XX%                    |            |       |                      |
| ImageNet       | ResNet50        | XX%                    |   XX%                    |            |       |                      |
| ^              | ResNet50        | XX%                    |   XX%                    |            |       |                      |


# Training Examples

We train using a mini-batch size of 128 distributed to 4 GPUs as the baseline. Please increase the learning rate to 0.1 when using mini-batch size 256.

